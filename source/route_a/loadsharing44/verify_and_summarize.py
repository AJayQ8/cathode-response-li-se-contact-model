"""Independent field/readout/gate audit for the load-sharing sensitivity."""
from pathlib import Path
import argparse
import importlib.util
import json
import time
import numpy as np
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


probe=module('loadsharing_runner',HERE/'run_loadsharing.py')
prior=module('loadsharing_prior_audit',HERE.parent/'refine43/precision_v2/verify_and_summarize_v3.py')
read,save,sha=probe.read,probe.save,probe.sha


def validate(c,w,reused=False):
    t=c['task']
    if 'result' not in c:
        assert not c['available'];return dict(id=probe.identifier(t),available=False,error=c['error'])
    if not reused:
        assert c['manifest_sha256']==sha(HERE/'manifest.json')
        if t['solver']=='symmetric':
            d=c['reduction'];assert d['duplicate_factor']==4 and d['ode_cells']==t['n']//4 and d['full_grid']==t['n'] and d['discarded_symmetry_relative_error']<=1e-12
        if t['kind']=='paired':
            snapshot=read(HERE/'checkpoints'/(probe.identifier(t)+'.json'))
            assert snapshot['task']==t and snapshot['manifest_sha256']==c['manifest_sha256']
            assert snapshot['coordinate']=='g' and snapshot['stage']=='after_30_min_rest'
            assert snapshot['worlds']==c['result']['rows'][-1]['independent']['worlds']
            assert c['result']['loading_domain']==probe.loading_domain(w,t)
    row=prior.previous.audit_case(c,w)
    return dict(row,law=t.get('law','local'),reused=reused)


def known(cases,m,n):
    saved=probe.known_gate(m,n);cc=[c for c in cases if c['task']['kind']=='known' and not c['task'].get('control') and c['task']['n']==n]
    if len(cc)!=7 or not all(c['available'] for c in cc):return saved
    errors=[];differences=[]
    for c in cc:
        t=c['task'];coarse=probe.case(dict(t,n=64)) if n==128 else None
        for i,r in enumerate(c['result']['rows']):
            errors.append(abs(r['independent']['ratio']-r['observed_ratio']))
            if coarse:differences.append(abs(r['independent']['ratio']-coarse['result']['rows'][i]['independent']['ratio']))
    prior.close(max(errors),saved['maximum_absolute_error']);prior.close(max(differences,default=0.),saved['grid_difference'])
    assert saved['passed']==bool(max(errors)<=.03 and max(differences,default=0.)<=1e-4)
    return saved


def compare_closures(local,common):
    rows=[]
    for cf in common:
        lf=next(f for f in local if f['Q']==cf['Q'])
        for a,b in zip(lf['interactions'],cf['interactions']):
            assert a['stage']==b['stage'];difference=b['effect']-a['effect']
            error=sum(r['grid_difference']+r['coordinate_difference'] for r in [a,b])
            eligible=a['qualified'] and b['qualified'];resolved=bool(eligible and abs(difference)>error)
            rows.append(dict(Q=cf['Q'],stage=a['stage'],local_percentage_points=a['percentage_points'],common_percentage_points=b['percentage_points'],difference_percentage_points=100*difference,summed_error_percentage_points=100*error,endpoints_qualified=eligible,comparison_qualified=resolved,common_larger=bool(resolved and difference>0),sign_disagreement=bool(resolved and a['sign_resolved'] and b['sign_resolved'] and a['effect']*b['effect']<0)))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['controls','pilot','known64','known128','future64','all'],required=True)
    phase=parser.parse_args().phase;checkpoint_guard();tic=time.perf_counter();m=read(HERE/'manifest.json');w=m['witness']
    for p,h in m['sha256'].items():assert sha(ROOT/p)==h,p
    controls=read(HERE/'controls.json');assert controls['manifest_sha256']==sha(HERE/'manifest.json')
    assert len(controls['checks'])==48 and len(controls['uniform_checks'])==12
    for r in controls['checks']:
        passed=r['rhs_error']<=1e-10 and r['jacobian_error']<=1e-10 and r['finite_difference_error']<=1e-5 and max(r[k] for k in ['physical_rhs_error','ledger_error','load_error','current_error','mean_identity_error'])<=1e-10
        assert passed==r['passed']
    for r in controls['uniform_checks']:assert r['passed']==bool(r['rhs_error']<=1e-10)
    passed=all(r['passed'] for r in controls['checks']+controls['uniform_checks']);assert passed==controls['passed']
    if phase=='controls':
        out=dict(manifest_sha256=sha(HERE/'manifest.json'),auditor_sha256=sha(Path(__file__)),passed=passed,static_checks=48,uniform_checks=12,controls_sha256=sha(HERE/'controls.json'),wall_seconds=time.perf_counter()-tic)
        save(HERE/'controls_summary.json',out);print(json.dumps(out),flush=True);return
    cases=[];rows=[];skipped=[]
    for t in m['tasks']:
        if phase=='pilot' and probe.identifier(t) not in m['pilot_ids']:continue
        if phase=='known64' and not(t['kind']=='known' and t['n']==64):continue
        if phase=='known128' and t['kind']!='known':continue
        if phase=='future64' and t['kind']=='paired' and t['n']==128:continue
        path=HERE/'cases'/(probe.identifier(t)+'.json');allowed,reason=probe.eligibility(t,m)
        if not path.exists():
            assert not allowed,(t,'Eligible task missing');skipped.append(dict(task=t,reason=reason));continue
        assert allowed,(t,'Task ran without prerequisite');c=read(path);assert c['task']==t;cases.append(c);rows.append(validate(c,w));checkpoint_guard()
    pilot=[c for c in cases if probe.identifier(c['task']) in m['pilot_ids']];benchmark=None;admitted=False
    if len(pilot)==3 and all(c['available'] for c in pilot):
        local=next(c for c in pilot if c['task']['law']=='local');full=next(c for c in pilot if c['task']['law']=='common' and c['task']['solver']=='full');reduced=next(c for c in pilot if c['task']['law']=='common' and c['task']['solver']=='symmetric')
        reference=read(probe.local_reference(local['task']));validate(reference,w,True)
        regression=probe.base.regression(local,reference);direct=probe.base.regression(reduced,full)
        benchmark=dict(local_wrapper_regression=regression,common_full_vs_reduced=direct,full_seconds=full['wall_seconds'],reduced_seconds=reduced['wall_seconds'],reduced_over_full_time=reduced['wall_seconds']/full['wall_seconds'],one_thread_per_task=True)
        admitted=bool(passed and regression['passed'] and direct['passed'])
    references=[]
    for t in m['reference_tasks']:
        c=read(probe.local_reference(t));references.append(c);rows.append(validate(c,w,True))
    future=[];local_future=[];volume=[]
    for Q in [.3,.6]:
        cc=[c for c in cases if c['task']['kind']=='paired' and c['task']['Q']==Q]
        ll=[c for c in references if c['task']['Q']==Q];assert len(ll)==4
        f=prior.p_summary(ll,.25);f.update(Q=Q,mode=2,law='local',grid_pair=[64,128]);local_future.append(f)
        if len(cc)!=4 or not all('result' in c for c in cc):continue
        f=prior.p_summary(cc,.25);f.update(Q=Q,mode=2,law='common',grid_pair=[64,128]);future.append(f)
        if all(c['available'] for c in cc):volume.extend(dict(v,Q=Q) for v in prior.full_volume_checks(cc,w))
    for n in [64,128]:
        cc=[c for c in cases if c['task']['kind']=='paired' and c['task']['n']==n and 'result' in c]
        if cc:assert all(c['result']['initial_contact']==cc[0]['result']['initial_contact'] for c in cc)
    grid=64 if phase in ['pilot','known64'] else 128
    out=dict(phase=phase,manifest_sha256=sha(HERE/'manifest.json'),auditor_sha256=sha(Path(__file__)),audit_passed=True,pilot_admitted=admitted,pilot=benchmark,
        new_case_count=len(cases),reused_local_future_cases=len(references),case_audits=rows,skipped=skipped,all_cases_available=all(c['available'] for c in cases),
        known_gate=known(cases,m,grid),known_gate64=known(cases,m,64),known_gate128=known(cases,m,128),future=future,local_future=local_future,closure_comparisons=compare_closures(local_future,future),
        depth_comparisons=prior.previous.comparisons(future),interpretation=prior.interpretation(cases,future,w),full_volume_checks=volume,maxima=prior.audit.MAX,
        central_claim_changed=False,external_validation=False,old_failed_tests_preserved=True,model_form_sensitivity=True,wall_seconds=time.perf_counter()-tic)
    save(HERE/(phase+'_summary.json'),out)
    print(json.dumps({k:out[k] for k in ['phase','audit_passed','pilot_admitted','pilot','new_case_count','known_gate','maxima','wall_seconds']}),flush=True)


if __name__=='__main__':main()
