"""Fixed-equation temporal refinement; original stage43/44 files stay unchanged."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import time
import traceback
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from scipy.optimize import brentq
from scipy.special import expit
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

load=module('temporal_loadsharing',HERE.parent/'loadsharing44/run_loadsharing.py')
read,save,sha=load.read,load.save,load.sha
prior,spatial,model=load.prior,load.spatial,load.model
loading_domain,make_system=load.loading_domain,load.make_system
compress_state,expand_state=load.compress_state,load.expand_state

def identifier(t):return load.identifier(t)+'_temporal'

def reference_path(t):
    return load.HERE/'cases'/(load.identifier(t)+'.json') if t['law']=='common' else load.local_reference(t)

REPLACEMENTS = [('routes={}', 'routes={};timestep_records=[]'), ("rtol=1e-10 if coordinate=='w' else 1e-12,", "rtol=1e-11 if coordinate=='w' else 1e-13,"), ("atol=1e-12 if coordinate=='w' else 1e-14,first_step=1e-4,events=event)", "atol=1e-13 if coordinate=='w' else 1e-15,first_step=1e-4,max_step=.002,events=event)"), ('assert sol.success and np.all(np.isfinite(sol.y[:,-1]));y=sol.y[:,-1];stopped=bool(len(sol.t_events[0]));values=[]', 'assert sol.success and np.all(np.isfinite(sol.y[:,-1]));y=sol.y[:,-1];stopped=bool(len(sol.t_events[0]));values=[]\n            timestep_records.append(dict(coordinate=coordinate,stage=stage,accepted_steps=len(sol.t)-1,maximum_step_hours=float(np.max(np.diff(sol.t))),minimum_step_hours=float(np.min(np.diff(sol.t))),nfev=int(sol.nfev),njev=int(sol.njev),nlu=int(sol.nlu),elapsed_hours=float(sol.t[-1])))'), ("evaluations={k:r['evaluations'] for k,r in routes.items()},available=", "evaluations={k:r['evaluations'] for k,r in routes.items()},time_steps=timestep_records,available=")]

def finite_symmetric(w,t):
    assert t['mode']==2
    loading=loading_domain(w,t);assert loading['valid'],loading
    x,a,Rch,q0=prior.setup(w,t);n=t['n'];m=n//4;L=10**x[0];A=x[4]
    cosine=np.cos(2*np.pi*t['mode']*(np.arange(n)+.5)/n)
    def mismatch(shift):return spatial.resistance(expit(shift+t['epsilon']*cosine),x[3]*w['rstar'],A*(1-x[3])*w['rstar'],1.)/Rch-1
    shift=brentq(mismatch,-35.,35.,xtol=1e-12,rtol=1e-14);a0=expit(shift+t['epsilon']*cosine)
    assert abs(spatial.resistance(a0,x[3]*w['rstar'],A*(1-x[3])*w['rstar'],1.)/Rch-1)<=1e-9
    routes={};timestep_records=[]
    for coordinate in ['w','g']:
        start=np.r_[(a0/A)**2 if coordinate=='w' else L*(1-a0),np.zeros(2*n)]
        start=compress_state(start,n)
        y=np.r_[start,start];rows=[];calls=0
        for stage,duration,J in [('end_discharge',t['Q']/t['J'],t['J']),('after_30_min_rest',t['rest'],0.)]:
            systems=[make_system(x,w['rstar'],f,m,t['history'],q0 if J else q0+t['Q'],J,coordinate,law=t['law']) for f in [0.,1.]]
            def rhs(time,state):return np.r_[systems[0].rhs(time,state[:3*m]),systems[1].rhs(time,state[3*m:])]
            def jac(time,state):return block_diag(systems[0].jac(time,state[:3*m]),systems[1].jac(time,state[3*m:]))
            def event(time,state):
                parts=np.r_[state[:m],state[3*m:4*m]]
                return np.min(parts)-model.ZMIN**2 if coordinate=='w' else L*(1-A*model.ZMIN)-np.max(parts)
            event.terminal=True;event.direction=-1
            sol=solve_ivp(rhs,(0.,duration),y,jac=jac,method='Radau',rtol=1e-11 if coordinate=='w' else 1e-13,
                          atol=1e-13 if coordinate=='w' else 1e-15,first_step=1e-4,max_step=.002,events=event)
            assert sol.success and np.all(np.isfinite(sol.y[:,-1]));y=sol.y[:,-1];stopped=bool(len(sol.t_events[0]));values=[]
            timestep_records.append(dict(coordinate=coordinate,stage=stage,accepted_steps=len(sol.t)-1,maximum_step_hours=float(np.max(np.diff(sol.t))),minimum_step_hours=float(np.min(np.diff(sol.t))),nfev=int(sol.nfev),njev=int(sol.njev),nlu=int(sol.nlu),elapsed_hours=float(sol.t[-1])))
            for j,world in enumerate(['limit','resolved']):
                z=expand_state(y[j*3*m:(j+1)*3*m],n)
                correction=max(0.,float(np.max(z[:n])-1/A**2)) if coordinate=='w' else max(0.,float(-np.min(z[:n])))*2/(L*A*A)
                assert correction<=1e-7*max(1.,1/A**2)
                row=spatial.observation(z,coordinate,x,w['rstar'],Rch,dict(n=n,fraction=1.),a0,t['Q'],stopped,correction)
                values.append(dict(world=world,**row))
            save(HERE/'checkpoints'/(identifier(t)+'.json'),dict(task=t,coordinate=coordinate,stage=stage,elapsed_hours=float(sol.t[-1]),worlds=values,previous_rows=rows,completed_routes=routes,manifest_sha256=sha(HERE/'manifest.json')))
            assert not stopped,'Task stopped at contact floor; endpoint checkpoint preserved'
            effect=values[1]['ratio']-values[0]['ratio']
            rows.append(dict(stage=stage,worlds=values,effect=effect,coefficient=effect/t['epsilon']**2))
            calls+=sum(s.evaluations for s in systems);checkpoint_guard()
        routes[coordinate]=dict(rows=rows,evaluations=calls)
    rows=[]
    for i,(p,g) in enumerate(zip(routes['w']['rows'],routes['g']['rows'])):
        re=max(abs(a['ratio']-b['ratio']) for a,b in zip(p['worlds'],g['worlds']));ee=abs(p['effect']-g['effect'])
        previous=None
        numeric=all(v['numerical_pass'] for route in [p,g] for v in route['worlds']) and re<=1e-6 and ee<=max(1e-9,.01*abs(g['effect']))
        rows.append(dict(stage=p['stage'],primary=p,independent=g,coordinate_ratio_difference=re,effect_coordinate_difference=ee,
                         numerical_pass=bool(numeric),physical_pass=all(v['physical_pass'] for route in [p,g] for v in route['worlds'])))
    return dict(loading_domain=loading,protocol=dict(J=t['J'],Q=t['Q'],rest=t['rest'],discharge_hours=t['Q']/t['J']),initial_contact=a0.tolist(),R_ch=Rch,q0=q0,initial_shift=float(shift),initial_matching_error=abs(mismatch(shift)),rows=rows,
                evaluations={k:r['evaluations'] for k,r in routes.items()},time_steps=timestep_records,available=all(r['numerical_pass'] and r['physical_pass'] for r in rows))


def check_source():
    original=inspect.getsource(load.finite_symmetric).strip()
    expected=original
    for a,b in REPLACEMENTS:
        assert expected.count(a)==1,a
        expected=expected.replace(a,b)
    actual=inspect.getsource(finite_symmetric).strip()
    assert actual==expected,'Unexpected scientific source change'
    return dict(original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),refined_function_sha256=hashlib.sha256(actual.encode()).hexdigest(),changes='Tolerances, maximum step, and time-step diagnostics only')


def freeze():
    m=read(load.HERE/'manifest.json');completed=read(load.HERE/'all_summary.json')
    assert completed['audit_passed'] and completed['all_cases_available'] and completed['new_case_count']==24
    assert completed['known_gate64']['passed'] and completed['known_gate128']['passed']
    for p,h in m['sha256'].items():assert sha(ROOT/p)==h,p
    source_check=check_source();tasks=[]
    for t in m['tasks']:
        if t['kind']=='paired' and t['n']==128:
            for law in ['local','common']:tasks.append(dict(t,law=law))
    assert len(tasks)==8
    files=[HERE/'PLAN.md',HERE/'run_temporal.py',HERE/'verify_and_summarize.py',*HERE.glob('*_job.json'),load.HERE/'manifest.json',load.HERE/'all_summary.json',load.HERE/'controls.json',*load.HERE.glob('*_summary.json'),*(load.HERE/'cases').glob('*.json'),*(load.HERE/'checkpoints').glob('*.json')]
    for t in tasks:
        ref=reference_path(t);assert read(ref)['available'];files.append(ref)
    hashes=dict(m['sha256']);hashes.update({str(f.relative_to(ROOT)):sha(f) for f in files})
    pilots=[identifier(t) for t in tasks if t['history']=='P-LCO' and ((t['law']=='local' and t['Q']==.3) or (t['law']=='common' and t['Q']==.6))];assert len(pilots)==2
    out=dict(sha256=hashes,witness=m['witness'],tasks=tasks,pilot_ids=pilots,source_check=source_check,reference_paths={identifier(t):str(reference_path(t).relative_to(ROOT)) for t in tasks},settings=dict(w_rtol=1e-11,w_atol=1e-13,g_rtol=1e-13,g_atol=1e-15,max_step_hours=.002,first_step_hours=1e-4),central_claim_changed=False,physics_changed=False,previous_failures_preserved=True)
    path=HERE/'manifest.json'
    if path.exists():assert read(path)==out,'Frozen manifest mismatch'
    else:save(path,out)
    return out


def one(t):
    checkpoint_guard();tic=time.perf_counter();m=read(HERE/'manifest.json');path=HERE/'cases'/(identifier(t)+'.json')
    if path.exists():
        c=read(path);assert c['task']==t and c['manifest_sha256']==sha(HERE/'manifest.json')
        return dict(id=identifier(t),available=c['available'],seconds=c['wall_seconds'],reused=True)
    load.base.SYMMETRY_ERROR=0.;out=dict(task=t,manifest_sha256=sha(HERE/'manifest.json'),reference_path=m['reference_paths'][identifier(t)])
    try:
        result=finite_symmetric(m['witness'],t)
        out.update(result=result,available=result['available'],reduction=dict(duplicate_factor=4,full_grid=t['n'],ode_cells=t['n']//4,discarded_symmetry_relative_error=load.base.SYMMETRY_ERROR))
    except Exception as e:out.update(available=False,error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
    out['wall_seconds']=time.perf_counter()-tic;save(path,out)
    return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','all'],required=True);phase=parser.parse_args().phase
    checkpoint_guard();tic=time.perf_counter();m=freeze();intended=m['tasks'] if phase=='all' else [t for t in m['tasks'] if identifier(t) in m['pilot_ids']]
    admitted=phase=='pilot' or ((HERE/'pilot_summary.json').exists() and read(HERE/'pilot_summary.json')['pilot_admitted'])
    tasks=[t for t in intended if identifier(t) in m['pilot_ids'] or admitted]
    skipped=[dict(task=t,reason='Temporal pilot did not pass; original criteria unchanged') for t in intended if t not in tasks]
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,t) for t in tasks]):
            r=future.result();done.append(r);print(json.dumps(r),flush=True);checkpoint_guard()
    out=dict(phase=phase,manifest_sha256=sha(HERE/'manifest.json'),completed=sorted(done,key=lambda r:r['id']),skipped=skipped,workers=workers,wall_seconds=time.perf_counter()-tic)
    save(HERE/(phase+'_execution.json'),out)


if __name__=='__main__':main()
