"""Fixed moderate geometry across two preselected compatible scalar fits."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
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
spec=importlib.util.spec_from_file_location('fits_previous',HERE.parent/'current39/run_current.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
prior=previous.prior
weak,spatial,model,projection=previous.weak,previous.spatial,previous.model,previous.projection
read,save,sha=previous.read,previous.save,previous.sha
loading_domain=previous.loading_domain
WITNESSES=['w033','w037']
CURRENTS=[1.2,4.8]


def identifier(t):
    return weak.identifier(t) if t['kind']=='known' else previous.identifier(t)


def baseline_path(t):
    if t['J']==1.2:
        return HERE.parent/'moderate38/cases'/('paired_w026_'+t['initial'][0]+t['history'][0]+'_n'+str(t['n'])+'_e0p25.json')
    return HERE.parent/'current39/cases'/(identifier(t)+'.json')


def freeze():
    old=read(HERE.parent/'current39/manifest.json')
    pool=read(HERE.parent/'interaction36/manifest.json')['witnesses']
    witnesses=[next(w for w in pool if w['id']==wid) for wid in WITNESSES]
    assert [w['setting']['extraction'] for w in witnesses]==['robust','published']
    assert inspect.getsource(finite)==inspect.getsource(previous.finite),'Paired scientific function changed'
    assert read(HERE.parent/'current39/regression_summary.json')['regression']['passed']
    controls=read(HERE.parent/'weak33/controls.json')
    assert controls['passed'] and all(any(t['witness_id']==wid and t['passed'] for t in controls['tests']) for wid in WITNESSES)
    tasks=[];domains=[];baseline=[]
    for w in witnesses:
        rstar,protocols=model.data(w['setting']['extraction'],w['setting']['registration'])
        assert abs(rstar-w['rstar'])<=1e-12
        for n in [32,64]:
            for p in protocols:
                if not p['excluded_ambiguous']:
                    t=dict(kind='known',witness_id=w['id'],cathode=p['cathode'],rate_c=p['rate_c'],fraction=1.,n=n,amplitude=.25)
                    tasks.append(t);domains.append(dict(task=t,loading=model.domain(p,w['x'][2])))
            for J in CURRENTS:
                for initial in weak.CATHODES:
                    for history in weak.CATHODES:
                        t=dict(kind='paired',witness_id=w['id'],initial=initial,history=history,n=n,epsilon=.25,J=J,Q=.3,rest=.5)
                        tasks.append(t);domains.append(dict(task=t,loading=loading_domain(w,t)))
    assert len(tasks)==60 and len(set(identifier(t) for t in tasks))==60
    assert all(d['loading']['valid'] for d in domains)
    for t in old['tasks']:
        if t['kind']=='paired' and t['J'] in CURRENTS:baseline.append(t)
    assert len(baseline)==16
    hashes=old['sha256'].copy()
    paths=[Path(__file__).resolve(),HERE/'verify_and_summarize.py',HERE/'PLAN.md',HERE.parent/'current39/manifest.json',
           HERE.parent/'current39/all_summary.json',HERE.parent/'current39/regression_summary.json',HERE.parent/'weak33/controls.json']
    paths += [baseline_path(t) for t in baseline]
    paths += sorted(HERE.glob('*_job.json'))
    for path in paths:hashes[str(path.relative_to(ROOT))]=sha(path)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    out=dict(sha256=hashes,witnesses=witnesses,tasks=tasks,domains=domains,currents=CURRENTS,grids=[32,64],
             baseline_witness=old['witness'],baseline_tasks=baseline,parameters_refitted=False,central_claim_changed=False,
             paired_function_exact_source_reuse=True,known_function_unchanged=True,response_drafting_deferred=True)
    path=HERE/'manifest.json'
    if path.exists():assert read(path)==out
    else:save(path,out)
    return out


def finite(w,t):
    loading=loading_domain(w,t);assert loading['valid'],loading
    x,a,Rch,q0=prior.setup(w,t);n=t['n'];L=10**x[0];A=x[4]
    cosine=np.cos(2*np.pi*(np.arange(n)+.5)/n)
    def mismatch(shift):return spatial.resistance(expit(shift+t['epsilon']*cosine),x[3]*w['rstar'],A*(1-x[3])*w['rstar'],1.)/Rch-1
    shift=brentq(mismatch,-35.,35.,xtol=1e-12,rtol=1e-14);a0=expit(shift+t['epsilon']*cosine)
    assert abs(spatial.resistance(a0,x[3]*w['rstar'],A*(1-x[3])*w['rstar'],1.)/Rch-1)<=1e-9
    routes={}
    for coordinate in ['w','g']:
        start=np.r_[(a0/A)**2 if coordinate=='w' else L*(1-a0),np.zeros(2*n)]
        y=np.r_[start,start];rows=[];calls=0
        for stage,duration,J in [('end_discharge',t['Q']/t['J'],t['J']),('after_30_min_rest',t['rest'],0.)]:
            systems=[spatial.System(x,w['rstar'],f,n,t['history'],q0 if J else q0+t['Q'],J,coordinate) for f in [0.,1.]]
            def rhs(time,state):return np.r_[systems[0].rhs(time,state[:3*n]),systems[1].rhs(time,state[3*n:])]
            def jac(time,state):return block_diag(systems[0].jac(time,state[:3*n]),systems[1].jac(time,state[3*n:]))
            def event(time,state):
                parts=np.r_[state[:n],state[3*n:4*n]]
                return np.min(parts)-model.ZMIN**2 if coordinate=='w' else L*(1-A*model.ZMIN)-np.max(parts)
            event.terminal=True;event.direction=-1
            sol=solve_ivp(rhs,(0.,duration),y,jac=jac,method='Radau',rtol=1e-10 if coordinate=='w' else 1e-12,
                          atol=1e-12 if coordinate=='w' else 1e-14,first_step=1e-4,events=event)
            assert sol.success and np.all(np.isfinite(sol.y[:,-1]));y=sol.y[:,-1];stopped=bool(len(sol.t_events[0]));values=[]
            for j,world in enumerate(['limit','resolved']):
                z=y[j*3*n:(j+1)*3*n]
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
                evaluations={k:r['evaluations'] for k,r in routes.items()},available=all(r['numerical_pass'] and r['physical_pass'] for r in rows))


def case(t):return read(HERE/'cases'/(identifier(t)+'.json'))


def known_gate(m,wid,n):
    ts=[t for t in m['tasks'] if t['kind']=='known' and t['witness_id']==wid and t['n']==n]
    assert len(ts)==7
    paths=[HERE/'cases'/(identifier(t)+'.json') for t in ts]
    if not all(p.exists() for p in paths):
        return dict(witness_id=wid,n=n,evaluated=False,passed=False,reason='Known trajectories not all completed')
    cs=[read(p) for p in paths]
    if not all('result' in c for c in cs):
        return dict(witness_id=wid,n=n,evaluated=True,passed=False,reason='Saved solver exception',all_available=False)
    rows=[r for c in cs for r in c['result']['rows']]
    assert len(rows)==14
    if any(r['error'] is None for r in rows):
        return dict(witness_id=wid,n=n,evaluated=True,passed=False,reason='Incomplete physical trajectory',all_available=False)
    maximum=max(abs(r['error']) for r in rows);changes=[]
    if n==64:
        for t,c in zip(ts,cs):
            old=case(dict(t,n=32))
            changes.extend(abs(r['ratio']-s['ratio']) for r,s in zip(c['result']['rows'],old['result']['rows']))
    available=all(c['available'] for c in cs)
    return dict(witness_id=wid,n=n,evaluated=True,maximum_absolute_error=maximum,all_available=available,
                grid_difference=None if not changes else max(changes),passed=bool(available and maximum<=.03 and (not changes or max(changes)<=1e-4)))


def eligibility(t,m):
    if t['kind']=='known' and t['n']==32:return True,None
    wid=t['witness_id']
    if not known_gate(m,wid,32)['passed']:return False,'Fit fails or has not completed the n32 known-data gate'
    if t['kind']=='known':return True,None
    if not known_gate(m,wid,64)['passed']:return False,'Fit fails or has not completed the n64 known-data gate'
    if t['n']==64:
        coarse=[u for u in m['tasks'] if u['kind']=='paired' and u['witness_id']==wid and u['J']==t['J'] and u['n']==32]
        assert len(coarse)==4
        if not all((HERE/'cases'/(identifier(u)+'.json')).exists() and case(u)['available'] for u in coarse):
            return False,'At least one n32 paired trajectory fails or is incomplete at this fit/current'
    return True,None


def one(t):
    checkpoint_guard();tic=time.perf_counter();m=read(HERE/'manifest.json');path=HERE/'cases'/(identifier(t)+'.json')
    w=next(w for w in m['witnesses'] if w['id']==t['witness_id'])
    if path.exists():
        out=read(path);assert out['task']==t and out['manifest_sha256']==sha(HERE/'manifest.json')
        return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=True)
    out=dict(task=t,manifest_sha256=sha(HERE/'manifest.json'))
    try:
        result=weak.finite_history(w,t) if t['kind']=='known' else finite(w,t)
        out.update(result=result,available=result['available'])
    except Exception as error:
        out.update(available=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    out['wall_seconds']=time.perf_counter()-tic;save(path,out)
    return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['known32','known64','future32','future64'],required=True)
    phase=parser.parse_args().phase;checkpoint_guard();tic=time.perf_counter();m=freeze()
    kind='known' if phase.startswith('known') else 'paired';n=int(phase[-2:])
    intended=[t for t in m['tasks'] if t['kind']==kind and t['n']==n];tasks=[];skipped=[]
    for t in intended:
        eligible,reason=eligibility(t,m)
        if eligible:tasks.append(t)
        else:skipped.append(dict(task=t,reason=reason))
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for future in as_completed(futures):
            item=future.result();done.append(item);print(json.dumps(item),flush=True);checkpoint_guard()
    out=dict(phase=phase,completed=sorted(done,key=lambda r:r['id']),skipped=skipped,workers=workers,
             wall_seconds=time.perf_counter()-tic,manifest_sha256=sha(HERE/'manifest.json'))
    if phase.startswith('known'):out['gates']=[known_gate(m,wid,n) for wid in WITNESSES]
    save(HERE/(phase+'_execution.json'),out)
    if 'gates' in out:print(json.dumps(dict(gates=out['gates'])),flush=True)


if __name__=='__main__':main()
