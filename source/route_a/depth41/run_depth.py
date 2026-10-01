"""Frozen discharge-depth comparison of two unchanged compatible fits."""
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
spec=importlib.util.spec_from_file_location('depth_previous',HERE.parent/'fits40/run_fits.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
prior=previous.prior
weak,spatial,model,projection=previous.weak,previous.spatial,previous.model,previous.projection
read,save,sha=previous.read,previous.save,previous.sha
loading_domain=previous.loading_domain
CURRENTS=[1.2,4.8]
DEPTHS=[.3,.6]


def identifier(t):
    return previous.identifier(t)+'_Q'+str(t['Q']).replace('.','p')


def baseline_path(t):
    if t['witness_id']=='w026':return previous.baseline_path(t)
    return previous.HERE/'cases'/(previous.identifier(t)+'.json')


def freeze():
    old=read(previous.HERE/'manifest.json');summary=read(previous.HERE/'all_summary.json')
    witnesses=[old['baseline_witness'],next(w for w in old['witnesses'] if w['id']=='w037')]
    known=read(HERE.parent/'moderate38/all_summary.json')['known_gates']
    assert all(any(g['amplitude']==.25 and g['n']==n and g['passed'] for g in known) for n in [32,64])
    assert all(any(g['witness_id']=='w037' and g['n']==n and g['passed'] for g in summary['known_gates']) for n in [32,64])
    assert inspect.getsource(finite)==inspect.getsource(previous.finite),'Scientific function changed'
    baseline=old['baseline_tasks']+[t for t in old['tasks'] if t['kind']=='paired' and t['witness_id']=='w037']
    assert len(baseline)==32
    tasks=[dict(t,Q=.6) for t in baseline]
    assert len(set(identifier(t) for t in tasks+baseline))==64
    domains=[];stock=[]
    for w in witnesses:
        for J in CURRENTS:
            for history in weak.CATHODES:
                t=next(t for t in tasks if t['witness_id']==w['id'] and t['J']==J and t['history']==history)
                domains.append(dict(witness_id=w['id'],J=J,Q=t['Q'],history=history,loading=loading_domain(w,t)))
        for initial in weak.CATHODES:
            t=next(t for t in baseline if t['witness_id']==w['id'] and t['initial']==initial and t['n']==64)
            a=np.array(read(baseline_path(t))['result']['initial_contact']);L=10**w['x'][0];V=L*a*a/2
            stock.append(dict(witness_id=w['id'],initial=initial,minimum_initial_inventory_um=float(min(V)),mean_initial_inventory_um=float(np.mean(V)),
                equivalent_uniform_stripped_depth_um=model.U*.6,minimum_initial_inventory_minus_uniform_stripping_um=float(min(V)-model.U*.6),
                geometric_depth_max_um=float(L*max(a)),stock_limit_um=40.,diagnostic_only=True))
    hashes=old['sha256'].copy()
    paths=[Path(__file__).resolve(),HERE/'verify_and_summarize.py',HERE/'PLAN.md',previous.HERE/'manifest.json',previous.HERE/'all_summary.json']
    paths += [baseline_path(t) for t in baseline]+sorted(HERE.glob('*_job.json'))
    for path in paths:hashes[str(path.relative_to(ROOT))]=sha(path)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    out=dict(sha256=hashes,witnesses=witnesses,tasks=tasks,baseline_tasks=baseline,currents=CURRENTS,depths=DEPTHS,grids=[32,64],
        domains=domains,initial_inventory_diagnostics=stock,parameters_refitted=False,central_claim_changed=False,
        response_drafting_deferred=True,paired_function_exact_source_reuse=True,known_compatibility_reused=True)
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


def eligibility(t,m):
    w=next(w for w in m['witnesses'] if w['id']==t['witness_id'])
    if not loading_domain(w,t)['valid']:return False,'Measured pressure support or positive-pressure gate failed'
    if t['n']==64:
        coarse=[u for u in m['tasks'] if u['witness_id']==t['witness_id'] and u['J']==t['J'] and u['Q']==t['Q'] and u['n']==32]
        assert len(coarse)==4
        if not all((HERE/'cases'/(identifier(u)+'.json')).exists() and case(u)['available'] for u in coarse):
            return False,'At least one n32 paired trajectory fails or is incomplete at this fit/current/depth'
    return True,None


def one(t):
    checkpoint_guard();tic=time.perf_counter();m=read(HERE/'manifest.json');path=HERE/'cases'/(identifier(t)+'.json')
    w=next(w for w in m['witnesses'] if w['id']==t['witness_id'])
    if path.exists():
        out=read(path);assert out['task']==t and out['manifest_sha256']==sha(HERE/'manifest.json')
        return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=True)
    out=dict(task=t,manifest_sha256=sha(HERE/'manifest.json'))
    try:
        result=finite(w,t);out.update(result=result,available=result['available'])
    except Exception as error:
        out.update(available=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    out['wall_seconds']=time.perf_counter()-tic;save(path,out)
    return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['domain','future32','future64'],required=True)
    phase=parser.parse_args().phase;checkpoint_guard();tic=time.perf_counter();m=freeze()
    if phase=='domain':
        out=dict(phase=phase,manifest_sha256=sha(HERE/'manifest.json'),domains=m['domains'],
            all_domains_valid=all(d['loading']['valid'] for d in m['domains']),initial_inventory_diagnostics=m['initial_inventory_diagnostics'],
            integration_advances_executed=0,wall_seconds=time.perf_counter()-tic)
        save(HERE/'domain_summary.json',out);print(json.dumps(out),flush=True);return
    n=int(phase[-2:]);intended=[t for t in m['tasks'] if t['n']==n];tasks=[];skipped=[]
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
    save(HERE/(phase+'_execution.json'),out)


if __name__=='__main__':main()
