"""Frozen fixed-charge current dependence of matched spatial feedback."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import importlib.util
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
spec=importlib.util.spec_from_file_location('current_prior',HERE.parent/'moderate38/run_moderate.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
prior=previous.prior
weak,spatial,model,projection=previous.weak,previous.spatial,previous.model,previous.projection
read,save,sha=previous.read,previous.save,previous.sha
CURRENTS=[1.2,2.4,4.8]


def identifier(t):
    return t['kind']+'_'+t['witness_id']+'_'+t['initial'][0]+t['history'][0]+'_n'+str(t['n'])+'_J'+str(t['J']).replace('.','p')


def reference(t):
    return HERE.parent/'moderate38/cases'/('paired_w026_'+t['initial'][0]+t['history'][0]+'_n'+str(t['n'])+'_e0p25.json')


def loading_domain(w,t):
    _,protocols=model.data(w['setting']['extraction'],w['setting']['registration'])
    history=next(p for p in protocols if p['cathode']==t['history'] and p['rate_c']==2.)
    p=dict(history,J=t['J'],t1=t['Q']/t['J'],t2=0.,rest=t['rest'])
    return model.domain(p,w['x'][2])


def freeze():
    old=read(HERE.parent/'moderate38/manifest.json');summary=read(HERE.parent/'moderate38/all_summary.json')
    w=old['witness'];assert w['id']=='w026'
    gates=[g for g in summary['known_gates'] if g['amplitude']==.25]
    assert len(gates)==2 and all(g['passed'] for g in gates)
    tasks=[]
    for J in CURRENTS:
        for n in [32,64]:
            for initial in weak.CATHODES:
                for history in weak.CATHODES:
                    tasks.append(dict(kind='paired',witness_id='w026',initial=initial,history=history,n=n,epsilon=.25,J=J,Q=.3,rest=.5))
    tasks.append(dict(kind='regression',witness_id='w026',initial='P-LCO',history='P-LCO',n=32,epsilon=.25,J=1.2,Q=.3,rest=.5))
    _,protocols=model.data(w['setting']['extraction'],w['setting']['registration'])
    source_currents=sorted(set(p['J'] for p in protocols if not p['excluded_ambiguous']))
    assert all(any(abs(J-s)<1e-12 for s in source_currents) for J in CURRENTS[1:]),source_currents
    hashes=old['sha256'].copy()
    paths=[Path(__file__).resolve(),HERE/'verify_and_summarize.py',HERE/'PLAN.md',HERE.parent/'moderate38/manifest.json',HERE.parent/'moderate38/all_summary.json',HERE.parent/'moderate38/verify_and_summarize.py']
    paths += [reference(t) for t in tasks if t['J']==1.2]
    paths += sorted(HERE.glob('*_job.json'))
    for path in paths:hashes[str(path.relative_to(ROOT))]=sha(path)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    domains=[dict(task=t,loading=loading_domain(w,t)) for t in tasks]
    assert all(d['loading']['valid'] for d in domains)
    out=dict(sha256=hashes,witness=w,tasks=tasks,currents=CURRENTS,source_currents=source_currents,known_gates=gates,domains=domains,
             parameters_refitted=False,central_claim_changed=False,response_drafting_deferred=True,baseline_reused=True,
             grid_ratio_tolerance=1e-4,grid_effect_absolute_tolerance=1e-8,grid_effect_relative_tolerance=.03)
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


def regression(result,t):
    path=reference(t);old=read(path)['result']
    initial=float(np.max(np.abs(np.array(result['initial_contact'])-old['initial_contact'])))
    ratios=[];effects=[]
    for r,s in zip(result['rows'],old['rows']):
        for route in ['primary','independent']:
            ratios.extend(abs(a['ratio']-b['ratio']) for a,b in zip(r[route]['worlds'],s[route]['worlds']))
            effects.append(abs(r[route]['effect']-s[route]['effect']))
    return dict(source=str(path.relative_to(ROOT)),initial_difference=initial,ratio_difference=max(ratios),
                effect_difference=max(effects),passed=bool(result['available'] and initial<=1e-12 and max(ratios)<=1e-7 and max(effects)<=1e-8))


def case(t):
    if t['kind']=='paired' and t['J']==1.2:
        old=read(reference(t));assert old['manifest_sha256']==sha(HERE.parent/'moderate38/manifest.json')
        return dict(old,task=t,reused_from=str(reference(t).relative_to(ROOT)))
    return read(HERE/'cases'/(identifier(t)+'.json'))


def one(t):
    checkpoint_guard();tic=time.perf_counter();m=read(HERE/'manifest.json');path=HERE/'cases'/(identifier(t)+'.json')
    if path.exists():
        out=read(path);assert out['task']==t and out['manifest_sha256']==sha(HERE/'manifest.json')
        return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=True)
    out=dict(task=t,manifest_sha256=sha(HERE/'manifest.json'))
    try:
        result=finite(m['witness'],t);out.update(result=result,available=result['available'])
        if t['kind']=='regression':out['regression']=regression(result,t)
    except Exception as error:
        out.update(available=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    out['wall_seconds']=time.perf_counter()-tic;save(path,out)
    return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['regression','future32','future64'],required=True)
    phase=parser.parse_args().phase;checkpoint_guard();tic=time.perf_counter();m=freeze();skipped=[]
    if phase=='regression':tasks=[t for t in m['tasks'] if t['kind']=='regression']
    else:
        reg=case(next(t for t in m['tasks'] if t['kind']=='regression'))
        assert reg['available'] and reg['regression']['passed'],'Frozen wrapper regression failed'
        tasks=[t for t in m['tasks'] if t['kind']=='paired' and t['J']!=1.2 and t['n']==int(phase[-2:])]
        if phase=='future64':
            eligible=[J for J in CURRENTS[1:] if all(case(t)['available'] for t in m['tasks'] if t['kind']=='paired' and t['J']==J and t['n']==32)]
            skipped=[dict(task=t,reason='One or more n32 histories failed physical/coordinate checks at this current') for t in tasks if t['J'] not in eligible]
            tasks=[t for t in tasks if t['J'] in eligible]
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for future in as_completed(futures):
            item=future.result();done.append(item);print(json.dumps(item),flush=True);checkpoint_guard()
    out=dict(phase=phase,completed=sorted(done,key=lambda r:r['id']),skipped=skipped,workers=workers,
             wall_seconds=time.perf_counter()-tic,manifest_sha256=sha(HERE/'manifest.json'))
    if phase=='regression':out['regression']=case(tasks[0]).get('regression')
    save(HERE/(phase+'_execution.json'),out)


if __name__=='__main__':main()
