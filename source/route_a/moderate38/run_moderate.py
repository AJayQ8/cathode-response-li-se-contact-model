"""Frozen intermediate-roughness screen followed by matched-interface diagnostics."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import importlib.util
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from scipy.optimize import brentq
from scipy.special import expit
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
spec=importlib.util.spec_from_file_location('moderate_prior',HERE.parent/'interaction36/run_interaction.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
weak,spatial,model,projection=prior.weak,prior.spatial,prior.model,prior.projection
read,save,sha=prior.read,prior.save,prior.sha
AMPLITUDES=[.25,.5]


def identifier(t):
    if t['kind']=='known':return weak.identifier(t)
    return t['kind']+'_'+t['witness_id']+'_'+t['initial'][0]+t['history'][0]+'_n'+str(t['n'])+'_e'+str(t['epsilon']).replace('.','p')


def freeze():
    old=read(HERE.parent/'interaction36/manifest.json')
    w=next(w for w in old['witnesses'] if w['id']=='w026')
    _,protocols=model.data(w['setting']['extraction'],w['setting']['registration'])
    tasks=[]
    for e in AMPLITUDES:
        for n in [32,64]:
            for p in protocols:
                if not p['excluded_ambiguous']:
                    tasks.append(dict(kind='known',witness_id='w026',cathode=p['cathode'],rate_c=p['rate_c'],fraction=1.,n=n,amplitude=e))
            for initial in weak.CATHODES:
                for history in weak.CATHODES:
                    tasks.append(dict(kind='paired',witness_id='w026',initial=initial,history=history,n=n,epsilon=e))
    for c in weak.CATHODES:tasks.append(dict(kind='regression',witness_id='w026',initial=c,history=c,n=32,epsilon=.02))
    hashes=old['sha256'].copy()
    files=[Path(__file__).resolve(),HERE/'PLAN.md',HERE.parent/'interaction36/manifest.json',HERE.parent/'interaction36/all_summary.json']
    files += [HERE.parent/'interaction36/cases'/('finite_w026_'+c[0]+c[0]+'_n32_e0p02.json') for c in weak.CATHODES]
    for p in files:hashes[str(p.relative_to(ROOT))]=sha(p)
    for p,digest in hashes.items():assert sha(ROOT/p)==digest,p
    out=dict(sha256=hashes,witness=w,tasks=tasks,amplitudes=AMPLITUDES,grids=[32,64],
             parameters_refitted=False,central_claim_changed=False,response_drafting_deferred=True,
             known_data_tolerance=.03,grid_ratio_tolerance=1e-4,grid_effect_absolute_tolerance=1e-8,grid_effect_relative_tolerance=.03)
    path=HERE/'manifest.json'
    if path.exists():assert read(path)==out
    else:save(path,out)
    return out


def finite(w,t):
    x,a,Rch,q0=prior.setup(w,t);n=t['n'];L=10**x[0];A=x[4]
    cosine=np.cos(2*np.pi*(np.arange(n)+.5)/n)
    def mismatch(shift):return spatial.resistance(expit(shift+t['epsilon']*cosine),x[3]*w['rstar'],A*(1-x[3])*w['rstar'],1.)/Rch-1
    shift=brentq(mismatch,-35.,35.,xtol=1e-12,rtol=1e-14);a0=expit(shift+t['epsilon']*cosine)
    assert abs(spatial.resistance(a0,x[3]*w['rstar'],A*(1-x[3])*w['rstar'],1.)/Rch-1)<=1e-9
    routes={}
    for coordinate in ['w','g']:
        start=np.r_[(a0/A)**2 if coordinate=='w' else L*(1-a0),np.zeros(2*n)]
        y=np.r_[start,start];rows=[];calls=0
        for stage,duration,J in [('end_discharge',.25,1.2),('after_30_min_rest',.5,0.)]:
            systems=[spatial.System(x,w['rstar'],f,n,t['history'],q0 if J else q0+.3,J,coordinate) for f in [0.,1.]]
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
                row=spatial.observation(z,coordinate,x,w['rstar'],Rch,dict(n=n,fraction=1.),a0,.3,stopped,correction)
                values.append(dict(world=world,**row))
            assert not stopped,'Preserved task stopped at contact floor'
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
    return dict(initial_contact=a0.tolist(),R_ch=Rch,q0=q0,initial_shift=float(shift),initial_matching_error=abs(mismatch(shift)),rows=rows,
                evaluations={k:r['evaluations'] for k,r in routes.items()},available=all(r['numerical_pass'] and r['physical_pass'] for r in rows))


def regression(result,t):
    path=HERE.parent/'interaction36/cases'/('finite_w026_'+t['initial'][0]+t['history'][0]+'_n32_e0p02.json')
    old=read(path)['result']
    initial=float(np.max(np.abs(np.array(result['initial_contact'])-old['initial_contact'])))
    ratios=[];effects=[]
    for r,s in zip(result['rows'],old['rows']):
        for route in ['primary','independent']:
            ratios += [abs(a['ratio']-b['ratio']) for a,b in zip(r[route]['worlds'],s[route]['worlds'])]
            effects.append(abs(r[route]['effect']-s[route]['effect']))
    return dict(source=str(path.relative_to(ROOT)),initial_difference=initial,ratio_difference=max(ratios),
                effect_difference=max(effects),passed=bool(result['available'] and initial<=1e-12 and max(ratios)<=1e-7 and max(effects)<=1e-8))


def one(t):
    checkpoint_guard();tic=time.perf_counter();m=read(HERE/'manifest.json');w=m['witness'];path=HERE/'cases'/(identifier(t)+'.json')
    if path.exists():
        out=read(path);assert out['task']==t and out['manifest_sha256']==sha(HERE/'manifest.json')
        return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=True)
    result=weak.finite_history(w,t) if t['kind']=='known' else finite(w,t)
    out=dict(task=t,result=result,available=result['available'],wall_seconds=time.perf_counter()-tic,manifest_sha256=sha(HERE/'manifest.json'))
    if t['kind']=='regression':out['regression']=regression(result,t)
    save(path,out)
    return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=False)


def case(t):return read(HERE/'cases'/(identifier(t)+'.json'))


def known_gate(m,e,n):
    ts=[t for t in m['tasks'] if t['kind']=='known' and t['amplitude']==e and t['n']==n]
    cs=[case(t) for t in ts];rows=[r for c in cs for r in c['result']['rows']]
    assert len(cs)==7 and len(rows)==14
    maximum=max(abs(r['error']) if r['error'] is not None else float('inf') for r in rows)
    changes=[]
    if n==64:
        for t,c in zip(ts,cs):
            old=case(dict(t,n=32))
            changes += [abs(r['ratio']-s['ratio']) if r['ratio'] is not None and s['ratio'] is not None else float('inf') for r,s in zip(c['result']['rows'],old['result']['rows'])]
    return dict(amplitude=e,n=n,observations=len(rows),maximum_absolute_error=maximum,
                all_available=all(c['available'] for c in cs),grid_difference=None if not changes else max(changes),
                passed=bool(all(c['available'] for c in cs) and maximum<=.03 and (not changes or max(changes)<=1e-4)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['known32','known64','future32','future64'],required=True)
    phase=parser.parse_args().phase;checkpoint_guard();tic=time.perf_counter();m=freeze()
    eligible=AMPLITUDES
    if phase!='known32':
        assert all(case(t)['regression']['passed'] for t in m['tasks'] if t['kind']=='regression')
        eligible=[e for e in eligible if known_gate(m,e,32)['passed']]
    if phase in ['future32','future64']:eligible=[e for e in eligible if known_gate(m,e,64)['passed']]
    if phase=='known32':tasks=[t for t in m['tasks'] if t['kind']=='regression' or (t['kind']=='known' and t['n']==32)]
    elif phase=='known64':tasks=[t for t in m['tasks'] if t['kind']=='known' and t['n']==64 and t['amplitude'] in eligible]
    else:tasks=[t for t in m['tasks'] if t['kind']=='paired' and t['n']==int(phase[-2:]) and t['epsilon'] in eligible]
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for future in as_completed(futures):
            item=future.result();done.append(item);print(json.dumps(item),flush=True);checkpoint_guard()
    out=dict(phase=phase,eligible_amplitudes=eligible,completed=sorted(done,key=lambda r:r['id']),workers=workers,
             wall_seconds=time.perf_counter()-tic,manifest_sha256=sha(HERE/'manifest.json'))
    if phase.startswith('known'):
        out['gates']=[known_gate(m,e,int(phase[-2:])) for e in eligible]
        print(json.dumps(dict(gates=out['gates'])),flush=True)
    if phase=='known32':out['regressions']=[case(t)['regression'] for t in m['tasks'] if t['kind']=='regression']
    save(HERE/(phase+'_execution.json'),out)


if __name__=='__main__':main()
