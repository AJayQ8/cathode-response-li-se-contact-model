"""Rest-only continuation and explicit finite-readout timing bounds."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location('frozen_weak_response', HERE.parent/'weak33/run_weak_response.py')
weak = importlib.util.module_from_spec(spec); spec.loader.exec_module(weak)
model, spatial, projection = weak.model, weak.spatial, weak.projection
read, save, sha = weak.read, weak.save, weak.sha
WINDOWS = [dict(name=name, start_s=u, end_s=v, reference_s=ref, primary=name=='first_120_s')
           for name,u,v,ref in [('first_10_s',0,10,0),('first_30_s',0,30,0),('first_60_s',0,60,0),
                               ('first_120_s',0,120,0),('delayed_30_60_s',30,60,0),
                               ('delayed_60_120_s',60,120,0),('delayed_120_180_s',120,180,0),
                               ('delayed_300_360_s',300,360,0),('late_1740_1800_s',1740,1800,1800),
                               ('late_1800_1860_s',1800,1860,1800)]]
TIMES = sorted(set([0,1,5,10,15,30,60,120,180,300,360,600,900,1200,1740,1800,1860]))
BUDGET = .001


def identifier(task):
    return task['witness_id'] if task['kind']=='scalar' else 'weak_'+task['cathode'][0]+'_f'+str(int(task['fraction']))


def finite_source(cathode, fraction):
    return HERE.parent/'weak33/cases'/('future_w026_'+cathode+'_c2p0_f'+str(fraction).replace('.','p')+'_n32_e0p02.json')


def freeze():
    prior = read(HERE.parent/'weak33/manifest.json')
    tasks = [dict(kind='scalar', witness_id=w['id']) for w in prior['witnesses']]
    tasks += [dict(kind='finite', witness_id='w026', cathode=c, fraction=f) for c in weak.CATHODES for f in [0.,1.]]
    paths = [Path(__file__).resolve(), HERE/'PLAN.md', HERE.parent/'weak33/manifest.json',
             HERE.parent/'weak33/summary.json', HERE.parent/'weak33/interpretation.json',
             HERE.parent/'impedance8/spectra.json']
    paths += [HERE.parent/'weak33/cases'/('linear_'+w['id']+'.json') for w in prior['witnesses']]
    paths += [finite_source(c,f) for c in weak.CATHODES for f in [0.,1.]]
    hashes = prior['sha256'].copy()
    for p in paths: hashes[str(p.relative_to(ROOT))] = sha(p)
    for path,digest in hashes.items(): assert sha(ROOT/path)==digest,path
    assert read(HERE.parent/'weak33/summary.json')['verification_passed']
    frozen = dict(sha256=hashes,witnesses=prior['witnesses'],tasks=tasks,times_s=TIMES,windows=WINDOWS,
                  timing_budget=BUDGET,pressure_at_rest='constant final charge-indexed pressure',
                  parameters_refitted=False,central_claim_changed=False,measurement_error_bound=False,
                  ranges_are_sampled=True,all_known_points_retrospective=True)
    p = HERE/'manifest.json'
    if p.exists(): assert read(p)==frozen
    else: save(p,frozen)
    return frozen


def integral(v0, v, m):
    return quad(lambda z: (-math.expm1(-z))**m, v0, v, epsabs=1e-13, epsrel=1e-12, limit=100)[0]


def implicit_contact(a0, gamma, seconds, m=model.M):
    assert 0<a0<=1 and gamma>=0 and seconds>=0
    if seconds==0 or gamma==0 or a0==1:return a0
    v0=-math.log1p(-a0); exposure=gamma*seconds/3600
    upper=v0+exposure/(a0**m)
    if upper==v0:return a0
    upper=float(np.nextafter(upper,math.inf))
    fun=lambda v:integral(v0,v,m)-exposure
    count=0
    while fun(upper)<0:
        upper=v0+2*(upper-v0);count+=1
        assert count<30
    v=brentq(fun,v0,upper,xtol=5e-15,rtol=1e-14)
    return -math.expm1(-v)


def deadline(a, gamma, rb, C, Rch):
    r=(rb+C/a)/Rch; denom=Rch*(r-BUDGET)-rb
    if gamma==0 or denom<=C:return None
    target=C/denom
    assert a<target<1
    return 3600*integral(-math.log1p(-a),-math.log1p(-target),model.M)/gamma


class Rest:
    def __init__(self,x,n,gamma,coordinate):
        self.L=10**x[0];self.A=x[4];self.n=n;self.gamma=gamma;self.coordinate=coordinate
    def calculate(self,y,jacobian=False):
        n=self.n; raw=self.A*np.sqrt(np.maximum(y[:n],0)) if self.coordinate=='w' else 1-y[:n]/self.L
        a=np.clip(raw,self.A*model.ZMIN/4,1.); inside=(raw>self.A*model.ZMIN/4)&(raw<1.)
        h=(1-a)*a**(-model.M);hp=-a**(-model.M)-model.M*(1-a)*a**(-model.M-1)
        F=self.gamma*h; K=self.L*a*F
        first=2*a*F/(self.A*self.A) if self.coordinate=='w' else -self.L*F
        if not jacobian:return np.r_[first,np.zeros(n),K]
        da=np.where(inside,self.A*self.A/(2*a) if self.coordinate=='w' else -1/self.L,0.)
        top=2*self.gamma*(h+a*hp)/(self.A*self.A) if self.coordinate=='w' else -self.L*self.gamma*hp
        out=np.zeros((3*n,3*n));out[:n,:n]=np.diag(top*da)
        out[2*n:,:n]=np.diag(self.L*self.gamma*(h+a*hp)*da)
        return out
    def rhs(self,t,y):return self.calculate(y)
    def jac(self,t,y):return self.calculate(y,True)


def pressures(w,cathode,q):
    eta=w['x'][2]
    fast=2+eta*model.pressure_increment(cathode,q)
    tt,pp,t0,p0,end=projection.original_wave(cathode)
    assert t0+q/.12<=end+1e-10
    direct=2+eta*(float(np.interp(t0+q/.12,tt,pp))-p0)/1000
    assert min(fast,direct)>0 and abs(fast-direct)<=1e-10*max(1.,abs(direct))
    return fast,direct


def controls(manifest):
    path=HERE/'controls.json'
    if path.exists():
        old=read(path);assert old['manifest_sha256']==sha(HERE/'manifest.json') and old['passed'];return
    tests=[]
    for wid in ['w037','w026','w033']:
        w=next(w for w in manifest['witnesses'] if w['id']==wid);x=np.array(w['x']);L,H=10**x[:2];A=x[4]
        for c in weak.CATHODES:
            h=next(h for h in w['diagonal_histories'] if h['cathode']==c);q=h['q0']+.3
            p,_=pressures(w,c,q);gamma=H*A**(model.M+1)*(p/2)**model.M
            a=np.array([.2,.4,.6,.8]);n=len(a)
            for route in ['w','g']:
                y=np.r_[(a/A)**2 if route=='w' else L*(1-a),np.zeros(2*n)]
                new=Rest(x,n,gamma,route);old=spatial.System(x,w['rstar'],1.,n,c,q,0.,route)
                f0,f1=new.rhs(0.,y),old.rhs(0.,y);j0,j1=new.jac(0.,y),old.jac(0.,y)
                e=max(float(np.max(np.abs(f0-f1)))/max(1.,float(np.max(np.abs(f1)))),float(np.max(np.abs(j0-j1)))/max(1.,float(np.max(np.abs(j1)))))
                tests.append(dict(name=wid+'_'+c+'_'+route,scaled_error=e,passed=e<=1e-10))
    exact=[]
    for a in [.1,.5,.99]:
        for gamma in [.00001,.1,1.]:
            for t in [1.,120.,1860.]:
                expected=1-(1-a)*math.exp(-gamma*t/3600)
                error=abs(implicit_contact(a,gamma,t,0.)-expected)
                exact.append(dict(initial_contact=a,gamma=gamma,seconds=t,error=error,passed=error<=1e-11))
        assert implicit_contact(a,0.,1860.)==a
    out=dict(tests=tests,implicit_exact_controls=exact,zero_recovery_pass=True,
             passed=all(r['passed'] for r in tests+exact),manifest_sha256=sha(HERE/'manifest.json'))
    save(path,out);assert out['passed']


def prepare(w,task,c):
    L=10**w['x'][0];A=w['x'][4]
    if task['kind']=='scalar':
        src=HERE.parent/'weak33/cases'/('linear_'+w['id']+'.json');case=read(src)
        h=next(h for h in case['result']['histories'] if h['initial_cathode']==h['history_cathode']==c)
        initial={route:np.array(h['rows'][0][route]['state'][:3]) for route in ['primary','independent']}
        volumes={route:np.array([h['rows'][0][route]['initial_volume']]) for route in initial}
        references={route:{0:h['rows'][0][route]['ratio'],1800:h['rows'][1][route]['ratio']} for route in initial}
        return dict(source=str(src.relative_to(ROOT)),R_ch=h['R_ch'],q0=h['q0'],n=1,fraction=0.,initial=initial,volumes=volumes,references=references)
    src=finite_source(c,task['fraction']);case=read(src);h=case['result'];assert case['available']
    initial={};volumes={};references={}
    for route,coordinate in [('primary','w'),('independent','g')]:
        r=h['rows'][0][route];a=np.array(r['contact'])
        initial[route]=np.r_[(a/A)**2 if coordinate=='w' else L*(1-a),r['stripped_charge_mAh_cm2'],r['replenished_volume_um']]
        volumes[route]=np.array(r['initial_volume_um'])
        references[route]={0:h['rows'][0][route]['ratio'],1800:h['rows'][1][route]['ratio']}
    return dict(source=str(src.relative_to(ROOT)),R_ch=h['R_ch'],q0=h['q0'],n=32,fraction=task['fraction'],initial=initial,volumes=volumes,references=references)


def observation(state,coordinate,x,rstar,info,V0):
    L=10**x[0];b,A=x[3:];n=info['n']
    raw=A*np.sqrt(np.maximum(state[:n],0)) if coordinate=='w' else 1-state[:n]/L
    projection_size=max(0.,float(np.max(state[:n])-1/(A*A))) if coordinate=='w' else max(0.,float(-np.min(state[:n])))*2/(L*A*A)
    a=np.clip(raw,0.,1.);Q=state[n:2*n];K=state[2*n:];V=L*a*a/2
    R=float(spatial.resistance(a,b*rstar,A*(1-b)*rstar,info['fraction']))
    ledger=float(np.max(np.abs(V-V0+model.U*Q-K)))
    physical=bool(np.all(raw>A*model.ZMIN) and np.all(raw<=1+1e-12) and np.all(L*a<=40+1e-10) and np.all(V+model.U*Q<=40+model.U*1e-7) and np.all(K>=-1e-7))
    numeric=bool(np.all(np.abs(V-V0+model.U*Q-K)<=1e-6*np.maximum(1.,np.maximum(V,np.abs(K)))) and
                 abs(float(np.mean(Q))-.3)<=1e-6 and projection_size<=1e-7*max(1.,1/(A*A)))
    return dict(contact=a.tolist(),raw_contact=raw.tolist(),volume_um=V.tolist(),initial_volume_um=V0.tolist(),
                stripped_charge=Q.tolist(),replenishment_um=K.tolist(),resistance=R,ratio=R/info['R_ch'],ledger_um=ledger,
                projection=projection_size,physical_pass=physical,numerical_pass=numeric)


def history(w,task,c):
    x=np.array(w['x']);L,H=10**x[:2];b,A=x[3:];info=prepare(w,task,c);n=info['n'];rstar=w['rstar']
    pp=pressures(w,c,info['q0']+.3);gammas=[H*A**(model.M+1)*(p/2)**model.M for p in pp]
    outputs={};calls={}
    for route,coordinate,gamma in zip(['primary','independent'],['w','g'],gammas):
        system=Rest(x,n,gamma,coordinate)
        sol=solve_ivp(system.rhs,(0.,TIMES[-1]/3600),info['initial'][route],jac=system.jac,method='Radau',
                      t_eval=np.array(TIMES)/3600,rtol=1e-9 if coordinate=='w' else 1e-11,
                      atol=1e-11 if coordinate=='w' else 1e-13,first_step=1e-4)
        assert sol.success and len(sol.t)==len(TIMES) and np.all(np.isfinite(sol.y)),sol.message
        outputs[route]=[observation(y,coordinate,x,rstar,info,info['volumes'][route]) for y in sol.y.T]
        calls[route]=dict(nfev=sol.nfev,njev=sol.njev,nlu=sol.nlu);checkpoint_guard()
    a_start=np.array(outputs['independent'][0]['contact']);rows=[]
    for i,t in enumerate(TIMES):
        primary,direct=outputs['primary'][i],outputs['independent'][i]
        a=np.array([implicit_contact(float(a0),gammas[1],t) for a0 in a_start])
        R=float(spatial.resistance(a,b*rstar,A*(1-b)*rstar,info['fraction']))
        delta=abs(primary['ratio']-direct['ratio']);quad_delta=abs(R/info['R_ch']-direct['ratio'])
        old={route:abs(outputs[route][i]['ratio']-info['references'][route][t]) for route in outputs} if t in [0,1800] else {}
        physical=primary['physical_pass'] and direct['physical_pass']
        numeric=primary['numerical_pass'] and direct['numerical_pass'] and delta<=1e-6 and quad_delta<=1e-7 and all(v<=1e-6 for v in old.values())
        rows.append(dict(time_s=t,primary=primary,independent=direct,implicit_contact=a.tolist(),implicit_ratio=R/info['R_ch'],
                         coordinate_ratio_difference=delta,implicit_ratio_difference=quad_delta,saved_reference_differences=old,
                         numerical_pass=bool(numeric),physical_pass=bool(physical)))
    monotone=all(np.min(np.array(b0[route]['contact'])-np.array(a0[route]['contact']))>=-1e-9 and
                 b0[route]['ratio']<=a0[route]['ratio']+1e-9
                 for a0,b0 in zip(rows,rows[1:]) for route in ['primary','independent'])
    tau=deadline(float(a_start[0]),gammas[1],b*rstar,A*(1-b)*rstar,info['R_ch']) if n==1 else None
    return dict(cathode=c,source=info['source'],R_ch=info['R_ch'],q0=info['q0'],n=n,fraction=info['fraction'],
                pressures_mpa=pp,gammas_per_hour=gammas,rows=rows,monotonicity_pass=bool(monotone),
                first_scalar_drift_budget_time_s=tau,evaluations=calls,
                available=bool(monotone and all(r['numerical_pass'] and r['physical_pass'] for r in rows)))


def run(task,witnesses):
    checkpoint_guard();tic=time.perf_counter();w=next(w for w in witnesses if w['id']==task['witness_id'])
    path=HERE/'cases'/(identifier(task)+'.json')
    if path.exists():
        d=read(path);assert d['task']==task and d['witness']==w and d['manifest_sha256']==sha(HERE/'manifest.json');return d
    histories=[]
    for c in weak.CATHODES if task['kind']=='scalar' else [task['cathode']]:
        try:histories.append(history(w,task,c))
        except (AssertionError,RuntimeError,ValueError,FloatingPointError) as exc:
            histories.append(dict(cathode=c,available=False,error=type(exc).__name__+': '+str(exc)))
    out=dict(task=task,witness=w,histories=histories,available=all(h['available'] for h in histories),
             wall_seconds=time.perf_counter()-tic,manifest_sha256=sha(HERE/'manifest.json'))
    checkpoint_guard();save(path,out);return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','all'],default='pilot');args=parser.parse_args()
    checkpoint_guard();tic=time.perf_counter();manifest=freeze();controls(manifest)
    tasks=manifest['tasks'] if args.phase=='all' else [t for t in manifest['tasks'] if t['witness_id']=='w026']
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));completed=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run,t,manifest['witnesses']) for t in tasks]):
            out=future.result();completed.append(identifier(out['task']))
            print(json.dumps(dict(id=identifier(out['task']),available=out['available'],seconds=out['wall_seconds'])),flush=True)
    save(HERE/(args.phase+'_execution.json'),dict(completed=sorted(completed),workers=workers,wall_seconds=time.perf_counter()-tic,
                                               manifest_sha256=sha(HERE/'manifest.json')))


if __name__=='__main__':main()
