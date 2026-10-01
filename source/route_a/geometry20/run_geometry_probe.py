"""Conditional contact-stock survival law with an explicit stripped-charge ledger."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import importlib.util
import json
import os
import time
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
spec=importlib.util.spec_from_file_location('frozen_protocol_reader',HERE.parent/'forecast9/run_recovery_probe.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
read,save,sha=source.read,source.save,source.sha
LOW=np.array([-5.,-10.,-10.,0.,.01])
HIGH=np.array([2.,5.,5.,.99,1.-1e-8])
ZMIN=1e-6
STEP=5e-5
STARTS=[dict(A=A,b=b) for A in [.05,.5,.95] for b in [.3,.8]]


def tasks():
    return [dict(m=m,timing=t,extraction=e,start=i) for m in [0.,6.6]
            for t in ['early','late'] for e in source.EXTRACTIONS for i in range(6)]


def task_id(t):return f"m{t['m']:g}_{t['timing']}_{t['extraction']}_start{t['start']}"
def z_from_v(v,A):return float(-np.expm1(-A*max(float(v),0.))/A)
def v_from_z(z,A):return float(-np.log1p(-A*z)/A)


def advance(v0,dt,H,A,D,m,independent=False):
    floor=v_from_z(ZMIN,A)
    if dt==0:return v0,False,0,0.
    if v0<=floor:return floor,True,0,0.
    def rhs(t,y):
        z=max(z_from_v(y[0],A),ZMIN/4)
        return [H*z**(1-m)-D]
    def event(t,y):return y[0]-floor
    event.terminal=True;event.direction=-1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out=solve_ivp(rhs,(0.,dt),[v0],method='Radau' if independent else 'LSODA',
            rtol=1e-11 if independent else 1e-9,atol=1e-13 if independent else 1e-11,events=event)
    if not out.success or not np.isfinite(out.y[0,-1]):raise RuntimeError('Volume integration failed: '+out.message)
    return float(max(floor,out.y[0,-1])),bool(len(out.t_events[0])),len(caught),float(out.t[-1])


def data(extraction):
    _,protocols=source.data(extraction)
    return min(p['R_ch'] for p in protocols if p['subset']=='train'),protocols


def unavailable(row):return row.get('observation_invalid',False) or row['disconnected']


def classify(rows):
    out={}
    for subset in ['train','check']:
        selected=[r for r in rows if r['subset']==subset]
        complete=all(not unavailable(r) for r in selected)
        errors=np.array([100. if unavailable(r) else r['error'] for r in selected])
        out[subset]=dict(n=len(selected),sse=float(errors@errors),
            rmse=float(np.sqrt(np.mean(errors**2))) if complete else None,
            max_abs_error=float(max(abs(errors))) if complete else None,
            screen_pass=bool(complete and max(abs(errors))<=.03))
    return out


def predictions(x,task,rstar,protocols,independent=False):
    alpha,hp,hn=10**np.asarray(x[:3]);rb=x[3]*rstar;A=x[4]
    out=[]
    for p in protocols:
        if p['excluded_ambiguous']:continue
        h=hp if p['cathode']=='P-LCO' else hn
        z0=None if p['R_ch']==rb else (rstar-rb)/(p['R_ch']-rb)
        a0=None if z0 is None else A*z0
        if a0 is None or not 0<a0<1:
            for target in p['targets']:
                out.append(dict(cathode=p['cathode'],rate_c=p['rate_c'],stage=target['stage'],subset=p['subset'],
                    observed_ratio=target['ratio'],predicted_ratio=None,error=None,disconnected=False,
                    observation_invalid=True,invalid_reason='Initial contact outside finite-stock domain 0<a<1',
                    initial_z=z0,initial_a=a0,initial_v=None,z=None,effective_a=None,v=None,
                    stripped_charge_mAh_cm2=None,replenishment_scaled=None,
                    required_initial_metal_stock_mAh_cm2=None,initial_contact_stock_mAh_cm2=None,integration_warnings=0))
            continue
        v0=v_from_z(z0,A)
        v1,s1,w1,t1=advance(v0,p['t1'],h,A,alpha*p['J'],task['m'],independent)
        vr,sr,wr,tr=(v1,True,0,0.) if s1 else advance(v1,p['rest'],h,A,0.,task['m'],independent)
        v2,s2,w2,t2=(vr,True,0,0.) if sr else advance(vr,p['t2'],h,A,alpha*p['J'],task['m'],independent)
        first=(v1,s1,p['J']*t1) if task['timing']=='early' else (vr,sr,p['J']*t1)
        second=(v2,s2,p['J']*(t1+t2))
        for target,(v,stop,q) in zip(p['targets'],[first,second]):
            z=z_from_v(v,A);a=A*z
            value=None if stop else float((rb+(rstar-rb)/z)/p['R_ch'])
            out.append(dict(cathode=p['cathode'],rate_c=p['rate_c'],stage=target['stage'],subset=p['subset'],
                observed_ratio=target['ratio'],predicted_ratio=value,error=None if value is None else value-target['ratio'],
                disconnected=stop,initial_z=z0,initial_a=a0,initial_v=v0,z=z,effective_a=a,v=v,
                stripped_charge_mAh_cm2=q,replenishment_scaled=v-v0+alpha*q,
                required_initial_metal_stock_mAh_cm2=v/alpha+q,
                initial_contact_stock_mAh_cm2=v0/alpha,integration_warnings=w1+wr+w2))
    return out


class Problem:
    def __init__(self,task):
        self.task=task;self.rstar,self.protocols=data(task['extraction'])
        self.train=[p for p in self.protocols if p['subset']=='train']
        self.evaluations=0;self.failures={}
        self.last_x=None;self.last_residual=None

    def residual(self,x,independent=False):
        if not independent and self.last_x is not None and np.array_equal(x,self.last_x):return self.last_residual.copy()
        self.evaluations+=1
        if self.evaluations%50==0:checkpoint_guard()
        try:
            points=predictions(x,self.task,self.rstar,self.train,independent)
            result=np.array([100. if unavailable(p) else p['error'] for p in points])
            if not np.all(np.isfinite(result)):raise FloatingPointError('Nonfinite calibration residual')
        except (RuntimeError,ValueError,FloatingPointError,OverflowError) as exc:
            message=type(exc).__name__+': '+str(exc)[:160]
            self.failures[message]=self.failures.get(message,0)+1
            result=np.full(8,100.)
        if not independent:self.last_x=np.asarray(x).copy();self.last_residual=result.copy()
        return result

    def jacobian(self,x):
        center=self.residual(x);jac=np.empty((8,5))
        for i,(lo,hi) in enumerate(zip(LOW,HIGH)):
            a,b=x.copy(),x.copy()
            if x[i]-STEP>=lo and x[i]+STEP<=hi:
                a[i]-=STEP;b[i]+=STEP
                jac[:,i]=(self.residual(b)-self.residual(a))/(2*STEP)
            else:
                sign=1. if x[i]-STEP<lo else -1.
                h=min(STEP,(hi-x[i])/2 if sign>0 else (x[i]-lo)/2)
                a[i]+=sign*h;b[i]+=2*sign*h
                jac[:,i]=sign*(-3*center+4*self.residual(a)-self.residual(b))/(2*h)
        return jac

    def assess(self,x,label):
        fast=predictions(x,self.task,self.rstar,self.protocols)
        direct=predictions(x,self.task,self.rstar,self.protocols,True)
        checks=[]
        for a,b in zip(fast,direct):
            same=a['disconnected']==b['disconnected'] and a.get('observation_invalid',False)==b.get('observation_invalid',False)
            delta=None if unavailable(a) or unavailable(b) else abs(a['predicted_ratio']-b['predicted_ratio'])
            checks.append(dict(cathode=a['cathode'],rate_c=a['rate_c'],stage=a['stage'],status_matches=same,
                ratio_difference=delta,passed=same and (delta is None or delta<=1e-4)))
        subsets=classify(direct)
        numerical=all(c['passed'] for c in checks)
        physical=all(not p.get('observation_invalid',False) and 0<p['initial_a']<1 and 0<p['effective_a']<=1 and p['replenishment_scaled']>=-1e-7 for p in direct)
        training_numeric=all(c['passed'] for c in checks if c['rate_c'] in [.2,2.])
        training_physical=all(not p.get('observation_invalid',False) and 0<p['initial_a']<1 and 0<p['effective_a']<=1 and p['replenishment_scaled']>=-1e-7
                              for p in direct if p['subset']=='train')
        return dict(label=label,x=x.tolist(),points=direct,optimizer_predictions=fast,numerical_checks=checks,
            numerical_pass=numerical,physical_observation_pass=physical,subsets=subsets,
            calibration_admissible=training_numeric and training_physical and all(not p['disconnected'] for p in direct if p['subset']=='train'),
            training_objective=float(np.dot(self.residual(x),self.residual(x))),
            max_required_initial_metal_stock_mAh_cm2=max((p['required_initial_metal_stock_mAh_cm2'] for p in direct if not p.get('observation_invalid',False)),default=None),
            bound_indices=np.flatnonzero(np.minimum(x-LOW,HIGH-x)/(HIGH-LOW)<1e-5).tolist())


def initial(task):
    hp,hn=(.02,.005) if task['m']==0 else (1e-4,1e-4)
    start=STARTS[task['start']]
    return np.array([np.log10(.2),np.log10(hp),np.log10(hn),start['b'],start['A']])


def controls():
    tests=[]
    for m in [0.,6.6]:
        A=.4;v0=1.1;D=.2;dt=.3
        actual,stopped,_,_=advance(v0,dt,0.,A,D,m)
        error=abs(actual-(v0-D*dt));tests.append(dict(name=f'pure_stripping_m{m:g}',error=error,passed=error<1e-7 and not stopped))
        floor=v_from_z(ZMIN,A);expected=(v0-floor)/D
        _,stopped,_,elapsed=advance(v0,2*expected,0.,A,D,m)
        tests.append(dict(name=f'stripping_stop_m{m:g}',error=abs(elapsed-expected),passed=stopped and abs(elapsed-expected)<1e-7))
        H=D/z_from_v(v0,A)**(1-m)
        actual,stopped,_,_=advance(v0,dt,H,A,D,m)
        tests.append(dict(name=f'steady_m{m:g}',error=abs(actual-v0),passed=not stopped and abs(actual-v0)<1e-7))
        for v0,H,D,dt in [(1.1,.02,.1,.3),(.4,.0001,.2,.1),(2.,.03,0.,.4)]:
            fast=advance(v0,dt,H,A,D,m);reference=advance(v0,dt,H,A,D,m,True)
            error=abs(fast[0]-reference[0]);tests.append(dict(name=f'mixed_m{m:g}',error=error,passed=fast[1]==reference[1] and error<1e-7))
    v0,A,H,dt=1.1,.4,.2,.3
    expected=np.log1p(np.expm1(A*v0)*np.exp(H*dt))/A
    actual,stop,_,_=advance(v0,dt,H,A,0.,0.)
    tests.append(dict(name='m0_exact_rest',error=float(abs(actual-expected)),passed=not stop and abs(actual-expected)<1e-7))
    for m in [0.,6.6]:
        task=dict(m=m,timing='early',extraction='ordinary',start=2)
        p=Problem(task);x=initial(task);jac=p.jacobian(x);errors=[]
        for i in range(5):
            a,b=x.copy(),x.copy();a[i]-=2e-5;b[i]+=2e-5
            fd=(p.residual(b,True)-p.residual(a,True))/(4e-5)
            errors.append(float(np.max(np.abs(fd-jac[:,i]))/max(1.,np.max(np.abs(fd)))))
        tests.append(dict(name=f'residual_derivatives_m{m:g}',errors=errors,passed=max(errors)<=1e-3))
        assessment=p.assess(x,'control')
        inverse_error=0.
        for row in assessment['points']:
            protocol=next(q for q in p.protocols if (q['cathode'],q['rate_c'])==(row['cathode'],row['rate_c']))
            z=z_from_v(row['initial_v'],x[4]);rb=x[3]*p.rstar
            inverse_error=max(inverse_error,abs((rb+(p.rstar-rb)/z)/protocol['R_ch']-1))
        tests.append(dict(name=f'initial_observation_and_integrators_m{m:g}',inverse_error=inverse_error,
            passed=inverse_error<=1e-10 and assessment['numerical_pass'] and assessment['physical_observation_pass']))
    for test in tests:test['passed']=bool(test['passed'])
    out=dict(tests=tests,passed=all(t['passed'] for t in tests));save(HERE/'controls.json',out)
    assert out['passed'],'Geometry controls must pass before fitting'


def manifest():
    prior=read(HERE.parent/'bounds18/manifest.json')['sha256']
    paths=[Path(__file__),HERE/'PLAN.md',HERE/'DOMAIN_REPORT_REPAIR.md',HERE/'domain_failure_v2/manifest.json',
           HERE/'domain_failure_v2/completed_output_hashes.json',HERE.parent/'difference19/PLAN.md',HERE.parent/'difference19/results.json',
           HERE.parent/'difference19/verification.json',HERE.parent/'bounds18/summary.json']
    hashes={**prior,**{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    for name,digest in hashes.items():assert sha(ROOT/name)==digest,name
    for name,digest in read(HERE/'domain_failure_v2/completed_output_hashes.json').items():assert sha(HERE/name)==digest,name
    assert read(HERE.parent/'difference19/verification.json')['passed']
    return dict(sha256=hashes,tasks=tasks(),low=LOW.tolist(),high=HIGH.tolist(),starts=STARTS,
                constitutive_hypothesis_changed=True,central_claim_changed=False,check_targets_used_in_calibration=False)


def fit(task):
    tic=time.perf_counter();checkpoint_guard()
    p=Problem(task);x0=initial(task)
    opt=least_squares(p.residual,x0,jac=p.jacobian,bounds=(LOW,HIGH),x_scale='jac',
        max_nfev=350,ftol=1e-9,xtol=1e-9,gtol=1e-8)
    candidates=[p.assess(x0,'start'),p.assess(opt.x,'optimizer')]
    allowed=[c for c in candidates if c['calibration_admissible']]
    chosen=min(allowed,key=lambda c:c['training_objective']) if allowed else None
    out=dict(id=task_id(task),task=task,rstar=p.rstar,candidates=candidates,
        selected_label=None if chosen is None else chosen['label'],
        optimizer=dict(success=bool(opt.success),status=int(opt.status),message=str(opt.message),nfev=int(opt.nfev),
                       cost=float(opt.cost),optimality=float(opt.optimality)),
        evaluations=p.evaluations,evaluation_failures=p.failures,wall_seconds=time.perf_counter()-tic,
        check_targets_used_in_calibration=False,prospective_prediction_qualified=False)
    save(HERE/'fits'/(out['id']+'.json'),out);checkpoint_guard()
    return dict(id=out['id'],seconds=out['wall_seconds'],optimizer_success=bool(opt.success),
        training_max=None if chosen is None else chosen['subsets']['train']['max_abs_error'],
        check_max=None if chosen is None else chosen['subsets']['check']['max_abs_error'],
        all_screens_pass=False if chosen is None else chosen['numerical_pass'] and chosen['physical_observation_pass'] and all(s['screen_pass'] for s in chosen['subsets'].values()))


def summarize():
    fits=[read(HERE/'fits'/(task_id(t)+'.json')) for t in tasks()]
    groups=[]
    for m in [0.,6.6]:
        for timing in ['early','late']:
            for extraction in source.EXTRACTIONS:
                matched=[f for f in fits if (f['task']['m'],f['task']['timing'],f['task']['extraction'])==(m,timing,extraction)]
                pool=[(f,c) for f in matched for c in f['candidates'] if c['label']==f['selected_label']]
                if not pool:groups.append(dict(m=m,timing=timing,extraction=extraction,selected_id=None));continue
                f,c=min(pool,key=lambda pair:pair[1]['training_objective'])
                groups.append(dict(m=m,timing=timing,extraction=extraction,selected_id=f['id'],selected_label=c['label'],
                    x=c['x'],subsets=c['subsets'],numerical_pass=c['numerical_pass'],physical_observation_pass=c['physical_observation_pass'],
                    training_objective=c['training_objective'],optimizer_success=f['optimizer']['success'],
                    all_screens_pass=c['numerical_pass'] and c['physical_observation_pass'] and all(s['screen_pass'] for s in c['subsets'].values()),
                    max_required_initial_metal_stock_mAh_cm2=c['max_required_initial_metal_stock_mAh_cm2']))
    out=dict(groups=groups,fit_count=len(fits),optimizer_success_count=sum(f['optimizer']['success'] for f in fits),
        selected_all_screens_pass=sum(g.get('all_screens_pass',False) for g in groups),
        prospective_prediction_qualified=False,central_claim_changed=False,prior_failures_reclassified=False)
    save(HERE/'summary.json',out);print(json.dumps({k:v for k,v in out.items() if k!='groups'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();checkpoint_guard()
    frozen=json.loads(json.dumps(manifest()));path=HERE/'manifest.json'
    if path.exists():assert read(path)==frozen,'Exact manifest required'
    else:save(path,frozen)
    chosen=tasks()
    if args.pilot:
        controls();chosen=[t for t in chosen if t['timing']=='early' and t['extraction']=='ordinary' and t['start']==2]
    else:assert read(HERE/'controls.json')['passed']
    pending=[t for t in chosen if not(args.resume and (HERE/'fits'/(task_id(t)+'.json')).exists())]
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));tic=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fit,t) for t in pending]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True);checkpoint_guard()
    save(HERE/('pilot.json' if args.pilot else 'batch.json'),dict(tasks=results,workers=workers,reused_tasks=len(chosen)-len(pending),wall_seconds=time.perf_counter()-tic))
    if not args.pilot:summarize()


if __name__=='__main__':main()
