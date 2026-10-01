"""Finite uniform contact heights with Faraday-conserving stripping."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import importlib.util
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
spec=importlib.util.spec_from_file_location('measured_history',HERE.parent/'pressure21/run_pressure_history.py')
pressure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pressure)
base=pressure.base;read,save,sha=pressure.read,pressure.save,pressure.sha
data,domain,pressure_increment=pressure.data,pressure.domain,pressure.pressure_increment
M=6.6;ZMIN=1e-6;LN10=np.log(10.)
U=read(HERE.parent/'mechanics3/mechanical_checks.json')['li_equivalent_um_per_mah_cm2']
LOW=np.array([np.log10(.01),-10.,0.,0.,.01])
HIGH=np.array([np.log10(40.),5.,70.,.99,1.-1e-8])


def tasks():
    return [dict(registration=r,timing='early',extraction='ordinary',start=0) for r in ['local','balance_aligned']]


def task_id(task):return pressure.task_id(task)
def initial(task):return np.array([np.log10(20.),np.log10(.001),20.,.3,.5])


def initial_state(x,rstar,Rch):
    b,A=x[3:];n=(1-b)*rstar;d=Rch-b*rstar
    z0=None if d==0 else n/d;a0=None if z0 is None else A*z0
    if a0 is None or not 0<a0<1:return None,z0,a0
    state=np.zeros(6);state[0]=z0*z0
    state[4]=2*z0*rstar*(rstar-Rch)/(d*d)
    return state,z0,a0


def advance(state,duration,x,cathode,q_start,J,independent=False):
    L,H0=10**x[:2];eta=x[2];A=x[4];D=U/(L*A*A)
    floor=ZMIN**2;upper=1/(A*A)
    if state[0]<=floor:return state.copy(),True,0.,0.
    if duration==0:return state.copy(),False,0.,0.
    def rhs(t,y):
        z=max(float(np.sqrt(max(y[0],0.))),ZMIN/4)
        bq=pressure_increment(cathode,q_start+J*t);scale=(2+eta*bq)/2
        if scale<=0:raise ValueError('Noncompressive pressure during integration')
        H=H0*scale**M
        if A*z<1:
            g=(1-A*z)*z**(1-M)
            dg=-A*z**(1-M)+(1-A*z)*(1-M)*z**(-M)
            direct_A=-2*H*z**(2-M)+4*D*J/A
        else:g=0.;dg=0.;direct_A=4*D*J/A
        F=2*H*g-2*D*J
        if independent:return [F]
        Fw=H*dg/z
        direct=np.array([2*LN10*D*J,2*LN10*H*g,
                         H0*M*scale**(M-1)*bq*g,0.,direct_A])
        result=np.empty(6);result[0]=F;result[1:]=Fw*y[1:]+direct
        return result
    def event(t,y):return y[0]-floor
    event.terminal=True;event.direction=-1
    result=solve_ivp(rhs,(0.,duration),state[:1] if independent else state,
                     method='Radau' if independent else 'LSODA',rtol=1e-11 if independent else 1e-9,
                     atol=1e-13 if independent else 1e-11,events=event)
    if not result.success or not np.all(np.isfinite(result.y[:,-1])):raise RuntimeError(result.message)
    output=state.copy();output[:len(result.y)]=result.y[:,-1]
    correction=max(0.,float(output[0]-upper))
    if correction>1e-7*max(1.,upper):raise RuntimeError('Full-contact projection exceeds numerical tolerance')
    output[0]=min(upper,max(floor,output[0]))
    return output,bool(len(result.t_events[0])),float(result.t[-1]),correction


def predictions(x,task,rstar,protocols,independent=False):
    L,H0=10**x[:2];b,A=x[3:];rb=b*rstar;D=U/(L*A*A);out=[]
    for p in protocols:
        if p['excluded_ambiguous']:continue
        state,z0,a0=initial_state(x,rstar,p['R_ch']);load=domain(p,x[2])
        if state is None or not load['valid']:
            for target in p['targets']:
                out.append(dict(cathode=p['cathode'],rate_c=p['rate_c'],stage=target['stage'],subset=p['subset'],
                    observed_ratio=target['ratio'],predicted_ratio=None,error=None,disconnected=False,observation_invalid=True,
                    invalid_reason='Initial contact outside finite-height domain' if state is None else load['reason'],
                    initial_z=z0,initial_a=a0,loading=load,derivative=[0.]*5))
            continue
        start=state.copy()
        s1,stop1,t1,c1=advance(state,p['t1'],x,p['cathode'],p['q0'],p['J'],independent)
        sr,stopr,tr,cr=(s1,True,0.,0.) if stop1 else advance(s1,p['rest'],x,p['cathode'],p['q0']+p['J']*t1,0.,independent)
        s2,stop2,t2,c2=(sr,True,0.,0.) if stopr else advance(sr,p['t2'],x,p['cathode'],p['q0']+p['J']*t1,p['J'],independent)
        first=(s1,stop1,p['J']*t1) if task['timing']=='early' else (sr,stopr,p['J']*t1)
        for target,(state,stop,Q) in zip(p['targets'],[first,(s2,stop2,p['J']*(t1+t2))]):
            w=float(state[0]);z=float(np.sqrt(w));a=A*z;n=rstar-rb
            value=None if stop else float((rb+n/z)/p['R_ch'])
            derivative=-n/(2*p['R_ch']*z**3)*state[1:]
            derivative[3]+=(rstar-rstar/z)/p['R_ch']
            out.append(dict(cathode=p['cathode'],rate_c=p['rate_c'],stage=target['stage'],subset=p['subset'],
                observed_ratio=target['ratio'],predicted_ratio=value,error=None if stop else value-target['ratio'],disconnected=stop,
                initial_z=z0,initial_a=a0,initial_w=float(start[0]),w=w,z=z,effective_a=a,loading=load,
                height_spread_um=L,maximum_remaining_height_um=L*a,mean_remaining_height_um=L*a/2,
                initial_mean_remaining_height_um=L*a0/2,creep_velocity_scale_um_h=H0*L*A**(M+1),damage_coefficient=D,
                stripped_charge_mAh_cm2=Q,replenishment_scaled=float(w-start[0]+2*D*Q),
                required_initial_metal_stock_mAh_cm2=float(w/(2*D)+Q),nominal_available_metal_stock_mAh_cm2=40/U,
                maximum_w_projection=max(c1,cr,c2),derivative=[0.]*5 if independent or stop else derivative.tolist()))
    return out


def physical(row):
    return (not row.get('observation_invalid',False) and 0<row['initial_a']<1 and 0<row['effective_a']<=1
            and row['replenishment_scaled']>=-1e-7 and row['maximum_remaining_height_um']<=40+1e-10
            and row['required_initial_metal_stock_mAh_cm2']<=row['nominal_available_metal_stock_mAh_cm2']+1e-7)


class Problem:
    def __init__(self,task):
        self.task=task;self.rstar,self.protocols=data(task['extraction'],task['registration'])
        self.train=[p for p in self.protocols if p['subset']=='train']
        self.last_x=None;self.last_r=None;self.last_j=None;self.evaluations=0;self.failures={}
    def calculate(self,x):
        if self.last_x is not None and np.array_equal(x,self.last_x):return
        self.evaluations+=1
        if self.evaluations%20==0:checkpoint_guard()
        try:
            rows=predictions(x,self.task,self.rstar,self.train)
            r=np.array([100. if base.unavailable(p) else p['error'] for p in rows]);j=np.array([p['derivative'] for p in rows])
            if not np.all(np.isfinite(r)) or not np.all(np.isfinite(j)):raise FloatingPointError('Nonfinite fit values')
        except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
            key=type(exc).__name__+': '+str(exc)[:160];self.failures[key]=self.failures.get(key,0)+1
            r=np.full(8,100.);j=np.zeros((8,5))
        self.last_x=np.asarray(x).copy();self.last_r=r;self.last_j=j
    def residual(self,x):self.calculate(x);return self.last_r.copy()
    def jacobian(self,x):self.calculate(x);return self.last_j.copy()
    def assess(self,x,label):
        fast=predictions(x,self.task,self.rstar,self.protocols)
        direct=predictions(x,self.task,self.rstar,self.protocols,True);checks=[]
        for a,b in zip(fast,direct):
            same=a['disconnected']==b['disconnected'] and a.get('observation_invalid',False)==b.get('observation_invalid',False)
            delta=None if base.unavailable(a) or base.unavailable(b) else abs(a['predicted_ratio']-b['predicted_ratio'])
            checks.append(dict(cathode=a['cathode'],rate_c=a['rate_c'],stage=a['stage'],subset=a['subset'],
                               status_matches=same,ratio_difference=delta,passed=same and (delta is None or delta<=1e-4)))
        return dict(label=label,x=x.tolist(),points=direct,optimizer_predictions=fast,numerical_checks=checks,
            numerical_pass=all(c['passed'] for c in checks),physical_observation_pass=all(physical(p) for p in direct),
            subsets=base.classify(direct),training_objective=float(self.residual(x)@self.residual(x)),
            calibration_admissible=all(c['passed'] for c in checks if c['subset']=='train') and all(physical(p) and not p['disconnected'] for p in direct if p['subset']=='train'),
            bound_indices=np.flatnonzero(np.minimum(x-LOW,HIGH-x)/(HIGH-LOW)<1e-5).tolist())


def controls():
    assert read(HERE.parent/'pressure21/input_audit.json')['all_source_domains_supported']
    tests=[];x=initial(tasks()[0]);L=10**x[0];A=x[4];D=U/(L*A*A)
    state=np.zeros(6);state[0]=.8**2
    zero=x.copy();zero[1]=-1000.;zero[2]=0.
    for independent in [False,True]:
        actual,stop,_,_=advance(state,.1,zero,'P-LCO',0.,.2,independent)
        expected=state[0]-2*D*.2*.1
        tests.append(dict(name=f'pure_stripping_{independent}',error=abs(actual[0]-expected),passed=not stop and abs(actual[0]-expected)<1e-7))
        duration=(state[0]-ZMIN**2)/(2*D*.2)
        actual,stop,elapsed,_=advance(state,2*duration,zero,'P-LCO',0.,.2,independent)
        tests.append(dict(name=f'pure_stripping_stop_{independent}',error=abs(elapsed-duration),passed=stop and abs(elapsed-duration)<1e-7))
    full=state.copy();full[0]=1/A**2
    actual,stop,_,correction=advance(full,.1,x,'N-LCO',0.,0.)
    tests.append(dict(name='full_contact_rest_fixed_point',error=abs(actual[0]-full[0]),passed=not stop and abs(actual[0]-full[0])<1e-7 and correction==0))
    constant=x.copy();constant[2]=0.
    for duration,J in [(.1,.6),(.04,0.),(.1,2.4)]:
        actual,stop,_,_=advance(state,duration,constant,'P-LCO',0.,J)
        expected,other_stop,_=base.source.advance(.8,duration,10**x[1],A,D*J,M,1,True)
        error=abs(np.sqrt(actual[0])-expected)
        tests.append(dict(name='eta_zero_earlier_area_equation',error=error,passed=stop==other_stop and error<1e-7))
    for task in tasks():
        p=Problem(task);x=initial(task);tic=time.perf_counter();jac=p.jacobian(x);seconds=time.perf_counter()-tic
        assessment=p.assess(x,'control');errors=[]
        for i,h in enumerate([1e-4,1e-4,.005,1e-4,1e-4]):
            a,b=x.copy(),x.copy();a[i]-=h;b[i]+=h
            lo=predictions(a,task,p.rstar,p.train,True);hi=predictions(b,task,p.rstar,p.train,True)
            fd=np.array([(q['error']-r['error'])/(2*h) for r,q in zip(lo,hi)])
            errors.append(float(np.max(np.abs(fd-jac[:,i]))/max(1.,np.max(np.abs(fd)))))
        tests.append(dict(name='source_forward_and_derivatives_'+task['registration'],errors=errors,sensitivity_seconds=seconds,
                          maximum_ratio_difference=max(c['ratio_difference'] or 0 for c in assessment['numerical_checks']),
                          passed=max(errors)<=.001 and assessment['numerical_pass'] and assessment['physical_observation_pass']))
        checkpoint_guard()
    for t in tests:t['passed']=bool(t['passed'])
    result=dict(tests=tests,passed=all(t['passed'] for t in tests));save(HERE/'controls.json',result)
    assert result['passed'],'Finite-height controls must pass before fitting'


def manifest():
    prior=read(HERE.parent/'pressure21/manifest.json')['sha256']
    paths=[Path(__file__),HERE/'PLAN.md',HERE.parent/'pressure21/verification.json',HERE.parent/'pressure21/scale_check.json',
           HERE.parent/'mechanics3/mechanical_checks.json']
    hashes={**prior,**{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    for name,digest in hashes.items():assert sha(ROOT/name)==digest,name
    return dict(sha256=hashes,tasks=tasks(),low=LOW.tolist(),high=HIGH.tolist(),nominal_foil_height_um=40.,
                new_geometry_hypothesis=True,central_claim_changed=False,check_targets_used_in_calibration=False)


def fit(task):
    tic=time.perf_counter();checkpoint_guard();p=Problem(task);x0=initial(task)
    opt=least_squares(p.residual,x0,jac=p.jacobian,bounds=(LOW,HIGH),x_scale='jac',max_nfev=150,
                      ftol=1e-9,xtol=1e-9,gtol=1e-8)
    candidates=[p.assess(x0,'start'),p.assess(opt.x,'optimizer')]
    allowed=[c for c in candidates if c['calibration_admissible']]
    chosen=min(allowed,key=lambda c:c['training_objective']) if allowed else None
    out=dict(id=task_id(task),task=task,rstar=p.rstar,candidates=candidates,selected_label=None if chosen is None else chosen['label'],
             optimizer=dict(success=bool(opt.success),status=int(opt.status),message=str(opt.message),nfev=int(opt.nfev),
                            njev=int(opt.njev),cost=float(opt.cost),optimality=float(opt.optimality)),
             evaluations=p.evaluations,evaluation_failures=p.failures,wall_seconds=time.perf_counter()-tic,
             prospective_prediction_qualified=False,check_targets_used_in_calibration=False)
    save(HERE/'fits'/(out['id']+'.json'),out);checkpoint_guard()
    return dict(id=out['id'],seconds=out['wall_seconds'],optimizer_success=bool(opt.success),
                training_max=None if chosen is None else chosen['subsets']['train']['max_abs_error'],
                check_max=None if chosen is None else chosen['subsets']['check']['max_abs_error'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();checkpoint_guard();frozen=json.loads(json.dumps(manifest()));path=HERE/'manifest.json'
    if path.exists():assert read(path)==frozen,'Exact manifest required'
    else:save(path,frozen)
    controls();pending=[t for t in tasks() if not(args.resume and (HERE/'fits'/(task_id(t)+'.json')).exists())]
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));tic=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fit,t) for t in pending]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True);checkpoint_guard()
    save(HERE/'pilot.json',dict(tasks=results,workers=workers,reused_tasks=len(tasks())-len(pending),wall_seconds=time.perf_counter()-tic))


if __name__=='__main__':main()
