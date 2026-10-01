"""Conditional measured pressure histories with a shared contact recovery law."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[4]
spec=importlib.util.spec_from_file_location('contact_stock',HERE.parent/'geometry20/run_geometry_probe.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
read,save,sha=base.read,base.save,base.sha
M=6.6; ZMIN=1e-6; LN10=np.log(10.)
LOW=np.array([-5.,-10.,0.,0.,.01]); HIGH=np.array([2.,5.,70.,.99,1.-1e-8])
STARTS=[[.2,.001,20.,.3,.5],[.35,.001,40.,.9,.02],[.2,.001,50.,.3,.95]]


def tasks():
    return [dict(registration=r,timing=t,extraction=e,start=s) for r in ['local','balance_aligned']
            for t in ['early','late'] for e in base.source.EXTRACTIONS for s in range(3)]


def task_id(t):return f"{t['registration']}_{t['timing']}_{t['extraction']}_start{t['start']}"


@lru_cache(None)
def source_histories():
    root=HERE.parent/'mechanics3'
    rows=list(csv.DictReader((root/'waveforms.csv').open()))
    summary=read(root/'waveform_summary.json');wave={}
    for meta in summary:
        c=meta['cathode'];points=[r for r in rows if r['cathode']==c and r['quantity']=='pressure_kpa']
        t=np.array([float(r['time_h']) for r in points]);p=np.array([float(r['value']) for r in points])
        assert len(t)==meta['pressure_points'] and np.all(np.diff(t)>0)
        t0=meta['charge_end_voltage_max']['time_h'];end=meta['common_end_h'];p0=float(np.interp(t0,t,p))
        inside=(t>t0)&(t<end);tt=np.r_[t0,t[inside],end];pp=np.interp(tt,t,p)
        assert abs(p0-meta['pressure_at_charge_end_kpa']['value'])<1e-9
        assert abs(pp[-1]-p0-meta['discharge_pressure_change_kpa'])<1e-9
        wave[c]=(np.asarray((tt-t0)*.12),np.asarray((pp-p0)/1000))
    history=read(root/'fullcell_charge_history.json')
    charged={(r['cathode'],r['rate_c']):r['balance_end_mah_cm2'] for r in history if r['segment']=='charge'}
    offsets={key:charged[(key[0],.2)]-value for key,value in charged.items()}
    return wave,offsets


def pressure_increment(cathode,q):
    knots,values=source_histories()[0][cathode]
    if q < -1e-10 or q > knots[-1]+1e-10:raise ValueError('Pressure source support exceeded')
    return float(np.interp(q,knots,values))


@lru_cache(None)
def data(extraction,registration):
    rstar,protocols=base.data(extraction)
    protocols=json.loads(json.dumps(protocols))
    for p in protocols:p['q0']=0. if registration=='local' else source_histories()[1][(p['cathode'],p['rate_c'])]
    return rstar,protocols


def domain(p,eta):
    qs,bs=source_histories()[0][p['cathode']]
    q0=p['q0'];q1=q0+p['J']*(p['t1']+p['t2'])
    if q0 < -1e-10 or q1 > qs[-1]+1e-10:
        return dict(valid=False,reason='Pressure source support exceeded',q0=q0,q_end=q1,source_qmax=float(qs[-1]))
    selected=np.r_[pressure_increment(p['cathode'],q0),bs[(qs>q0)&(qs<q1)],pressure_increment(p['cathode'],q1)]
    minimum=float(np.min(2+eta*selected));maximum=float(np.max(2+eta*selected))
    return dict(valid=minimum>0,reason=None if minimum>0 else 'Noncompressive cathode-only pressure trial',
                q0=q0,q_end=q1,source_qmax=float(qs[-1]),minimum_pressure_mpa=minimum,maximum_pressure_mpa=maximum)


def z_terms(v,A):
    v=max(float(v),0.);E=np.exp(-A*v);z=-np.expm1(-A*v)/A
    za=(-.5*v*v+A*v**3/3-A*A*v**4/8) if A*v<1e-4 else (v*E-z)/A
    return max(float(z),ZMIN/4),float(E),float(za)


def initial_state(x,rstar,Rch):
    b,A=x[3:];n=(1-b)*rstar;d=Rch-b*rstar
    z0=None if d==0 else n/d;a0=None if z0 is None else A*z0
    if a0 is None or not 0<a0<1:return None,z0,a0
    v0=-np.log1p(-a0)/A
    state=np.zeros(6);state[0]=v0
    dzdb=rstar*(rstar-Rch)/(d*d)
    state[4]=dzdb/(1-a0)
    state[5]=(A*z0/(1-a0)+np.log1p(-a0))/(A*A)
    return state,z0,a0


def advance(state,duration,x,cathode,q_start,J,independent=False):
    alpha,H0=10**x[:2];eta=x[2];A=x[4];floor=base.v_from_z(ZMIN,A)
    if state[0]<=floor:return state.copy(),True,0.
    if duration==0:return state.copy(),False,0.
    def rhs(t,y):
        v=y[0];z,E,zA=z_terms(v,A)
        bq=pressure_increment(cathode,q_start+J*t)
        scale=(2+eta*bq)/2
        if scale<=0:raise ValueError('Noncompressive pressure during integration')
        H=H0*scale**M
        F=H*z**(1-M)-alpha*J
        if independent:return [F]
        Fv=H*(1-M)*z**(-M)*E
        direct=np.array([-LN10*alpha*J,LN10*H*z**(1-M),
                         H0*M*scale**(M-1)*bq/2*z**(1-M),0.,H*(1-M)*z**(-M)*zA])
        return np.r_[F,Fv*y[1:]+direct]
    def event(t,y):return y[0]-floor
    event.terminal=True;event.direction=-1
    result=solve_ivp(rhs,(0.,duration),state[:1] if independent else state,
                     method='Radau' if independent else 'LSODA',rtol=1e-11 if independent else 1e-9,
                     atol=1e-13 if independent else 1e-11,events=event)
    if not result.success or not np.all(np.isfinite(result.y[:,-1])):raise RuntimeError(result.message)
    output=state.copy();output[:len(result.y)]=result.y[:,-1]
    return output,bool(len(result.t_events[0])),float(result.t[-1])


def predictions(x,task,rstar,protocols,independent=False):
    alpha=10**x[0];b,A=x[3:];rb=b*rstar;out=[]
    for p in protocols:
        if p['excluded_ambiguous']:continue
        state,z0,a0=initial_state(x,rstar,p['R_ch']);load=domain(p,x[2])
        if state is None or not load['valid']:
            for target in p['targets']:
                out.append(dict(cathode=p['cathode'],rate_c=p['rate_c'],stage=target['stage'],subset=p['subset'],
                    observed_ratio=target['ratio'],predicted_ratio=None,error=None,disconnected=False,observation_invalid=True,
                    invalid_reason='Initial contact outside finite-stock domain' if state is None else load['reason'],
                    initial_z=z0,initial_a=a0,loading=load,derivative=[0.]*5))
            continue
        initial=state.copy();s1,stop1,t1=advance(state,p['t1'],x,p['cathode'],p['q0'],p['J'],independent)
        sr,stopr,tr=(s1,True,0.) if stop1 else advance(s1,p['rest'],x,p['cathode'],p['q0']+p['J']*t1,0.,independent)
        s2,stop2,t2=(sr,True,0.) if stopr else advance(sr,p['t2'],x,p['cathode'],p['q0']+p['J']*t1,p['J'],independent)
        first=(s1,stop1,p['J']*t1) if task['timing']=='early' else (sr,stopr,p['J']*t1)
        for target,(state,stop,Q) in zip(p['targets'],[first,(s2,stop2,p['J']*(t1+t2))]):
            v=state[0];z,E,zA=z_terms(v,A);n=rstar-rb
            value=None if stop else float((rb+n/z)/p['R_ch'])
            deriv=-(n/p['R_ch'])*E/z**2*state[1:]
            deriv[3]+=(rstar-rstar/z)/p['R_ch'];deriv[4]-=n/p['R_ch']*zA/z**2
            out.append(dict(cathode=p['cathode'],rate_c=p['rate_c'],stage=target['stage'],subset=p['subset'],
                observed_ratio=target['ratio'],predicted_ratio=value,error=None if stop else value-target['ratio'],
                disconnected=stop,initial_z=z0,initial_a=a0,initial_v=float(initial[0]),v=float(v),z=z,effective_a=A*z,
                stripped_charge_mAh_cm2=Q,replenishment_scaled=float(v-initial[0]+alpha*Q),
                required_initial_metal_stock_mAh_cm2=float(v/alpha+Q),loading=load,
                derivative=[0.]*5 if independent or stop else deriv.tolist()))
    return out


def physical(row):
    return not row.get('observation_invalid',False) and 0<row['initial_a']<1 and 0<row['effective_a']<=1 and row['replenishment_scaled']>=-1e-7


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


def initial(task):
    x=np.array(STARTS[task['start']]);x[:2]=np.log10(x[:2]);return x


def input_audit():
    rows=[]
    for registration in ['local','balance_aligned']:
        _,protocols=data('ordinary',registration)
        for p in protocols:
            if p['excluded_ambiguous']:continue
            q=[p['q0'],p['q0']+p['J']*p['t1'],p['q0']+p['J']*(p['t1']+p['t2'])]
            rows.append(dict(registration=registration,cathode=p['cathode'],rate_c=p['rate_c'],subset=p['subset'],
                             charge_coordinates=q,assay_changes_mpa=[pressure_increment(p['cathode'],a) for a in q],eta70_domain=domain(p,70.)))
    output=dict(rows=rows,all_source_domains_supported=all(r['eta70_domain']['valid'] for r in rows),
                source_clock_shift_fitted=False,absolute_soc_identified=False)
    save(HERE/'input_audit.json',output)
    assert output['all_source_domains_supported'],'Inspect unsupported histories before fitting'


def controls():
    input_audit();tests=[]
    for registration in ['local','balance_aligned']:
        task=dict(registration=registration,timing='early',extraction='ordinary',start=0)
        p=Problem(task);x=initial(task);t=time.perf_counter();jac=p.jacobian(x);sensitivity_seconds=time.perf_counter()-t
        compare=p.assess(x,'control')
        tests.append(dict(name='independent_forward_'+registration,passed=compare['numerical_pass'] and compare['physical_observation_pass'],
                          maximum_ratio_difference=max(c['ratio_difference'] or 0 for c in compare['numerical_checks']),
                          sensitivity_forward_seconds=sensitivity_seconds))
        errors=[];t=time.perf_counter()
        for i,h in enumerate([2e-5,2e-5,2e-3,2e-5,2e-5]):
            a,b=x.copy(),x.copy();a[i]-=h;b[i]+=h
            lo=predictions(a,task,p.rstar,p.train,True);hi=predictions(b,task,p.rstar,p.train,True)
            fd=np.array([(b['error']-a['error'])/(2*h) for a,b in zip(lo,hi)])
            errors.append(float(np.max(np.abs(fd-jac[:,i]))/max(1.,np.max(np.abs(fd)))))
        tests.append(dict(name='analytic_derivatives_'+registration,errors=errors,passed=max(errors)<=.001,
                          independent_finite_difference_seconds=time.perf_counter()-t))
        x[2]=0.;old=np.array([x[0],x[1],x[1],x[3],x[4]])
        a=predictions(x,task,p.rstar,p.protocols,True)
        b=base.predictions(old,dict(m=M,timing='early'),p.rstar,p.protocols,True)
        error=max(abs(r['predicted_ratio']-s['predicted_ratio']) for r,s in zip(a,b))
        tests.append(dict(name='eta_zero_equal_recovery_'+registration,error=error,passed=error<=1e-6))
        checkpoint_guard()
    for t in tests:t['passed']=bool(t['passed'])
    result=dict(tests=tests,passed=all(t['passed'] for t in tests));save(HERE/'controls.json',result)
    assert result['passed'],'Pressure-history controls must pass before fitting'


def manifest():
    prior=read(HERE.parent/'geometry20/manifest.json')['sha256']
    paths=[Path(__file__),HERE/'PLAN.md',HERE.parent/'geometry20/verification.json',
           HERE.parent/'mechanics3/waveforms.csv',HERE.parent/'mechanics3/waveform_summary.json',HERE.parent/'mechanics3/fullcell_charge_history.json']
    hashes={**prior,**{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    for name,digest in hashes.items():assert sha(ROOT/name)==digest,name
    return dict(sha256=hashes,tasks=tasks(),starts=STARTS,low=LOW.tolist(),high=HIGH.tolist(),
                new_loading_hypothesis=True,central_claim_changed=False,check_targets_used_in_calibration=False)


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
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();checkpoint_guard();frozen=json.loads(json.dumps(manifest()));path=HERE/'manifest.json'
    if path.exists():assert read(path)==frozen,'Exact input manifest required'
    else:save(path,frozen)
    if args.pilot:controls()
    else:assert read(HERE/'controls.json')['passed']
    selected=[t for t in tasks() if not args.pilot or (t['timing']=='early' and t['extraction']=='ordinary' and t['start']==0)]
    pending=[t for t in selected if not(args.resume and (HERE/'fits'/(task_id(t)+'.json')).exists())]
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));tic=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fit,t) for t in pending]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True);checkpoint_guard()
    save(HERE/('pilot.json' if args.pilot else 'batch.json'),dict(tasks=results,workers=workers,reused_tasks=len(selected)-len(pending),wall_seconds=time.perf_counter()-tic))


if __name__=='__main__':main()
