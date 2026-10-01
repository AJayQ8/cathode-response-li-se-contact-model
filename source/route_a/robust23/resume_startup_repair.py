"""Scalar Radau assessment with an explicitly bounded startup step."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import importlib.util
import json
import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
spec=importlib.util.spec_from_file_location('frozen_robustness',HERE/'run_robustness.py')
batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
model=batch.model
read,save,sha=batch.read,batch.save,batch.sha
U,M,ZMIN=model.U,model.M,model.ZMIN
initial_state,domain,pressure_increment=model.initial_state,model.domain,model.pressure_increment


def advance(state,duration,x,cathode,q_start,J,independent=True):
    assert independent
    L,H0=10**x[:2];eta=x[2];A=x[4];D=U/(L*A*A)
    floor=ZMIN**2;upper=1/(A*A)
    if state[0]<=floor:return state.copy(),True,0.,0.
    if duration==0:return state.copy(),False,0.,0.
    def rhs(t,y):
        assert -1e-14<=t<=duration+1e-14, 'RHS outside integration interval'
        z=max(float(np.sqrt(max(y[0],0.))),ZMIN/4)
        scale=(2+eta*pressure_increment(cathode,q_start+J*t))/2
        if scale<=0:raise ValueError('Noncompressive pressure during integration')
        g=(1-A*z)*z**(1-M) if A*z<1 else 0.
        return [2*H0*scale**M*g-2*D*J]
    def event(t,y):return y[0]-floor
    event.terminal=True;event.direction=-1
    result=solve_ivp(rhs,(0.,duration),state[:1],method='Radau',rtol=1e-11,atol=1e-13,
                     events=event,first_step=min(duration,1e-4))
    if not result.success or not np.all(np.isfinite(result.y[:,-1])):raise RuntimeError(result.message)
    output=state.copy();output[0]=result.y[0,-1]
    correction=max(0.,float(output[0]-upper))
    if correction>1e-7*max(1.,upper):raise RuntimeError('Full-contact projection exceeds numerical tolerance')
    output[0]=min(upper,max(floor,output[0]))
    return output,bool(len(result.t_events[0])),float(result.t[-1]),correction


def bounded_predictions(x,task,rstar,protocols,independent=True):
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


def assess(p,x,label):
    fast=model.predictions(x,p.task,p.rstar,p.protocols)
    direct=bounded_predictions(x,p.task,p.rstar,p.protocols,True);checks=[]
    for a,b in zip(fast,direct):
        same=a['disconnected']==b['disconnected'] and a.get('observation_invalid',False)==b.get('observation_invalid',False)
        delta=None if model.base.unavailable(a) or model.base.unavailable(b) else abs(a['predicted_ratio']-b['predicted_ratio'])
        checks.append(dict(cathode=a['cathode'],rate_c=a['rate_c'],stage=a['stage'],subset=a['subset'],
                           status_matches=same,ratio_difference=delta,passed=same and (delta is None or delta<=1e-4)))
    return dict(label=label,x=x.tolist(),points=direct,optimizer_predictions=fast,numerical_checks=checks,
        numerical_pass=all(c['passed'] for c in checks),physical_observation_pass=all(model.physical(p) for p in direct),
        subsets=model.base.classify(direct),training_objective=float(p.residual(x)@p.residual(x)),
        calibration_admissible=all(c['passed'] for c in checks if c['subset']=='train') and all(model.physical(p) and not p['disconnected'] for p in direct if p['subset']=='train'),
        bound_indices=np.flatnonzero(np.minimum(x-model.LOW,model.HIGH-x)/(model.HIGH-model.LOW)<1e-5).tolist())


def repair_manifest():
    original=read(HERE/'manifest.json')
    for name,digest in original['sha256'].items():assert sha(ROOT/name)==digest,name
    preserved=read(HERE/'startup_failure_v1/preserved_fits.json')
    for name,digest in preserved.items():assert sha(HERE/name)==digest,name
    return dict(original_manifest_sha256=sha(HERE/'manifest.json'),preserved_fits_sha256=preserved,
                repair_sha256=sha(Path(__file__)),plan_sha256=sha(HERE/'STARTUP_REPAIR.md'),
                first_step_rule_h='min(segment_duration,1e-4)',equations_changed=False,tolerances_changed=False,
                pending=[t for t in batch.tasks() if 'fits/'+model.task_id(t)+'.json' not in preserved])


def controls():
    tests=[];duration=.01
    def slow(t,y):
        if not 0<=t<=duration:raise ValueError('Synthetic startup outside interval')
        return [-1e-6]
    try:
        solve_ivp(slow,(0.,duration),[1.],method='Radau',rtol=1e-11,atol=1e-13)
    except ValueError as exc:
        original_error=str(exc)
    else:
        original_error=None
    fixed=solve_ivp(slow,(0.,duration),[1.],method='Radau',rtol=1e-11,atol=1e-13,first_step=min(duration,1e-4))
    error=abs(fixed.y[0,-1]-(1-1e-6*duration))
    tests.append(dict(name='guarded_slow_ode_startup',original_error=original_error,corrected_endpoint_error=float(error),
                      passed=bool(original_error=='Synthetic startup outside interval' and fixed.success and error<1e-12)))
    for identifier in ['local_early_ordinary_start0','balance_aligned_early_ordinary_start0','local_late_published_start1']:
        fit=read(HERE/'fits'/(identifier+'.json'));saved=next(c for c in fit['candidates'] if c['label']=='optimizer')
        task=fit['task'];rstar,protocols=model.data(task['extraction'],task['registration'])
        corrected=bounded_predictions(np.array(saved['x']),task,rstar,protocols)
        deltas=[];same=True
        for a,b in zip(saved['points'],corrected):
            same &= a['disconnected']==b['disconnected'] and a.get('observation_invalid',False)==b.get('observation_invalid',False)
            if not model.base.unavailable(a) and not model.base.unavailable(b):deltas.append(abs(a['predicted_ratio']-b['predicted_ratio']))
        tests.append(dict(name=identifier,maximum_ratio_difference=max(deltas),status_matches=bool(same),passed=bool(same and max(deltas)<=1e-4)))
        checkpoint_guard()
    report=dict(tests=tests,passed=all(t['passed'] for t in tests),resume_manifest_sha256=sha(HERE/'resume_manifest.json'))
    save(HERE/'startup_repair_controls.json',report);print(json.dumps(report),flush=True)
    assert report['passed'],'Startup repair controls must pass'


def fit(task):
    tic=time.perf_counter();checkpoint_guard();p=model.Problem(task);x0=batch.initial(task)
    opt=least_squares(p.residual,x0,jac=p.jacobian,bounds=(model.LOW,model.HIGH),x_scale='jac',max_nfev=150,
                      ftol=1e-9,xtol=1e-9,gtol=1e-8)
    optimizer=dict(success=bool(opt.success),status=int(opt.status),message=str(opt.message),nfev=int(opt.nfev),
                   njev=int(opt.njev),cost=float(opt.cost),optimality=float(opt.optimality))
    save(HERE/'optimizer_returns'/(model.task_id(task)+'.json'),dict(task=task,x=opt.x.tolist(),optimizer=optimizer,
         resume_manifest_sha256=sha(HERE/'resume_manifest.json'),evaluation_failures=p.failures,wall_seconds=time.perf_counter()-tic))
    candidates=[assess(p,x0,'start'),assess(p,opt.x,'optimizer')]
    allowed=[c for c in candidates if c['calibration_admissible']]
    chosen=min(allowed,key=lambda c:c['training_objective']) if allowed else None
    out=dict(id=model.task_id(task),task=task,rstar=p.rstar,candidates=candidates,
             selected_label=None if chosen is None else chosen['label'],optimizer=optimizer,
             evaluations=p.evaluations,evaluation_failures=p.failures,wall_seconds=time.perf_counter()-tic,
             prospective_prediction_qualified=False,check_targets_used_in_calibration=False,
             input_manifest_sha256=sha(HERE/'manifest.json'),assessment_repair_manifest_sha256=sha(HERE/'resume_manifest.json'))
    save(HERE/'fits'/(out['id']+'.json'),out);checkpoint_guard()
    return dict(id=out['id'],seconds=out['wall_seconds'],optimizer_success=bool(opt.success),
                training_max=None if chosen is None else chosen['subsets']['train']['max_abs_error'],
                check_max=None if chosen is None else chosen['subsets']['check']['max_abs_error'])


def main():
    checkpoint_guard();frozen=json.loads(json.dumps(repair_manifest()));path=HERE/'resume_manifest.json'
    if path.exists():assert read(path)==frozen,'Exact repair manifest required'
    else:save(path,frozen)
    assert len(frozen['preserved_fits_sha256'])==34 and len(frozen['pending'])==2
    controls();tic=time.perf_counter();results=[]
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fit,t) for t in frozen['pending']]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True);checkpoint_guard()
    for name,digest in frozen['preserved_fits_sha256'].items():assert sha(HERE/name)==digest,name
    save(HERE/'resume_execution.json',dict(tasks=results,workers=workers,reused_unchanged=34,wall_seconds=time.perf_counter()-tic))
    batch.summarize()


if __name__=='__main__':main()
