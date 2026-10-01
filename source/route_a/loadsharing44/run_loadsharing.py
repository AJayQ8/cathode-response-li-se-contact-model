"""Frozen test of two total-load-conserving spatial recovery assumptions."""
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
spec=importlib.util.spec_from_file_location('loadsharing_base',HERE.parent/'refine43/precision_v2/run_precision.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
prior,weak,spatial,model,projection=base.prior,base.weak,base.spatial,base.model,base.projection
read,save,sha=base.read,base.save,base.sha
loading_domain=base.loading_domain
lift_map,expand_state,compress_state=base.lift_map,base.expand_state,base.compress_state
STAGES=weak.STAGES


class CommonSystem(base.System):
    def __init__(self,*args,full=False):
        super().__init__(*args);self.circuit=spatial.circuit if full else base.circuit

    def calculate(self,t,y,jacobian=False):
        self.evaluations+=1
        if self.evaluations%2500==0:checkpoint_guard()
        n=self.n;u=y[:n]
        raw=self.A*np.sqrt(np.maximum(u,0.)) if self.coordinate=='w' else 1-u/self.L
        lower=self.A*model.ZMIN/4;a=np.clip(raw,lower,1.);interior=(raw>lower)&(raw<1.)
        mu=float(np.mean(a));power=mu**(-model.M)
        pressure=2+self.eta*model.pressure_increment(self.cathode,self.qstart+self.current*t);assert pressure>0
        scale=(pressure/2)**model.M;q,qa,_=self.circuit(a,self.current,self.beta,jacobian)
        shape=a*(1-a)
        shape_jac=np.diag((1-2*a)*power)-np.outer(model.M*shape*power/mu,np.ones(n)/n) if jacobian else None
        if self.coordinate=='w':
            B=2*self.H0*self.A**(model.M-1)*scale
            recovery=B*shape*power;first=recovery-2*self.D*q
            replenishment=.5*self.L*self.A*self.A*recovery
            if jacobian:
                da=np.where(interior,self.A*self.A/(2*a),0.)
                recovery_jac=B*shape_jac*da[None,:];qj=qa*da[None,:]
                top=recovery_jac-2*self.D*qj;bottom=.5*self.L*self.A*self.A*recovery_jac
        else:
            recover=self.vc*scale;h=(1-a)*power
            first=model.U*q/a-recover*h;replenishment=recover*shape*power
            if jacobian:
                da=np.where(interior,-1/self.L,0.)
                dh=-power*np.eye(n)-np.outer(model.M*(1-a)*power/mu,np.ones(n)/n)
                top=(model.U*qa/a[:,None]-np.diag(model.U*q/(a*a))-recover*dh)*da[None,:]
                qj=qa*da[None,:];bottom=recover*shape_jac*da[None,:]
        if not jacobian:return np.r_[first,q,replenishment]
        result=np.zeros((3*n,3*n));result[:n,:n]=top;result[n:2*n,:n]=qj;result[2*n:,:n]=bottom;return result


def make_system(*args,law,full=False):
    assert law in ['local','common']
    if law=='local':return (spatial.System if full else base.System)(*args)
    return CommonSystem(*args,full=full)


def advance_precise(y,duration,x,rstar,task,cathode,qstart,current,coordinate):
    assert task['mode']==2
    full_n=task['n'];reduced=task['solver']=='symmetric';n=full_n//4 if reduced else full_n
    y=compress_state(y,full_n) if reduced else np.asarray(y).copy();A=x[4];L=10**x[0]
    lift=(lambda state:expand_state(state,full_n)) if reduced else (lambda state:state)
    sys=make_system(x,rstar,task['fraction'],n,cathode,qstart,current,coordinate,law=task['law'],full=not reduced)
    def event(t,state):
        return np.min(state[:n])-model.ZMIN**2 if coordinate=='w' else L*(1-A*model.ZMIN)-np.max(state[:n])
    event.terminal=True;event.direction=-1
    if event(0,y)<=0:
        return lift(y),True,0.,0.,0
    sol=solve_ivp(sys.rhs,(0.,duration),y,method='Radau',jac=sys.jac,
                  rtol=1e-10 if coordinate=='w' else 1e-12,atol=1e-12 if coordinate=='w' else 1e-14,
                  first_step=min(duration,1e-4),events=event)
    assert sol.success and np.all(np.isfinite(sol.y[:,-1])),sol.message
    result=sol.y[:,-1]
    projection_size=max(0.,float(np.max(result[:n])-1/(A*A))) if coordinate=='w' else max(0.,float(-np.min(result[:n])))*2/(L*A*A)
    assert projection_size<=1e-7*max(1.,1/(A*A))
    if coordinate=='w':result[:n]=np.clip(result[:n],model.ZMIN**2,1/(A*A))
    else:result[:n]=np.clip(result[:n],0.,L*(1-A*model.ZMIN))
    return lift(result),bool(len(sol.t_events[0])),float(sol.t[-1]),projection_size,sys.evaluations


def finite_symmetric(w,t):
    assert t['mode']==2
    loading=loading_domain(w,t);assert loading['valid'],loading
    x,a,Rch,q0=prior.setup(w,t);n=t['n'];m=n//4;L=10**x[0];A=x[4]
    cosine=np.cos(2*np.pi*t['mode']*(np.arange(n)+.5)/n)
    def mismatch(shift):return spatial.resistance(expit(shift+t['epsilon']*cosine),x[3]*w['rstar'],A*(1-x[3])*w['rstar'],1.)/Rch-1
    shift=brentq(mismatch,-35.,35.,xtol=1e-12,rtol=1e-14);a0=expit(shift+t['epsilon']*cosine)
    assert abs(spatial.resistance(a0,x[3]*w['rstar'],A*(1-x[3])*w['rstar'],1.)/Rch-1)<=1e-9
    routes={}
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
            sol=solve_ivp(rhs,(0.,duration),y,jac=jac,method='Radau',rtol=1e-10 if coordinate=='w' else 1e-12,
                          atol=1e-12 if coordinate=='w' else 1e-14,first_step=1e-4,events=event)
            assert sol.success and np.all(np.isfinite(sol.y[:,-1]));y=sol.y[:,-1];stopped=bool(len(sol.t_events[0]));values=[]
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
                evaluations={k:r['evaluations'] for k,r in routes.items()},available=all(r['numerical_pass'] and r['physical_pass'] for r in rows))


def finite_history(w, task):
    x = np.array(w['x']); L = 10**x[0]; b, A = x[3:]
    rstar, protocols = model.data(w['setting']['extraction'], w['setting']['registration'])
    source = next(p for p in protocols if p['cathode'] == task['cathode'] and p['rate_c'] == task['rate_c'])
    p = dict(source)
    new = task['kind'] == 'future'
    if new:
        p.update(J=projection.J, t1=projection.Q/projection.J, t2=0., rest=projection.REST)
    assert model.domain(p, x[2])['valid'] and not p['excluded_ambiguous']
    n = task['n']; cosine = np.cos(2*np.pi*task['mode']*(np.arange(n)+.5)/n)
    def mismatch(shift):
        return spatial.resistance(expit(shift+task['amplitude']*cosine), b*rstar, A*(1-b)*rstar, task['fraction'])/p['R_ch']-1
    shift = brentq(mismatch, -35., 35., xtol=1e-12, rtol=1e-14)
    a0 = expit(shift+task['amplitude']*cosine)
    match = abs(mismatch(shift)); assert match <= 1e-9
    initial_mode = float(np.dot(a0, cosine)/np.dot(cosine, cosine))
    outputs = {}; evaluations = {}
    for coordinate in ['w', 'g']:
        state = np.r_[(a0/A)**2 if coordinate == 'w' else L*(1-a0), np.zeros(2*n)]
        snapshots = {}; stopped = False; stripped = 0.; qstart = p['q0']; max_projection = 0.; calls = 0
        segments = [('first', p['t1'], p['J']), ('rest', p['rest'], 0.)]
        if not new:
            segments.append(('second', p['t2'], p['J']))
        for stage, duration, current in segments:
            if not stopped:
                state, stopped, elapsed, correction, count = advance_precise(state, duration, x, rstar, task, p['cathode'], qstart, current, coordinate)
            else:
                elapsed = 0.; correction = 0.; count = 0
            stripped += current*elapsed; qstart += current*elapsed; calls += count
            max_projection = max(max_projection, correction)
            row = spatial.observation(state, coordinate, x, rstar, p['R_ch'], task, a0, stripped, stopped, max_projection)
            a = np.asarray(row['contact']); amplitude = float(np.dot(a, cosine)/np.dot(cosine, cosine))
            row.update(fundamental_amplitude=amplitude, finite_gain=None if initial_mode == 0 else amplitude/initial_mode,
                       mean_contact=float(np.mean(a)), second_harmonic=float(2*np.mean(a*np.cos(4*np.pi*task['mode']*(np.arange(n)+.5)/n))))
            snapshots[stage] = row
            checkpoint_guard()
        outputs[coordinate] = snapshots; evaluations[coordinate] = calls
    rows = []
    for i, stage in enumerate(['first', 'rest'] if new else ['rest', 'second']):
        fast, direct = outputs['w'][stage], outputs['g'][stage]
        delta = None if fast['ratio'] is None or direct['ratio'] is None else abs(fast['ratio']-direct['ratio'])
        numeric = (fast['stopped'] == direct['stopped'] and (delta is None or delta <= 1e-4) and fast['numerical_pass'] and direct['numerical_pass'])
        observed = None if new else p['targets'][i]['ratio']
        rows.append(dict(stage=STAGES[i] if new else p['targets'][i]['stage'], observed_ratio=observed,
                         ratio=direct['ratio'], error=None if observed is None or direct['ratio'] is None else direct['ratio']-observed,
                         primary=fast, independent=direct, ratio_difference=delta, numerical_pass=bool(numeric),
                         physical_pass=fast['physical_pass'] and direct['physical_pass']))
    return dict(cathode=p['cathode'], source_rate_c=p['rate_c'], R_ch=p['R_ch'], q0=p['q0'],
                initial_contact=a0.tolist(), initial_mode=initial_mode, initial_shift=shift, initial_matching_error=match,
                rows=rows, evaluations=evaluations, available=all(r['numerical_pass'] and r['physical_pass'] for r in rows))


def identifier(t):return base.identifier(t)+'_'+t['law']


def local_reference(t):
    if t.get('control') and t['kind']=='known':return base.HERE/'cases'/(base.identifier(dict(t,solver='symmetric'))+'.json')
    if t['n']==128:return base.HERE/'cases'/(base.identifier(dict(t,solver='symmetric'))+'.json')
    return base.baseline_path(t)


def freeze():
    old=read(base.HERE/'manifest.json');w=old['witness'];parent=read(base.HERE/'all_summary.json')
    assert parent['audit_passed'] and parent['continuation_admitted']
    expected=inspect.getsource(base.finite_symmetric).replace("System(x,w['rstar'],f,m,t['history'],q0 if J else q0+t['Q'],J,coordinate)","make_system(x,w['rstar'],f,m,t['history'],q0 if J else q0+t['Q'],J,coordinate,law=t['law'])")
    assert inspect.getsource(finite_symmetric)==expected
    assert inspect.getsource(finite_history)==inspect.getsource(base.finite_history)
    expected=inspect.getsource(base.advance_precise).replace("system_type=System if reduced else spatial.System\n    sys=system_type(x,rstar,task['fraction'],n,cathode,qstart,current,coordinate)","sys=make_system(x,rstar,task['fraction'],n,cathode,qstart,current,coordinate,law=task['law'],full=not reduced)")
    assert inspect.getsource(advance_precise)==expected
    baseline=old['baseline_tasks'];tasks=[dict(t,n=n,solver='symmetric',law='common') for n in [64,128] for t in baseline]
    pilot=next(t for t in tasks if t['kind']=='known' and t['n']==64 and t['cathode']=='P-LCO' and t['rate_c']==2.)
    local=dict(pilot,law='local',control=True);full=dict(pilot,solver='full',control=True)
    local['reference_path']=str(local_reference(local).relative_to(ROOT));tasks=[local,full]+tasks
    pilots=[identifier(t) for t in [local,full,pilot]]
    references=[dict(t,n=n) for n in [64,128] for t in baseline if t['kind']=='paired']
    assert len(tasks)==len(set(identifier(t) for t in tasks))==24
    _,protocols=model.data(w['setting']['extraction'],w['setting']['registration']);domains=[]
    for t in tasks:
        domain=model.domain(next(p for p in protocols if p['cathode']==t['cathode'] and p['rate_c']==t['rate_c']),w['x'][2]) if t['kind']=='known' else loading_domain(w,t)
        assert domain['valid'];domains.append(dict(task=t,domain=domain))
    hashes=old['sha256'].copy()
    paths=[Path(__file__),HERE/'verify_and_summarize.py',HERE/'PLAN.md',base.HERE/'manifest.json',base.HERE/'all_summary.json',base.HERE/'verify_and_summarize_v3.py',base.HERE/'audit_path_repair_manifest.json',HERE.parent/'forecast9/SOURCE_CONTEXT.md',ROOT/'science_lab/.source_cache/lithium_creep_2019/lepage_2019_author_manuscript.pdf',local_reference(local)]
    paths += [local_reference(t) for t in references]+sorted(HERE.glob('*_job.json'))
    for p in paths:hashes[str(p.relative_to(ROOT))]=sha(p)
    for path,h in hashes.items():assert sha(ROOT/path)==h,path
    out=dict(sha256=hashes,witness=w,tasks=tasks,pilot_ids=pilots,reference_tasks=references,domains=domains,grids=[64,128],law_alternatives=['local','common'],parameters_refitted=False,central_claim_changed=False,model_form_sensitivity=True,previous_failures_preserved=True)
    p=HERE/'manifest.json'
    if p.exists():assert read(p)==out
    else:save(p,out)
    return out


def controls(m):
    tic=time.perf_counter();w=m['witness'];x=np.array(w['x']);A=x[4];L=10**x[0]
    _,protocols=model.data(w['setting']['extraction'],w['setting']['registration'])
    q0=next(p for p in protocols if p['cathode']=='P-LCO' and p['rate_c']==2.)['q0'];rows=[];uniform=[]
    for n in [64,128]:
        z=(np.arange(n)+.5)/n;_,P=lift_map(n);P3=block_diag(P,P,P)
        profiles=[.55+.06*np.cos(4*np.pi*z),.30+.09*np.cos(4*np.pi*z)+.03*np.cos(8*np.pi*z)+.01*np.cos(12*np.pi*z),.99+.008*np.cos(4*np.pi*z)]
        for pi,a in enumerate(profiles):
            for coordinate in ['w','g']:
                y=np.r_[(a/A)**2 if coordinate=='w' else L*(1-a),np.zeros(2*n)];yr=compress_state(y,n)
                for f in [0.,1.]:
                    for J in [0.,1.2]:
                        full=make_system(x,w['rstar'],f,n,'P-LCO',q0,J,coordinate,law='common',full=True)
                        reduced=make_system(x,w['rstar'],f,n//4,'P-LCO',q0,J,coordinate,law='common')
                        F=full.rhs(0.,y);Fr=reduced.rhs(0.,yr);Jac=full.jac(0.,y);Jr=reduced.jac(0.,yr)
                        rhs=float(np.max(abs(F-P3@Fr))/max(1.,float(np.max(abs(F)))))
                        je=float(np.max(abs(Jac@P3-P3@Jr))/max(1.,float(np.max(abs(Jac@P3)))))
                        fd=np.zeros_like(Jr)
                        for k in range(n//4):
                            step=1e-6*max(1.,abs(yr[k]));plus=yr.copy();minus=yr.copy();plus[k]+=step;minus[k]-=step
                            fd[:,k]=(reduced.rhs(0.,plus)-reduced.rhs(0.,minus))/(2*step)
                        fe=float(np.max(abs(fd-Jr))/max(1.,float(np.max(abs(Jr)))))
                        pressure=2+x[2]*model.pressure_increment('P-LCO',q0);mu=float(np.mean(a));Gamma=10**x[1]*A**(model.M+1)*(pressure/2)**model.M
                        current=spatial.circuit(a,J,f*x[3]/(A*(1-x[3])))[0]
                        recovery=Gamma*(1-a)*mu**(-model.M);da=recovery-model.U*current/(L*a)
                        independent=np.r_[2*a/A**2*da if coordinate=='w' else -L*da,current,L*a*recovery]
                        physical=float(np.max(abs(F-independent))/max(1.,float(np.max(abs(independent)))))
                        ledger=float(np.max(abs(L*a*da+model.U*current-F[2*n:]))/max(1.,float(np.max(abs(F[2*n:])))))
                        load=abs(float(np.mean(pressure*a/mu))/pressure-1);current_error=abs(float(np.mean(current))-J)/max(1.,J)
                        mean_error=0. if f else abs(float(np.mean(da))-(Gamma*(1-mu)*mu**(-model.M)-model.U*J/(L*mu)))/max(1.,abs(float(np.mean(da))))
                        passed=rhs<=1e-10 and je<=1e-10 and fe<=1e-5 and max(physical,ledger,load,current_error,mean_error)<=1e-10
                        rows.append(dict(n=n,profile=pi,coordinate=coordinate,fraction=f,current=J,rhs_error=rhs,jacobian_error=je,finite_difference_error=fe,physical_rhs_error=physical,ledger_error=ledger,load_error=load,current_error=current_error,mean_identity_error=mean_error,passed=bool(passed)))
                        checkpoint_guard()
    for a0 in [.2,.6,.99]:
        n=64;a=np.full(n,a0)
        for coordinate in ['w','g']:
            y=np.r_[(a/A)**2 if coordinate=='w' else L*(1-a),np.zeros(2*n)]
            for J in [0.,1.2]:
                args=(x,w['rstar'],1.,n,'P-LCO',q0,J,coordinate)
                u=make_system(*args,law='local',full=True).rhs(0.,y);v=make_system(*args,law='common',full=True).rhs(0.,y)
                error=float(np.max(abs(u-v))/max(1.,float(np.max(abs(u)))))
                uniform.append(dict(contact=a0,coordinate=coordinate,current=J,rhs_error=error,passed=bool(error<=1e-10)))
    out=dict(manifest_sha256=sha(HERE/'manifest.json'),checks=rows,uniform_checks=uniform,passed=all(r['passed'] for r in rows+uniform),wall_seconds=time.perf_counter()-tic)
    assert len(rows)==48 and len(uniform)==12;save(HERE/'controls.json',out);return out


def case(t):return read(HERE/'cases'/(identifier(t)+'.json'))


def known_gate(m,n):
    ts=[t for t in m['tasks'] if t['kind']=='known' and not t.get('control') and t['n']==n];assert len(ts)==7
    if not all((HERE/'cases'/(identifier(t)+'.json')).exists() for t in ts):return dict(evaluated=False,passed=False,reason='Known grid incomplete',n=n)
    cases=[case(t) for t in ts]
    if not all(c['available'] for c in cases):return dict(evaluated=True,passed=False,reason='Known case unavailable',n=n)
    errors=[];grid=[]
    for t,c in zip(ts,cases):
        for r in c['result']['rows']:errors.append(dict(cathode=t['cathode'],rate_c=t['rate_c'],stage=r['stage'],error=r['error']))
        if n==128:
            coarse=case(dict(t,n=64));grid.extend(abs(a['ratio']-b['ratio']) for a,b in zip(c['result']['rows'],coarse['result']['rows']))
    worst=max(errors,key=lambda r:abs(r['error']));maximum=abs(worst['error']);difference=max(grid,default=0.)
    return dict(evaluated=True,n=n,comparison_n=64 if n==128 else None,maximum_absolute_error=maximum,worst=worst,grid_difference=difference,errors=errors,passed=bool(maximum<=.03 and difference<=1e-4))


def eligibility(t,m):
    p=HERE/'controls_summary.json'
    if not p.exists() or not read(p)['passed']:return False,'New constitutive controls not independently qualified'
    if identifier(t) in m['pilot_ids']:return True,None
    p=HERE/'pilot_summary.json'
    if not p.exists() or not read(p)['pilot_admitted']:return False,'Full/reduced pilot or wrapper regression incomplete/failed'
    if t['kind']=='known' and t['n']==64:return True,None
    p=HERE/'known64_summary.json'
    if not p.exists() or not read(p)['audit_passed'] or not read(p)['known_gate']['passed']:return False,'Known n64 audit or data screen incomplete/failed'
    if t['kind']=='known':return True,None
    p=HERE/'known128_summary.json'
    if not p.exists() or not read(p)['audit_passed'] or not read(p)['known_gate']['passed']:return False,'Known n128 audit or data/grid screen incomplete/failed'
    return True,None


def one(t):
    checkpoint_guard();tic=time.perf_counter();m=read(HERE/'manifest.json');path=HERE/'cases'/(identifier(t)+'.json')
    if path.exists():
        c=read(path);assert c['task']==t and c['manifest_sha256']==sha(HERE/'manifest.json');return dict(id=identifier(t),available=c['available'],seconds=c['wall_seconds'],reused=True)
    base.SYMMETRY_ERROR=0.;out=dict(task=t,manifest_sha256=sha(HERE/'manifest.json'))
    try:
        result=finite_history(m['witness'],t) if t['kind']=='known' else finite_symmetric(m['witness'],t)
        out.update(result=result,available=result['available'])
        if t['solver']=='symmetric':out['reduction']=dict(duplicate_factor=4,full_grid=t['n'],ode_cells=t['n']//4,discarded_symmetry_relative_error=base.SYMMETRY_ERROR)
    except Exception as e:out.update(available=False,error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
    out['wall_seconds']=time.perf_counter()-tic;save(path,out);return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['controls','pilot','known64','known128','future64','future128'],required=True)
    phase=parser.parse_args().phase;checkpoint_guard();tic=time.perf_counter();m=freeze()
    if phase=='controls':
        out=controls(m);print(json.dumps(dict(passed=out['passed'],checks=len(out['checks']),seconds=out['wall_seconds'])),flush=True);return
    if phase=='pilot':intended=[t for t in m['tasks'] if identifier(t) in m['pilot_ids']]
    else:
        n=64 if phase.endswith('64') else 128;kind='known' if phase.startswith('known') else 'paired'
        intended=[t for t in m['tasks'] if not t.get('control') and t['n']==n and t['kind']==kind]
    tasks=[];skipped=[]
    for t in intended:
        allowed,reason=eligibility(t,m)
        if allowed:tasks.append(t)
        else:skipped.append(dict(task=t,reason=reason))
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,t) for t in tasks]):
            row=future.result();done.append(row);print(json.dumps(row),flush=True);checkpoint_guard()
    out=dict(phase=phase,manifest_sha256=sha(HERE/'manifest.json'),completed=sorted(done,key=lambda r:r['id']),skipped=skipped,workers=workers,wall_seconds=time.perf_counter()-tic)
    if phase.startswith('known'):out['gate']=known_gate(m,n)
    save(HERE/(phase+'_execution.json'),out)


if __name__=='__main__':main()
