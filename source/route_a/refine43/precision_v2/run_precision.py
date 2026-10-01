"""Separate known-history precision test and conditional n128 refinement."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import importlib.util
import inspect
import json
import os
import time
import traceback
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import brentq
from scipy.special import expit
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
ORIGINAL=HERE.parent
spec=importlib.util.spec_from_file_location('refine_previous',HERE.parent.parent/'wavelength42/run_wavelength.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
prior=previous.prior
weak,spatial,model,projection=previous.weak,previous.spatial,previous.model,previous.projection
read,save,sha=previous.read,previous.save,previous.sha
loading_domain=previous.loading_domain
STAGES=weak.STAGES
SYMMETRY_ERROR=0.


@lru_cache(maxsize=4)
def lift_map(n):
    assert n%4==0
    m=n//4;labels=np.tile(np.r_[np.arange(m),np.arange(m)[::-1]],2)
    return labels,np.eye(m)[labels]


@lru_cache(maxsize=4)
def boundary_operator(m):
    n=4*m;_,P=lift_map(n);z=spatial.electrical.boundary_operator(n)
    out=P.T@z@P/4;out=(out+out.T)/2;out.setflags(write=False);return out


def expand_state(y,n):
    labels,_=lift_map(n);return np.asarray(y).reshape(3,n//4)[:,labels].reshape(-1)


def compress_state(y,n):
    global SYMMETRY_ERROR
    out=np.asarray(y).reshape(3,n)[:,:n//4].reshape(-1).copy()
    error=float(np.max(np.abs(expand_state(out,n)-y)/np.maximum(1.,np.abs(y))))
    SYMMETRY_ERROR=max(SYMMETRY_ERROR,error);assert error<=1e-12,error
    return out


def solve(active, current, beta):
    a=np.asarray(active,dtype=float)
    assert a.ndim==1 and np.all((a>=0)&(a<=1)) and beta>0
    if np.mean(a)<=0:
        return {'status':'disconnected'}, None, None
    n=len(a)
    z=boundary_operator(n)
    b=np.sqrt(beta*a)
    matrix=(b[:,None]*z)*b[None,:]
    matrix[np.diag_indices(n)]+=1.
    factors=cho_factor(matrix,lower=True,check_finite=False)
    w=cho_solve(factors,b,check_finite=False)
    multiplier=float(n/np.dot(b,w))
    unit_q=multiplier*b*w
    unit_surface=multiplier-z@unit_q
    q=current*unit_q
    active_q=np.where(a>0,current*beta*unit_surface,0.)
    resistance=1+multiplier
    interface=float(np.sum(np.divide(unit_q*unit_q,beta*a,out=np.zeros(n),where=a>0))/n)
    bulk=1+float(np.dot(unit_q,z@unit_q))/n
    residual=unit_q-beta*a*unit_surface
    diagnostics={'status':'solved','resistance_over_bulk':resistance,
                 'current_relative_error':abs(float(np.mean(unit_q))-1),
                 'power_relative_error':abs(interface+bulk-resistance)/resistance,
                 'boundary_law_relative_l2':float(np.linalg.norm(residual)/max(np.linalg.norm(unit_q),1e-30)),
                 'minimum_current_over_applied':float(np.min(unit_q)),
                 'maximum_current_over_applied':float(np.max(unit_q)),
                 'maximum_active_current_over_applied':float(np.max(np.where(a>0,beta*unit_surface,0.))),
                 'zero_gap_current':bool(np.all(q[a==0]==0)),
                 'mean_active_fraction':float(np.mean(a))}
    return diagnostics,q,active_q


electrical=SimpleNamespace(solve=solve,boundary_operator=boundary_operator)


def circuit(a, current, beta, jacobian=False):
    n = len(a)
    if beta==0:
        mean = np.mean(a)
        q = current*a/mean
        derivative = current*(np.eye(n)/mean-a[:,None]/(n*mean*mean)) if jacobian else None
        return q,derivative,None
    diag,q,_ = electrical.solve(a,current,beta)
    assert diag['status']=='solved'
    assert max(diag[k] for k in ['current_relative_error','power_relative_error','boundary_law_relative_l2'])<=1e-9
    derivative = None
    if jacobian:
        if current==0:
            derivative = np.zeros((n,n))
        else:
            bb = np.sqrt(beta*a)
            mat = (bb[:,None]*electrical.boundary_operator(n))*bb[None,:]+np.eye(n)
            G = (bb[:,None]*cho_solve(cho_factor(mat,lower=True,check_finite=False),np.eye(n),check_finite=False))*bb[None,:]
            v = G@np.ones(n)
            tangent = G-np.outer(v,v)/np.sum(v)
            derivative = tangent*(q/(beta*a*a))[None,:]
    return q,derivative,diag


class System:
    def __init__(self,x,rstar,fraction,n,cathode,qstart,current,coordinate):
        self.x=np.asarray(x);self.n=n;self.cathode=cathode;self.qstart=qstart;self.current=current;self.coordinate=coordinate
        self.L,self.H0=10**self.x[:2]
        self.eta,self.b,self.A=self.x[2:]
        self.Rb=self.b*rstar;self.C=self.A*(1-self.b)*rstar
        self.B=fraction*self.Rb;self.S=(1-fraction)*self.Rb;self.beta=self.B/self.C
        self.D=model.U/(self.L*self.A*self.A)
        self.vc=self.H0*self.L*self.A**(model.M+1)
        self.evaluations=0

    def calculate(self,t,y,jacobian=False):
        self.evaluations+=1
        if self.evaluations%2500==0:
            checkpoint_guard()
        u=y[:self.n]
        raw=self.A*np.sqrt(np.maximum(u,0.)) if self.coordinate=='w' else 1-u/self.L
        lower=self.A*model.ZMIN/4
        a=np.clip(raw,lower,1.)
        interior=(raw>lower)&(raw<1.)
        pressure=2+self.eta*model.pressure_increment(self.cathode,self.qstart+self.current*t)
        assert pressure>0
        scale=(pressure/2)**model.M
        q,qa,_=circuit(a,self.current,self.beta,jacobian)
        if self.coordinate=='w':
            z=a/self.A
            recovery=2*self.H0*scale*(1-a)*z**(1-model.M)
            first=recovery-2*self.D*q
            replenishment=.5*self.L*self.A*self.A*recovery
            if jacobian:
                da=np.where(interior,self.A*self.A/(2*a),0.)
                direct=2*self.H0*scale*(-z**(1-model.M)+(1-a)*(1-model.M)*z**(-model.M)/self.A)
                recovery_jac=direct*da
                qj=qa*da[None,:]
                top=np.diag(recovery_jac)-2*self.D*qj
                bottom=np.diag(.5*self.L*self.A*self.A*recovery_jac)
        else:
            recover=self.vc*scale
            h=(1-a)*a**(-model.M)
            first=model.U*q/a-recover*h
            replenishment=recover*(1-a)*a**(1-model.M)
            if jacobian:
                da=np.where(interior,-1/self.L,0.)
                dh=-a**(-model.M)-model.M*(1-a)*a**(-model.M-1)
                top=(model.U*qa/a[:,None]-np.diag(model.U*q/(a*a)+recover*dh))*da[None,:]
                qj=qa*da[None,:]
                dk=recover*(-a**(1-model.M)+(1-a)*(1-model.M)*a**(-model.M))
                bottom=np.diag(dk*da)
        if not jacobian:
            return np.r_[first,q,replenishment]
        result=np.zeros((3*self.n,3*self.n))
        result[:self.n,:self.n]=top
        result[self.n:2*self.n,:self.n]=qj
        result[2*self.n:,:self.n]=bottom
        return result

    def rhs(self,t,y):return self.calculate(t,y,False)
    def jac(self,t,y):return self.calculate(t,y,True)


def advance_precise(y,duration,x,rstar,task,cathode,qstart,current,coordinate):
    assert task['mode']==2
    full_n=task['n'];reduced=task['solver']=='symmetric';n=full_n//4 if reduced else full_n
    y=compress_state(y,full_n) if reduced else np.asarray(y).copy();A=x[4];L=10**x[0]
    lift=(lambda state:expand_state(state,full_n)) if reduced else (lambda state:state)
    system_type=System if reduced else spatial.System
    sys=system_type(x,rstar,task['fraction'],n,cathode,qstart,current,coordinate)
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


def finite_full(w,t):
    loading=loading_domain(w,t);assert loading['valid'],loading
    x,a,Rch,q0=prior.setup(w,t);n=t['n'];L=10**x[0];A=x[4]
    cosine=np.cos(2*np.pi*t['mode']*(np.arange(n)+.5)/n)
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
            systems=[System(x,w['rstar'],f,m,t['history'],q0 if J else q0+t['Q'],J,coordinate) for f in [0.,1.]]
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


def identifier(t):return previous.identifier(t)+'_'+t.get('solver','reference')


def baseline_path(t):return previous.HERE/'cases'/(previous.identifier(t)+'.json')


def freeze():
    old=read(ORIGINAL/'manifest.json');w=old['witness']
    assert inspect.getsource(solve)==inspect.getsource(spatial.electrical.solve)
    assert inspect.getsource(circuit)==inspect.getsource(spatial.circuit)
    def class_source(path):
        text=Path(path).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='System')
        return ast.get_source_segment(text,node)
    assert class_source(__file__)==class_source(spatial.__file__)
    parent_text=(ORIGINAL/'run_refinement.py').read_text()
    parent_ast=ast.parse(parent_text)
    def parent_function(name):
        return ast.get_source_segment(parent_text,next(n for n in parent_ast.body if isinstance(n,ast.FunctionDef) and n.name==name))+'\n'
    for f in [lift_map,boundary_operator,expand_state,compress_state,finite_full,finite_symmetric]:
        source=inspect.getsource(f)
        if source.startswith('@'):source=source[source.index('def '):]
        assert source==parent_function(f.__name__)
    assert inspect.getsource(finite_history)==inspect.getsource(previous.finite_history).replace('spatial.advance(','advance_precise(')
    baseline=old['baseline_tasks'];known=next(t for t in baseline if t['kind']=='known' and t['cathode']=='P-LCO' and t['rate_c']==2.)
    pilot=[dict(known,solver=k,control=True,reference_path=str(baseline_path(known).relative_to(ROOT))) for k in ['full','symmetric']]
    tasks=pilot+[dict(t,n=128,solver='symmetric') for t in baseline]
    assert len(tasks)==len(set(identifier(t) for t in tasks))==13
    hashes=old['sha256'].copy()
    paths=[Path(__file__),HERE/'verify_and_summarize.py',HERE/'PLAN.md',ORIGINAL/'manifest.json',ORIGINAL/'controls.json',ORIGINAL/'pilot_summary.json',ORIGINAL/'pilot_execution.json']
    paths+=sorted((ORIGINAL/'cases').glob('*.json'))+sorted((ORIGINAL/'checkpoints').glob('*.json'))+sorted(HERE.glob('*_job.json'))
    for path in paths:hashes[str(path.relative_to(ROOT))]=sha(path)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    out=dict(sha256=hashes,witness=w,tasks=tasks,baseline_tasks=baseline,domains=old['domains'],grids=[64,128],mode=2,duplicate_factor=4,
        known_precision=dict(w_rtol=1e-10,w_atol=1e-12,g_rtol=1e-12,g_atol=1e-14),
        parameters_refitted=False,central_claim_changed=False,criteria_relaxed=False,original_pilot_failed=True,previous_failed_grid_comparisons_preserved=True)
    path=HERE/'manifest.json'
    if path.exists():assert read(path)==out
    else:save(path,out)
    return out


def regression(c,reference=None):
    t=c['task'];h=c['result'];old=read(ROOT/t['reference_path'])['result'] if reference is None else reference['result']
    initial=float(np.max(abs(np.array(h['initial_contact'])-old['initial_contact'])));ratios=[];effects=[];contact=[];ledger=[]
    for r,s in zip(h['rows'],old['rows']):
        for route in ['primary','independent']:
            rr,ss=([r[route]],[s[route]]) if t['kind']=='known' else (r[route]['worlds'],s[route]['worlds'])
            if t['kind']!='known':effects.append(abs(r[route]['effect']-s[route]['effect']))
            for a,b in zip(rr,ss):
                ratios.append(abs(a['ratio']-b['ratio']));contact.append(float(np.max(abs(np.array(a['contact'])-b['contact']))))
                for name in ['stripped_charge_mAh_cm2','replenished_volume_um']:
                    original=np.array(b[name]);ledger.append(float(np.max(abs(np.array(a[name])-original)/np.maximum(1.,abs(original)))))
    passed=bool(h['available'] and initial<=1e-12 and max(ratios)<=1e-7 and max(contact)<=1e-8 and max(ledger)<=1e-7 and max(effects,default=0.)<=1e-9)
    return dict(initial_error=initial,ratio_error=max(ratios),contact_error=max(contact),ledger_relative_error=max(ledger),
        effect_error=max(effects,default=0.),passed=passed)


def case(t):return read(HERE/'cases'/(identifier(t)+'.json'))


def known_gate(m):
    ts=[t for t in m['tasks'] if t['kind']=='known' and t['n']==128];assert len(ts)==7
    if not all((HERE/'cases'/(identifier(t)+'.json')).exists() for t in ts):return dict(evaluated=False,passed=False,reason='Known n128 cases incomplete')
    cc=[case(t) for t in ts]
    if not all(c['available'] for c in cc):return dict(evaluated=True,passed=False,reason='Known n128 case unavailable')
    errors=[];grid=[]
    for t,c in zip(ts,cc):
        old=read(baseline_path(dict(t,n=64)))
        for a,b in zip(c['result']['rows'],old['result']['rows']):
            errors.append(dict(cathode=t['cathode'],rate_c=t['rate_c'],stage=a['stage'],error=a['error']))
            grid.append(abs(a['ratio']-b['ratio']))
    worst=max(errors,key=lambda r:abs(r['error']));maximum=abs(worst['error'])
    return dict(evaluated=True,n=128,comparison_n=64,maximum_absolute_error=maximum,worst=worst,grid_difference=max(grid),
        errors=errors,passed=bool(maximum<=.03 and max(grid)<=1e-4))


def eligibility(t,m):
    original=read(ORIGINAL/'pilot_summary.json')
    if not original['audit_passed'] or not original['static_controls_passed']:return False,'Original full-field/static controls failed'
    if t.get('control'):return True,None
    p=HERE/'pilot_summary.json'
    if not p.exists() or not read(p)['continuation_admitted']:return False,'New precision test not independently qualified'
    if t['kind']=='known':return True,None
    p=HERE/'known_summary.json'
    if not p.exists() or not read(p)['audit_passed'] or not read(p)['known_gate']['passed']:return False,'Known n128 audit or compatibility gate failed/incomplete'
    return True,None


def one(t):
    global SYMMETRY_ERROR
    checkpoint_guard();tic=time.perf_counter();m=read(HERE/'manifest.json');path=HERE/'cases'/(identifier(t)+'.json')
    if path.exists():
        out=read(path);assert out['task']==t and out['manifest_sha256']==sha(HERE/'manifest.json')
        return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=True)
    SYMMETRY_ERROR=0.;out=dict(task=t,manifest_sha256=sha(HERE/'manifest.json'))
    try:
        if t['kind']=='known':result=finite_history(m['witness'],t)
        elif t['solver']=='full':result=finite_full(m['witness'],t)
        else:result=finite_symmetric(m['witness'],t)
        out.update(result=result,available=result['available'])
        if t.get('control'):out['regression']=regression(out)
        if t['solver']=='symmetric':out['reduction']=dict(duplicate_factor=4,full_grid=t['n'],ode_cells=t['n']//4,discarded_symmetry_relative_error=SYMMETRY_ERROR)
    except Exception as error:
        out.update(available=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    out['wall_seconds']=time.perf_counter()-tic;save(path,out)
    return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','known','future'],required=True)
    phase=parser.parse_args().phase;checkpoint_guard();tic=time.perf_counter();m=freeze()
    if phase=='pilot':intended=[t for t in m['tasks'] if t.get('control')]
    else:intended=[t for t in m['tasks'] if not t.get('control') and t['kind']==('known' if phase=='known' else 'paired')]
    tasks=[];skipped=[]
    for t in intended:
        allowed,reason=eligibility(t,m)
        if allowed:tasks.append(t)
        else:skipped.append(dict(task=t,reason=reason))
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,t) for t in tasks]):
            row=future.result();done.append(row);print(json.dumps(row),flush=True);checkpoint_guard()
    out=dict(phase=phase,manifest_sha256=sha(HERE/'manifest.json'),completed=sorted(done,key=lambda r:r['id']),skipped=skipped,
        workers=workers,wall_seconds=time.perf_counter()-tic)
    if phase=='known':out['gate']=known_gate(m)
    save(HERE/(phase+'_execution.json'),out)


if __name__=='__main__':main()
