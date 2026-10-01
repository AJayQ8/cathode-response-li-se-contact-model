"""Exploratory patchwise finite-height kinetics with conserving transport."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import importlib.util
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq
from scipy.special import expit
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


projection = module('scalar_projection', HERE.parent/'prospective27/run_projection.py')
electrical = module('spatial_electrical', HERE.parent/'coupling5/electrical_boundary.py')
model, assessment = projection.model, projection.assessment
read, save, sha = model.read, model.save, model.sha
CONFIGS = [('uniform', .5), ('cosine', 0.), ('cosine', .5), ('cosine', 1.)]


def identifier(task):
    return task['witness_id']+'_'+task['shape']+'_f'+str(task['fraction']).replace('.','p')+'_n'+str(task['n'])


def freeze():
    bridge = read(HERE.parent/'bridge31/manifest_v2.json')
    assert read(HERE.parent/'bridge31/summary.json')['all_independent_checks_pass']
    witnesses = [w for wid in bridge['representative_ids'] for w in bridge['witnesses'] if w['id']==wid]
    tasks = [dict(witness_id=w['id'], shape=shape, fraction=f, n=n)
             for w in witnesses for shape,f in CONFIGS for n in [16,32]]
    hashes = bridge['sha256'].copy()
    for p in [Path(__file__).resolve(), HERE/'PLAN.md', HERE.parent/'bridge31/manifest_v2.json',
              HERE.parent/'bridge31/summary.json', HERE.parent/'bridge31/verify_and_summarize.py']:
        hashes[str(p.relative_to(ROOT))] = sha(p)
    for name,digest in hashes.items():
        assert sha(ROOT/name)==digest,name
    out = dict(sha256=hashes,witnesses=witnesses,tasks=tasks,new_patchwise_kinetic_hypothesis=True,
               source_parameters_refitted=False,all_known_points_are_retrospective=True,
               central_claim_changed=False,prospective_prediction_qualified=False)
    path = HERE/'manifest.json'
    if path.exists():
        assert read(path)==out
    else:
        save(path,out)
    return out


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


def resistance(a,Rb,C,fraction):
    if fraction==0 or Rb==0:
        return Rb+C/np.mean(a)
    diag,_,_=electrical.solve(a,1.,fraction*Rb/C)
    return (1-fraction)*Rb+fraction*Rb*diag['resistance_over_bulk']


def initial_profile(x,rstar,Rch,task):
    n=task['n'];b,A=x[3:]
    Rb,C=b*rstar,A*(1-b)*rstar
    z=np.zeros(n) if task['shape']=='uniform' else 2*np.cos(2*np.pi*(np.arange(n)+.5)/n)
    def f(shift):return resistance(expit(shift+z),Rb,C,task['fraction'])/Rch-1
    shift=brentq(f,-35.,35.,xtol=1e-12,rtol=1e-14)
    a=expit(shift+z)
    assert abs(f(shift))<=1e-9
    return a,float(shift)


def advance(y,duration,x,rstar,task,cathode,qstart,current,coordinate):
    n=task['n'];A=x[4];L=10**x[0]
    sys=System(x,rstar,task['fraction'],n,cathode,qstart,current,coordinate)
    def event(t,state):
        return np.min(state[:n])-model.ZMIN**2 if coordinate=='w' else L*(1-A*model.ZMIN)-np.max(state[:n])
    event.terminal=True;event.direction=-1
    if event(0,y)<=0:
        return y.copy(),True,0.,0.,0
    sol=solve_ivp(sys.rhs,(0.,duration),y,method='Radau',jac=sys.jac,
                  rtol=1e-9 if coordinate=='w' else 1e-11,atol=1e-11 if coordinate=='w' else 1e-13,
                  first_step=min(duration,1e-4),events=event)
    assert sol.success and np.all(np.isfinite(sol.y[:,-1])),sol.message
    result=sol.y[:,-1]
    projection_size=max(0.,float(np.max(result[:n])-1/(A*A))) if coordinate=='w' else max(0.,float(-np.min(result[:n])))*2/(L*A*A)
    assert projection_size<=1e-7*max(1.,1/(A*A))
    if coordinate=='w':result[:n]=np.clip(result[:n],model.ZMIN**2,1/(A*A))
    else:result[:n]=np.clip(result[:n],0.,L*(1-A*model.ZMIN))
    return result,bool(len(sol.t_events[0])),float(sol.t[-1]),projection_size,sys.evaluations


def observation(y,coordinate,x,rstar,Rch,task,a0,stripped,stopped,projection_size):
    n=task['n'];L=10**x[0];b,A=x[3:]
    a=A*np.sqrt(y[:n]) if coordinate=='w' else 1-y[:n]/L
    V=L*a*a/2;V0=L*a0*a0/2;charge=y[n:2*n];K=y[2*n:]
    ledger=np.abs(V-V0+model.U*charge-K)
    physical=bool(not stopped and np.all(a>0) and np.all(a<=1+1e-12) and np.all(L*a<=40+1e-10) and
                  np.all(V+model.U*charge<=40+model.U*1e-7) and np.all(K>=-1e-7))
    numeric=bool(np.all(ledger<=1e-6*np.maximum(1.,np.maximum(np.abs(V),np.abs(K)))) and
                 abs(float(np.mean(charge))-stripped)<=1e-6 and projection_size<=1e-7*max(1.,1/(A*A)))
    R=None if stopped else float(resistance(np.clip(a,0.,1.),b*rstar,A*(1-b)*rstar,task['fraction']))
    return dict(contact=a.tolist(),volume_um=V.tolist(),initial_volume_um=V0.tolist(),
                stripped_charge_mAh_cm2=charge.tolist(),replenished_volume_um=K.tolist(),
                nominal_stripped_charge_mAh_cm2=stripped,maximum_ledger_error_um=float(np.max(ledger)),
                stopped=stopped,resistance=R,ratio=None if R is None else R/Rch,
                physical_pass=physical,numerical_pass=numeric,projection=projection_size)


def protocol(x,rstar,p,task,new=False):
    checkpoint_guard()
    a0,shift=initial_profile(x,rstar,p['R_ch'],task)
    assert model.domain(p,x[2])['valid']
    n=task['n'];L=10**x[0];A=x[4]
    outputs={};evaluations={}
    for coordinate in ['w','g']:
        y=np.r_[(a0/A)**2 if coordinate=='w' else L*(1-a0),np.zeros(2*n)]
        snapshots={};stopped=False;stripped=0.;qstart=p['q0'];max_projection=0.;calls=0
        segments=[('first',p['t1'],p['J']),('rest',p['rest'],0.)]
        if not new:segments.append(('second',p['t2'],p['J']))
        for stage,duration,current in segments:
            if not stopped:
                y,stopped,elapsed,correction,counts=advance(y,duration,x,rstar,task,p['cathode'],qstart,current,coordinate)
            else:elapsed=0.;correction=0.;counts=0
            stripped+=current*elapsed;qstart+=current*elapsed
            max_projection=max(max_projection,correction);calls+=counts
            snapshots[stage]=observation(y,coordinate,x,rstar,p['R_ch'],task,a0,stripped,stopped,max_projection)
            checkpoint_guard()
        outputs[coordinate]=snapshots;evaluations[coordinate]=calls
    stages=['first','rest'] if new else ['rest','second']
    rows=[]
    for i,stage in enumerate(stages):
        fast,direct=outputs['w'][stage],outputs['g'][stage]
        same=fast['stopped']==direct['stopped']
        delta=None if fast['ratio'] is None or direct['ratio'] is None else abs(fast['ratio']-direct['ratio'])
        numeric=bool(same and (delta is None or delta<=1e-4) and fast['numerical_pass'] and direct['numerical_pass'])
        label=['end_discharge','after_30_min_rest'][i] if new else p['targets'][i]['stage']
        observed=None if new else p['targets'][i]['ratio']
        rows.append(dict(stage=label,observed_ratio=observed,ratio=direct['ratio'],
                         error=None if observed is None or direct['ratio'] is None else direct['ratio']-observed,
                         primary=fast,independent=direct,ratio_difference=delta,numerical_pass=numeric,
                         physical_pass=fast['physical_pass'] and direct['physical_pass']))
    return dict(cathode=p['cathode'],rate_c=p['rate_c'],R_ch=p['R_ch'],q0=p['q0'],initial_shift=shift,
                initial_contact=a0.tolist(),evaluations=evaluations,rows=rows,
                available=all(r['numerical_pass'] and r['physical_pass'] and r['ratio'] is not None for r in rows))


def controls(frozen):
    path=HERE/'controls.json'
    if path.exists():
        old=read(path)
        assert old['passed'] and old['manifest_sha256']==sha(HERE/'manifest.json')
        return
    w=next(w for w in frozen['witnesses'] if w['setting']['extraction']=='ordinary')
    x=np.array(w['x']);rstar=w['rstar'];n=8
    a=np.linspace(.35,.65,n);beta=x[3]/(x[4]*(1-x[3]));tests=[]
    q,jac,_=circuit(a,1.2,beta,True)
    fd=np.empty_like(jac)
    for i in range(n):
        d=np.eye(n)[i]*1e-5
        fd[:,i]=(circuit(a+d,1.2,beta)[0]-circuit(a-d,1.2,beta)[0])/(2e-5)
    discrepancy=float(np.max(np.abs(fd-jac))/max(1.,np.max(np.abs(fd))))
    balance=float(np.max(np.abs(np.sum(jac,axis=0))))
    tests.append(dict(name='electrical_current_jacobian',scaled_error=discrepancy,column_sum_error=balance,
                      passed=bool(discrepancy<=.001 and balance<=1e-9)))
    for fraction in [0.,1.]:
        for coordinate in ['w','g']:
            system=System(x,rstar,fraction,n,'P-LCO',0.,1.2,coordinate)
            state=np.r_[(a/x[4])**2 if coordinate=='w' else system.L*(1-a),np.zeros(2*n)]
            exact=system.jac(0.,state)
            differences=[]
            for i in range(n):
                h=1e-5*max(1.,abs(state[i]));d=np.eye(3*n)[i]*h
                fd=(system.rhs(0.,state+d)-system.rhs(0.,state-d))/(2*h)
                differences.append(float(np.max(np.abs(fd-exact[:,i]))/max(1.,np.max(np.abs(fd)))))
            assert np.all(exact[:,n:]==0.)
            rhs=system.rhs(0.,state)
            dvolume=.5*system.L*x[4]**2*rhs[:n] if coordinate=='w' else -a*rhs[:n]
            ledger=float(np.max(np.abs(dvolume+model.U*rhs[n:2*n]-rhs[2*n:])))
            null=System(x,rstar,fraction,n,'P-LCO',0.,0.,coordinate)
            zero_charge=float(np.max(np.abs(null.rhs(0.,state)[n:2*n])))
            tests.append(dict(name=coordinate+'_jacobian_f'+str(fraction),scaled_errors=differences,
                              differential_ledger_error=ledger,zero_current_charge_rate=zero_charge,
                              passed=bool(max(differences)<=.001 and ledger<=1e-9 and zero_charge==0.)))
            checkpoint_guard()
    out=dict(tests=tests,passed=all(t['passed'] for t in tests),manifest_sha256=sha(HERE/'manifest.json'))
    save(path,out)
    assert out['passed'],'Spatial analytic/volume controls failed'


def reference_points(witness):
    saved=read(ROOT/witness['source'])
    if witness['source_label'] is None:
        w=saved['witness'];fit=read(ROOT/w['source'])
        return next(c['points'] for c in fit['candidates'] if c['label']==w['label'])
    return next(c['points'] for c in saved['candidates'] if c['label']==witness['source_label'])


def run(task,witness):
    checkpoint_guard();tic=time.perf_counter()
    path=HERE/'cases'/(identifier(task)+'.json')
    if path.exists():
        old=read(path)
        assert old['task']==task and old['witness']==witness and old['manifest_sha256']==sha(HERE/'manifest.json')
        return old
    assert witness['setting']['timing']=='late'
    x=np.array(witness['x']);rstar,protocols=model.data(witness['setting']['extraction'],witness['setting']['registration'])
    assert rstar==witness['rstar']
    known=[];new=[]
    for p in protocols:
        if p['excluded_ambiguous']:continue
        result=protocol(x,rstar,p,task)
        result['subset']=p['subset'];known.append(result)
    for cathode in ['P-LCO','N-LCO']:
        original=next(p for p in protocols if p['cathode']==cathode and p['rate_c']==2.)
        p=dict(original,J=projection.J,t1=projection.Q/projection.J,t2=0.,rest=projection.REST)
        new.append(protocol(x,rstar,p,task,True))
    known_rows=[r for h in known for r in h['rows']]
    assert len(known_rows)==14
    maximum=max(100. if r['error'] is None else abs(r['error']) for r in known_rows)
    numerical=all(r['numerical_pass'] for h in known+new for r in h['rows'])
    physical=all(r['physical_pass'] for h in known+new for r in h['rows'])
    uniform=[]
    if task['shape']=='uniform':
        refs=reference_points(witness)
        for h in known:
            for r in h['rows']:
                ref=next(v for v in refs if (v['cathode'],v['rate_c'],v['stage'])==(h['cathode'],h['rate_c'],r['stage']))
                same=(r['ratio'] is None)==(ref['predicted_ratio'] is None)
                difference=None if r['ratio'] is None or ref['predicted_ratio'] is None else abs(r['ratio']-ref['predicted_ratio'])
                uniform.append(dict(cathode=h['cathode'],rate_c=h['rate_c'],stage=r['stage'],new=False,
                                    difference=difference,passed=same and (difference is None or difference<=1e-4)))
        for h in new:
            reference=next(v for v in witness['diagonal_histories'] if v['cathode']==h['cathode'])
            for r,ref in zip(h['rows'],reference['rows']):
                same=(r['ratio'] is None)==(ref['ratio'] is None)
                difference=None if r['ratio'] is None or ref['ratio'] is None else abs(r['ratio']-ref['ratio'])
                uniform.append(dict(cathode=h['cathode'],stage=r['stage'],new=True,difference=difference,
                                    passed=same and (difference is None or difference<=1e-4)))
    out=dict(task=task,witness=witness,known=known,new_protocol=new,maximum_known_error=maximum,
             all_known_screen_pass=bool(maximum<=.03 and all(h['available'] for h in known)),
             numerical_pass=bool(numerical),physical_pass=bool(physical),uniform_controls=uniform,
             uniform_controls_pass=all(c['passed'] for c in uniform),wall_seconds=time.perf_counter()-tic,
             manifest_sha256=sha(HERE/'manifest.json'),prospective_prediction_qualified=False)
    save(path,out)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();checkpoint_guard();tic=time.perf_counter()
    frozen=freeze();controls(frozen)
    ordinary=next(w['id'] for w in frozen['witnesses'] if w['setting']['extraction']=='ordinary')
    tasks=[t for t in frozen['tasks'] if not args.pilot or (t['witness_id']==ordinary and t['n']==16 and
            ((t['shape']=='uniform' and t['fraction']==.5) or (t['shape']=='cosine' and t['fraction']==1.)))]
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run,t,next(w for w in frozen['witnesses'] if w['id']==t['witness_id'])) for t in tasks]
        for future in as_completed(futures):
            item=future.result();done.append(identifier(item['task']))
            print(json.dumps(dict(id=identifier(item['task']),max_known_error=item['maximum_known_error'],
                                  numerical_pass=item['numerical_pass'],physical_pass=item['physical_pass'],
                                  uniform_pass=item['uniform_controls_pass'],seconds=item['wall_seconds'])),flush=True)
            checkpoint_guard()
    save(HERE/('pilot_execution.json' if args.pilot else 'execution.json'),dict(completed=done,workers=workers,
             wall_seconds=time.perf_counter()-tic,manifest_sha256=sha(HERE/'manifest.json')))


if __name__=='__main__':
    main()
