"""Matched-interface second-order and finite aggregate feedback interactions."""
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
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
spec=importlib.util.spec_from_file_location('matched_weak',HERE.parent/'weak33/run_weak_response.py')
weak=importlib.util.module_from_spec(spec);spec.loader.exec_module(weak)
spatial,model,projection=weak.spatial,weak.model,weak.projection
read,save,sha=weak.read,weak.save,weak.sha
GRIDS=[16,32,'continuum']


def identifier(t):
    s=t['kind']+'_'+t['witness_id']+'_'+t['initial'][0]+t['history'][0]
    return s if t['kind']=='asymptotic' else s+'_n'+str(t['n'])+'_e'+str(t['epsilon']).replace('.','p')


def profile_path(t):
    return HERE.parent/'weak33/cases'/f"future_w026_{t['initial']}_c2p0_f1p0_n{t['n']}_e{str(t['epsilon']).replace('.','p')}.json"


def freeze():
    old=read(HERE.parent/'weak33/manifest.json');tasks=[]
    for w in old['witnesses']:
        for initial in weak.CATHODES:
            for history in weak.CATHODES:tasks.append(dict(kind='asymptotic',witness_id=w['id'],initial=initial,history=history))
    for initial in weak.CATHODES:
        for history in weak.CATHODES:
            for n in [16,32]:
                for epsilon in [.02,.01]:tasks.append(dict(kind='finite',witness_id='w026',initial=initial,history=history,n=n,epsilon=epsilon))
    hashes=old['sha256'].copy()
    paths=[Path(__file__).resolve(),HERE/'PLAN.md',HERE.parent/'weak33/manifest.json',HERE.parent/'weak33/summary.json']
    paths+=[HERE.parent/'weak33/cases'/('linear_'+w['id']+'.json') for w in old['witnesses']]
    paths+=[profile_path(t) for t in tasks if t['kind']=='finite']
    for p in paths:hashes[str(p.relative_to(ROOT))]=sha(p)
    for p,digest in hashes.items():assert sha(ROOT/p)==digest,p
    out=dict(sha256=hashes,witnesses=old['witnesses'],tasks=tasks,grids=GRIDS,
             same_initial_interface=True,same_resistance_observation=True,diagnostic_comparator=True,
             parameters_refitted=False,central_claim_changed=False,response_drafting_deferred=True)
    p=HERE/'manifest.json'
    if p.exists():assert read(p)==out
    else:save(p,out)
    return out


def coeff(a,gamma,D,bz):
    m=model.M;h=(1-a)*a**(-m);hp=-a**(-m)-m*(1-a)*a**(-m-1)
    hpp=m*a**(-m-2)*((m+1)-(m-1)*a)
    chi=bz*a/(1+bz*a)
    return h,hp,hpp,chi,gamma*hp+D/a**2


def setup(w,t):
    _,pp=model.data(w['setting']['extraction'],w['setting']['registration'])
    initial=next(p for p in pp if p['cathode']==t['initial'] and p['rate_c']==2.)
    history=next(p for p in pp if p['cathode']==t['history'] and p['rate_c']==2.)
    x=np.array(w['x']);a0=model.initial_state(x,w['rstar'],initial['R_ch'])[2]
    return x,a0,initial['R_ch'],history['q0']


class Moments:
    def __init__(self,w,t,coordinate,qstart,J):
        self.x,self.a0,self.Rch,_=setup(w,t);self.L,self.H=10**self.x[:2];self.eta,self.b,self.A=self.x[2:]
        self.bz=self.b/(self.A*(1-self.b))*np.array([weak.impedance(n,1) for n in GRIDS])
        self.u0=self.a0*(1-self.a0);self.coordinate=coordinate;self.qstart=qstart;self.J=J;self.cathode=t['history']
        self.tt,self.pp,self.t0,self.p0,_=projection.original_wave(self.cathode);self.calls=0
    def rhs(self,time,y):
        self.calls+=1
        if self.calls%2500==0:checkpoint_guard()
        raw=self.A*np.sqrt(max(0.,y[0])) if self.coordinate=='w' else 1-y[0]/self.L
        a=float(np.clip(raw,self.A*model.ZMIN/4,1.));q=self.qstart+self.J*time
        p=2+self.eta*(model.pressure_increment(self.cathode,q) if self.coordinate=='w' else
                      (float(np.interp(self.t0+q/.12,self.tt,self.pp))-self.p0)/1000)
        assert p>0
        gamma=self.H*self.A**(model.M+1)*(p/2)**model.M;D=model.U*self.J/self.L
        h,hp,hpp,chi,Fp=coeff(a,gamma,D,self.bz)
        F=gamma*h-D/a;first=2*a*F/self.A**2 if self.coordinate=='w' else -self.L*F
        uu=self.u0**2*np.exp(2*y[3]);diff=uu*np.expm1(2*y[4:7]);uon=uu+diff
        if self.coordinate=='w':
            mu=y[7:10];delta=y[10:13]
            middle=Fp*mu+.25*gamma*hpp*uu
            final=Fp*delta+.25*gamma*hpp*diff-D*chi*uon/(2*a**3)
        else:
            mu=(y[7:10]/self.L-uu/4)/a;delta=(y[10:13]/self.L-diff/4)/a
            middle=self.L*gamma*((h+a*hp)*mu+(2*hp+a*hpp)*uu/4)
            final=self.L*gamma*((h+a*hp)*delta+(2*hp+a*hpp)*diff/4)
        return np.r_[first,self.J,self.L*a*gamma*h,gamma*hp,D*chi/a**2,middle,final]


def moment_observation(y,sys,w):
    a=sys.A*np.sqrt(y[0]) if sys.coordinate=='w' else 1-y[0]/sys.L
    uu=sys.u0**2*np.exp(2*y[3]);du=uu*np.expm1(2*y[4:7]);chi=sys.bz*a/(1+sys.bz*a)
    mu=y[7:10] if sys.coordinate=='w' else (y[7:10]/sys.L-uu/4)/a
    delta=y[10:13] if sys.coordinate=='w' else (y[10:13]/sys.L-du/4)/a
    C=sys.A*(1-sys.b)*w['rstar'];Rb=sys.b*w['rstar']
    k=C/sys.Rch*(-delta/a**2+chi*du/(2*a**3))
    off=C/sys.Rch*(-mu/a**2+chi*uu/(2*a**3))
    ledger=abs(sys.L*(a*a-sys.a0**2)/2+model.U*y[1]-y[2])
    physical=bool(0<a<=1+1e-12 and sys.L*a<=40+1e-10 and sys.L*a*a/2+model.U*y[1]<=40+model.U*1e-7 and y[2]>=-1e-7)
    numeric=bool(ledger<=1e-6*max(1.,sys.L*a*a/2,abs(y[2])) and abs(y[1]-.3)<=1e-6)
    return dict(state=y.tolist(),a=float(a),ratio=float((Rb+C/a)/sys.Rch),u_limit_squared=float(uu),u_squared_difference=du.tolist(),
                mu_limit=mu.tolist(),delta_mu=delta.tolist(),limit_R_second_order=off.tolist(),
                feedback_R_second_order=k.tolist(),ledger_um=float(ledger),physical_pass=physical,numerical_pass=numeric)


def asymptotic(w,t):
    x,a0,Rch,q0=setup(w,t);L=10**x[0];A=x[4];u0=a0*(1-a0);routes={}
    for coordinate in ['w','g']:
        system=Moments(w,t,coordinate,q0,projection.J);chi=system.bz*a0/(1+system.bz*a0);mu=chi*u0*u0/(2*a0)
        initial=mu if coordinate=='w' else L*(a0*mu+u0*u0/4)
        y=np.r_[(a0/A)**2 if coordinate=='w' else L*(1-a0),np.zeros(6),initial,np.zeros(3)]
        rows=[];calls=0
        for stage,duration,J in [('end_discharge',.25,1.2),('after_30_min_rest',.5,0.)]:
            system=Moments(w,t,coordinate,q0 if J else q0+.3,J)
            sol=solve_ivp(system.rhs,(0.,duration),y,method='Radau' if coordinate=='w' else 'DOP853',
                          rtol=1e-10 if coordinate=='w' else 1e-11,atol=1e-12 if coordinate=='w' else 1e-13,first_step=1e-4)
            assert sol.success and np.all(np.isfinite(sol.y[:,-1]));y=sol.y[:,-1];calls+=system.calls
            rows.append(dict(stage=stage,**moment_observation(y,system,w)));checkpoint_guard()
        routes[coordinate]=dict(rows=rows,evaluations=calls)
    old=read(HERE.parent/'weak33/cases'/('linear_'+w['id']+'.json'))
    reference=next(h for h in old['result']['histories'] if h['initial_cathode']==t['initial'] and h['history_cathode']==t['history'])
    rows=[]
    for p,g,r in zip(routes['w']['rows'],routes['g']['rows'],reference['rows']):
        err=np.abs(np.array(p['feedback_R_second_order'])-g['feedback_R_second_order']);scale=np.maximum(1e-7,.001*np.abs(g['feedback_R_second_order']))
        olderr=abs(g['ratio']-r['independent']['ratio']);baseerr=abs(p['ratio']-g['ratio'])
        numeric=bool(p['numerical_pass'] and g['numerical_pass'] and olderr<=1e-4 and baseerr<=1e-6 and np.all(err<=scale))
        rows.append(dict(stage=p['stage'],primary=p,independent=g,reference_ratio_difference=olderr,
                         coordinate_ratio_difference=baseerr,coefficient_coordinate_difference=err.tolist(),numerical_pass=numeric,
                         physical_pass=p['physical_pass'] and g['physical_pass']))
    return dict(R_ch=Rch,initial_a=a0,q0=q0,rows=rows,evaluations={k:r['evaluations'] for k,r in routes.items()},
                available=all(r['numerical_pass'] and r['physical_pass'] for r in rows))


def finite(w,t):
    x,a,Rch,q0=setup(w,t);source=read(profile_path(t));a0=np.array(source['result']['initial_contact']);n=t['n'];L=10**x[0];A=x[4]
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
        prior=None if t['initial']!=t['history'] else abs(g['worlds'][1]['ratio']-source['result']['rows'][i]['independent']['ratio'])
        numeric=all(v['numerical_pass'] for route in [p,g] for v in route['worlds']) and re<=1e-6 and ee<=max(1e-9,.01*abs(g['effect'])) and (prior is None or prior<=1e-6)
        rows.append(dict(stage=p['stage'],primary=p,independent=g,coordinate_ratio_difference=re,effect_coordinate_difference=ee,
                         prior_resolved_ratio_difference=prior,numerical_pass=bool(numeric),physical_pass=all(v['physical_pass'] for route in [p,g] for v in route['worlds'])))
    return dict(initial_contact=a0.tolist(),R_ch=Rch,q0=q0,source_profile=str(profile_path(t).relative_to(ROOT)),rows=rows,
                evaluations={k:r['evaluations'] for k,r in routes.items()},available=all(r['numerical_pass'] and r['physical_pass'] for r in rows))


def controls(m):
    p=HERE/'controls.json'
    if p.exists():
        out=read(p);assert out['manifest_sha256']==sha(HERE/'manifest.json') and out['passed'];return
    rows=[]
    for wid in ['w026','w033','w037']:
        w=next(w for w in m['witnesses'] if w['id']==wid)
        for c in weak.CATHODES:
            t=dict(initial=c,history=c);x,a,Rch,q0=setup(w,t);L,H=10**x[:2];b,A=x[3:];C=A*(1-b)*w['rstar'];Rb=b*w['rstar']
            pressure=2+x[2]*model.pressure_increment(c,q0);gamma=H*A**(model.M+1)*(pressure/2)**model.M;D=model.U*1.2/L
            for n in [16,32]:
                v=np.cos(2*np.pi*(np.arange(n)+.5)/n);Z=weak.impedance(n,1)
                for f in [0.,1.]:
                    bz=f*b/(A*(1-b))*Z;h,hp,hpp,chi,Fp=coeff(a,gamma,D,np.array([bz]));chi=float(chi[0])
                    expected_mean=.25*gamma*hpp-D*chi/(2*a**3);expected_R=C*chi/(2*a**3)
                    errors=[];identity=[]
                    sys=spatial.System(x,w['rstar'],f,n,c,q0,1.2,'g')
                    def mean_rhs(aa):return float(np.mean(-sys.rhs(0.,np.r_[L*(1-aa),np.zeros(2*n)])[:n]/L))
                    F0=mean_rhs(np.full(n,a));R0=spatial.resistance(np.full(n,a),Rb,C,f)
                    for step in [.001*min(a,1-a),.0005*min(a,1-a)]:
                        ap,am=a+step*v,a-step*v
                        observed_mean=(mean_rhs(ap)+mean_rhs(am)-2*F0)/(2*step*step)
                        observed_R=(spatial.resistance(ap,Rb,C,f)+spatial.resistance(am,Rb,C,f)-2*R0)/(2*step*step)
                        errors.append(max(abs(observed_mean-expected_mean)/max(1.,abs(expected_mean)),abs(observed_R-expected_R)/max(1.,abs(expected_R))))
                        q=spatial.circuit(ap,1.2,f*Rb/C)[0]
                        identity.append(abs((Rb+C*np.mean(q/ap)/1.2)/spatial.resistance(ap,Rb,C,f)-1))
                    # Direct coefficient transform for a nonzero mean and variance.
                    uu=.017;mu=.023;muprime=Fp*mu+.25*gamma*hpp*uu-D*chi*uu/(2*a**3)
                    uprime_factor=gamma*hp+D*chi/a**2
                    via_contact=L*((gamma*h-D/a)*mu+a*muprime+uprime_factor*uu/2)
                    via_volume=L*gamma*((h+a*hp)*mu+(2*hp+a*hpp)*uu/4)
                    transform=abs(via_contact-via_volume)/max(1.,abs(via_volume))
                    rows.append(dict(witness=wid,cathode=c,n=n,fraction=f,derivative_errors=errors,resistance_identity_errors=identity,
                                     volume_transform_error=transform,zero_current_feedback=float(0.*chi/a**2),
                                     zero_beta_chi=float(coeff(a,gamma,D,np.array([0.]))[3][0]),
                                     passed=bool(max(errors)<=1e-4 and max(identity)<=1e-10 and transform<=1e-12)))
            checkpoint_guard()
    out=dict(rows=rows,passed=all(r['passed'] for r in rows),manifest_sha256=sha(HERE/'manifest.json'));save(p,out);assert out['passed']


def one(t):
    tic=time.perf_counter();m=read(HERE/'manifest.json');w=next(w for w in m['witnesses'] if w['id']==t['witness_id']);checkpoint_guard()
    result=asymptotic(w,t) if t['kind']=='asymptotic' else finite(w,t)
    out=dict(task=t,result=result,available=result['available'],wall_seconds=time.perf_counter()-tic,manifest_sha256=sha(HERE/'manifest.json'))
    save(HERE/'cases'/(identifier(t)+'.json'),out)
    return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','all'],required=True);a=parser.parse_args();tic=time.perf_counter()
    checkpoint_guard();m=freeze();controls(m)
    tasks=[t for t in m['tasks'] if a.phase=='all' or (t['kind']=='asymptotic' and t['witness_id'] in ['w026','w033','w037']) or
           (t['kind']=='finite' and t['initial']=='P-LCO' and t['n']==16 and t['epsilon']==.02)]
    pending=[];reused={}
    for t in tasks:
        p=HERE/'cases'/(identifier(t)+'.json')
        if p.exists():
            r=read(p);assert r['manifest_sha256']==sha(HERE/'manifest.json') and r['task']==t;reused[str(p.relative_to(HERE))]=sha(p)
        else:pending.append(t)
    records=[];workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,t) for t in pending]
        for future in as_completed(futures):
            r=future.result();records.append(r);print(json.dumps(r),flush=True);checkpoint_guard()
    save(HERE/(a.phase+'_execution.json'),dict(phase=a.phase,requested=len(tasks),reused_sha256=reused,new_results=records,
                                              wall_seconds=time.perf_counter()-tic,workers=workers,manifest_sha256=sha(HERE/'manifest.json')))


if __name__=='__main__':main()
