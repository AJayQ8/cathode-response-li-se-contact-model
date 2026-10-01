"""Frozen, finite-amplitude contact-wavelength pilot without refitting."""
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
spec=importlib.util.spec_from_file_location('wavelength_previous',HERE.parent/'depth41/run_depth.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
prior=previous.prior
weak,spatial,model,projection=previous.weak,previous.spatial,previous.model,previous.projection
read,save,sha=previous.read,previous.save,previous.sha
loading_domain=previous.loading_domain
STAGES=weak.STAGES


def identifier(t):
    if t['kind']=='known':return weak.identifier(t)+'_k'+str(t['mode'])
    return previous.identifier(t)+'_k'+str(t['mode'])


def baseline_path(t):
    if t['Q']==.3:return previous.baseline_path(t)
    return previous.HERE/'cases'/(previous.identifier(t)+'.json')


def freeze():
    old=read(previous.HERE/'manifest.json');w=next(w for w in old['witnesses'] if w['id']=='w026')
    assert inspect.getsource(finite)==inspect.getsource(previous.finite).replace('2*np.pi*(np.arange(n)+.5)/n',"2*np.pi*t['mode']*(np.arange(n)+.5)/n")
    expected=inspect.getsource(weak.finite_history).replace('2*np.pi*(np.arange(n)+.5)/n',"2*np.pi*task['mode']*(np.arange(n)+.5)/n").replace('4*np.pi*(np.arange(n)+.5)/n',"4*np.pi*task['mode']*(np.arange(n)+.5)/n")
    assert inspect.getsource(finite_history)==expected
    rstar,protocols=model.data(w['setting']['extraction'],w['setting']['registration']);assert abs(rstar-w['rstar'])<1e-12
    tasks=[];domains=[]
    for n in [32,64]:
        for p in protocols:
            if p['excluded_ambiguous']:continue
            t=dict(kind='known',witness_id=w['id'],cathode=p['cathode'],rate_c=p['rate_c'],fraction=1.,n=n,amplitude=.25,mode=2)
            tasks.append(t);domains.append(dict(task=t,loading=model.domain(p,w['x'][2])))
        for Q in [.3,.6]:
            for history in weak.CATHODES:
                t=dict(kind='paired',witness_id=w['id'],initial='P-LCO',history=history,n=n,epsilon=.25,J=1.2,Q=Q,rest=.5,mode=2)
                tasks.append(t);domains.append(dict(task=t,loading=loading_domain(w,t)))
    known=dict(next(t for t in tasks if t['kind']=='known' and t['n']==32 and t['cathode']=='P-LCO' and t['rate_c']==2.),mode=1,control=True)
    paired=dict(next(t for t in tasks if t['kind']=='paired' and t['n']==32 and t['history']=='P-LCO' and t['Q']==.6),mode=1,control=True)
    known['regression_source']=str((HERE.parent/'moderate38/cases'/(weak.identifier(known)+'.json')).relative_to(ROOT))
    paired['regression_source']=str(baseline_path(paired).relative_to(ROOT))
    tasks=[known,paired]+tasks
    baseline=[dict(t,mode=1) for t in old['tasks']+old['baseline_tasks'] if t['witness_id']=='w026' and t['J']==1.2 and t['initial']=='P-LCO']
    assert len(tasks)==24 and len(baseline)==8 and len(set(identifier(t) for t in tasks+baseline))==31
    assert all(d['loading']['valid'] for d in domains)
    hashes=old['sha256'].copy()
    paths=[Path(__file__),HERE/'verify_and_summarize.py',HERE/'PLAN.md',previous.HERE/'manifest.json',previous.HERE/'all_summary.json',
        HERE.parent/'moderate38/all_summary.json',HERE.parent/'weak33/controls.json']
    paths += [ROOT/t['regression_source'] for t in tasks if t.get('control')]+[baseline_path(t) for t in baseline]+sorted(HERE.glob('*_job.json'))
    for p in paths:hashes[str(p.relative_to(ROOT))]=sha(p)
    for p,digest in hashes.items():assert sha(ROOT/p)==digest,p
    out=dict(sha256=hashes,witness=w,tasks=tasks,baseline_tasks=baseline,domains=domains,grids=[32,64],modes=[1,2],depths=[.3,.6],
        parameters_refitted=False,central_claim_changed=False,response_drafting_deferred=True,initial_mean_rematched=True)
    path=HERE/'manifest.json'
    if path.exists():assert read(path)==out
    else:save(path,out)
    return out


def finite(w,t):
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
                state, stopped, elapsed, correction, count = spatial.advance(state, duration, x, rstar, task, p['cathode'], qstart, current, coordinate)
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


def controls(m):
    checkpoint_guard();w=m['witness'];x=np.array(w['x']);L=10**x[0];tests=[]
    _,protocols=model.data(w['setting']['extraction'],w['setting']['registration'])
    for cathode in weak.CATHODES:
        p=next(p for p in protocols if p['cathode']==cathode and p['rate_c']==2.)
        _,_,a=model.initial_state(x,w['rstar'],p['R_ch']);pressure=2+x[2]*model.pressure_increment(cathode,p['q0'])
        for n in [32,64]:
            for fraction in [0.,1.]:
                mode=2;system=spatial.System(x,w['rstar'],fraction,n,cathode,p['q0'],1.2,'g')
                y=np.r_[np.full(n,L*(1-a)),np.zeros(2*n)];jac=system.jac(0.,y)[:n,:n]
                v=np.cos(2*np.pi*mode*(np.arange(n)+.5)/n)
                variant=[dict(fraction=fraction,mode=mode,grid=n)]
                gamma,_,hp,_,feedback,_=weak.coefficients(x,a,pressure,1.2,variant)
                expected=float(gamma*hp+feedback[0]);action=jac@v;actual=float(np.dot(v,action)/np.dot(v,v))
                step=1e-5*min(a,1-a);d=np.r_[-L*step*v,np.zeros(2*n)]
                fd=-(system.rhs(0.,y+d)[:n]-system.rhs(0.,y-d)[:n])/(2*L*step)
                derivative=float(np.dot(v,fd)/np.dot(v,v))
                error=max(abs(actual-expected),abs(derivative-expected))/max(1.,abs(expected))
                leakage=float(np.linalg.norm(action-expected*v)/max(1.,np.linalg.norm(action)))
                mean_expected=gamma*hp+model.U*1.2/(L*a*a)
                mean_error=float(np.max(np.abs(jac@np.ones(n)-mean_expected))/max(1.,abs(mean_expected)))
                rest=float(weak.coefficients(x,a,pressure,0.,variant)[4][0])
                tests.append(dict(cathode=cathode,n=n,fraction=fraction,mode=mode,formula=expected,spatial_jacobian=actual,
                    finite_difference=derivative,scaled_error=error,leakage=leakage,mean_mode_error=mean_error,zero_current_feedback=rest,
                    passed=bool(error<=1e-5 and leakage<=1e-9 and mean_error<=1e-9 and rest==0.)))
        checkpoint_guard()
    out=dict(manifest_sha256=sha(HERE/'manifest.json'),tests=tests,passed=all(t['passed'] for t in tests))
    save(HERE/'controls.json',out);return out


def case(t):return read(HERE/'cases'/(identifier(t)+'.json'))


def known_gate(m,n):
    ts=[t for t in m['tasks'] if t['kind']=='known' and t['mode']==2 and t['n']==n];assert len(ts)==7
    if not all((HERE/'cases'/(identifier(t)+'.json')).exists() for t in ts):
        return dict(n=n,evaluated=False,passed=False,reason='Known trajectories not all completed')
    cc=[case(t) for t in ts]
    if not all('result' in c and c['available'] for c in cc):
        return dict(n=n,evaluated=True,passed=False,reason='Incomplete or unavailable known trajectory',all_available=False)
    rows=[r for c in cc for r in c['result']['rows']];maximum=max(abs(r['error']) for r in rows);changes=[]
    if n==64:
        for t,c in zip(ts,cc):changes.extend(abs(r['ratio']-s['ratio']) for r,s in zip(c['result']['rows'],case(dict(t,n=32))['result']['rows']))
    return dict(n=n,evaluated=True,maximum_absolute_error=maximum,all_available=True,grid_difference=None if not changes else max(changes),
        passed=bool(maximum<=.03 and (not changes or max(changes)<=1e-4)))


def eligibility(t,m):
    if t.get('control'):return True,None
    path=HERE/'controls.json'
    if not path.exists() or not read(path)['passed']:return False,'Static mode controls not passed'
    regressions=[u for u in m['tasks'] if u.get('control')]
    if not all((HERE/'cases'/(identifier(u)+'.json')).exists() and case(u)['available'] and case(u).get('regression',{}).get('passed',False) for u in regressions):
        return False,'Wrapper regression controls not passed'
    if t['kind']=='known' and t['n']==32:return True,None
    if not known_gate(m,32)['passed']:return False,'k=2 n32 known-data gate failed or incomplete'
    if t['kind']=='known':return True,None
    if not known_gate(m,64)['passed']:return False,'k=2 n64 known-data gate failed or incomplete'
    if t['n']==64:
        coarse=[u for u in m['tasks'] if u['kind']=='paired' and u['mode']==2 and u['Q']==t['Q'] and u['n']==32]
        assert len(coarse)==2
        if not all((HERE/'cases'/(identifier(u)+'.json')).exists() and case(u)['available'] for u in coarse):
            return False,'A coarse paired history failed or is incomplete at this depth'
    return True,None


def regression(c):
    t=c['task'];h=c['result'];old=read(ROOT/t['regression_source'])['result'];ratios=[];effects=[]
    initial=float(np.max(np.abs(np.array(h['initial_contact'])-old['initial_contact'])))
    for r,s in zip(h['rows'],old['rows']):
        for route in ['primary','independent']:
            if t['kind']=='known':ratios.append(abs(r[route]['ratio']-s[route]['ratio']))
            else:
                ratios.extend(abs(a['ratio']-b['ratio']) for a,b in zip(r[route]['worlds'],s[route]['worlds']))
                effects.append(abs(r[route]['effect']-s[route]['effect']))
    return dict(source=t['regression_source'],initial_difference=initial,ratio_difference=max(ratios),effect_difference=max(effects,default=0.),
        passed=bool(h['available'] and initial<=1e-12 and max(ratios)<=1e-7 and max(effects,default=0.)<=1e-8))


def one(t):
    checkpoint_guard();tic=time.perf_counter();m=read(HERE/'manifest.json');path=HERE/'cases'/(identifier(t)+'.json')
    if path.exists():
        out=read(path);assert out['task']==t and out['manifest_sha256']==sha(HERE/'manifest.json')
        return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=True)
    out=dict(task=t,manifest_sha256=sha(HERE/'manifest.json'))
    try:
        result=finite_history(m['witness'],t) if t['kind']=='known' else finite(m['witness'],t)
        out.update(result=result,available=result['available'])
        if t.get('control'):out['regression']=regression(out)
    except Exception as error:
        out.update(available=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    out['wall_seconds']=time.perf_counter()-tic;save(path,out)
    return dict(id=identifier(t),available=out['available'],seconds=out['wall_seconds'],reused=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['controls','known32','known64','future32','future64'],required=True)
    phase=parser.parse_args().phase;checkpoint_guard();tic=time.perf_counter();m=freeze()
    if phase=='controls':
        static=controls(m)
        if not static['passed']:raise AssertionError('Static mode controls failed; results retained')
        intended=[t for t in m['tasks'] if t.get('control')]
    else:
        kind='known' if phase.startswith('known') else 'paired';n=int(phase[-2:])
        intended=[t for t in m['tasks'] if t['kind']==kind and t['mode']==2 and t['n']==n]
    tasks=[];skipped=[]
    for t in intended:
        allowed,reason=eligibility(t,m)
        if allowed:tasks.append(t)
        else:skipped.append(dict(task=t,reason=reason))
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,t) for t in tasks]):
            item=future.result();done.append(item);print(json.dumps(item),flush=True);checkpoint_guard()
    out=dict(phase=phase,manifest_sha256=sha(HERE/'manifest.json'),completed=sorted(done,key=lambda r:r['id']),skipped=skipped,
        workers=workers,wall_seconds=time.perf_counter()-tic)
    if phase.startswith('known'):out['gate']=known_gate(m,int(phase[-2:]))
    save(HERE/(phase+'_execution.json'),out)
    if 'gate' in out:print(json.dumps(out['gate']),flush=True)


if __name__=='__main__':main()
