"""Conditional recovery/observation sensitivity, not the spatial paper model."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import time
import warnings
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares, brentq
from shared_compute import checkpoint_guard

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
OLD=HERE.parent/'kinetics7'
SPECTRAL=HERE.parent/'impedance8'
spec=importlib.util.spec_from_file_location('frozen_kinetics7',OLD/'run_kinetic_probe.py')
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
BOUNDS=[(-5.,2.),(-5.,5.),(-5.,5.),(0.,.99),(.01,1.)]
ZMIN=1e-6
EXTRACTIONS=['published','ordinary','robust']


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    t.replace(p)


def manifest():
    paths=[Path(__file__),HERE/'PLAN.md',HERE/'SOURCE_CONTEXT.md',OLD/'run_kinetic_probe.py',
        HERE.parent/'calibration2/resistance_rows.json',HERE.parent/'calibration2/time_segments.json']
    for loss in ['linear','soft_l1']:
        paths+=sorted((SPECTRAL/'fits').glob(f'*__source__{loss}__0.json'))
    return {'sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'numpy':np.__version__,'scipy':scipy.__version__,
        'bounds':BOUNDS,'seeds':[1729,2718],'m':[0.,6.6],'nu':[0,1],
        'source_pdf_sha256':sha(ROOT/'science_lab/.source_cache/lithium_creep_2019/lepage_2019_author_manuscript.pdf')}


def data(extraction):
    _,protocols=legacy.data()
    protocols=copy.deepcopy(protocols)
    if extraction!='published':
        loss='linear' if extraction=='ordinary' else 'soft_l1'
        for p in protocols:
            prefix=f"{p['cathode'][0]}_{p['rate_c']:g}C"
            def value(stage):
                f=SPECTRAL/'fits'/f'{prefix}_{stage}__source__{loss}__0.json'
                return read(f)['best']['parameters']['rs_plus_fast_r']
            p['R_ch']=value('Ch')
            for t in p['targets']:
                t['ratio']=value(t['stage'])/p['R_ch']
    return min(p['R_ch'] for p in protocols),protocols


def rhs_value(z,H,A,D,m,nu):
    q=max(float(z),ZMIN/4)
    return H*(1-A*q)/q**m-D/q**nu


def advance(z0,dt,H,A,D,m,nu,independent=False):
    if dt==0:
        return z0,False,0
    if z0<=ZMIN:
        return ZMIN,True,0
    if m==0 and not independent:
        z,stopped=legacy.advance(z0,dt,H*A,1/A,D,nu)
        return float(z),bool(stopped),0
    def rhs(t,y):
        return [rhs_value(y[0],H,A,D,m,nu)]
    def event(t,y):
        return y[0]-ZMIN
    event.terminal=True
    event.direction=-1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        s=solve_ivp(rhs,(0.,dt),[z0],method='Radau' if independent else 'LSODA',
            rtol=1e-10 if independent else 1e-8,atol=1e-12 if independent else 1e-10,
            events=event)
    if not s.success or not np.isfinite(s.y[0,-1]):
        raise RuntimeError('Integrator failed: '+s.message)
    return float(max(ZMIN,s.y[0,-1])),bool(len(s.t_events[0])),len(caught)


def parameters(x,m,nu,rstar):
    alpha,hp,hn=10**np.asarray(x[:3])
    rb,A=x[3]*rstar,x[4]
    return {'alpha':float(alpha),'H_P_per_hour':float(hp),'H_N_per_hour':float(hn),
        'background_ohm_cm2':float(rb),'A_star':float(A),'K_ohm_cm2':float(A*(rstar-rb)),
        'k_per_mah_cm2':float(alpha*A**(nu+1)),
        'h_P_per_hour':float(hp*A**(m+1)),'h_N_per_hour':float(hn*A**(m+1))}


def predictions(x,m,nu,timing,rstar,protocols,independent=False):
    alpha,hp,hn=10**np.asarray(x[:3])
    rb,A=x[3]*rstar,x[4]
    out=[]
    for p in protocols:
        if p['excluded_ambiguous']:continue
        h=hp if p['cathode']=='P-LCO' else hn
        z0=(rstar-rb)/(p['R_ch']-rb)
        z1,s1,w1=advance(z0,p['t1'],h,A,alpha*p['J'],m,nu,independent)
        zr,sr,wr=(ZMIN,True,0) if s1 else advance(z1,p['rest'],h,A,0.,m,nu,independent)
        z2,s2,w2=(ZMIN,True,0) if sr else advance(zr,p['t2'],h,A,alpha*p['J'],m,nu,independent)
        for target,z,stop in zip(p['targets'],[z1 if timing=='early' else zr,z2],[s1 if timing=='early' else sr,s2]):
            ratio=None if stop else float((rb+(rstar-rb)/z)/p['R_ch'])
            out.append({'cathode':p['cathode'],'rate_c':p['rate_c'],'stage':target['stage'],
                'subset':p['subset'],'observed_ratio':target['ratio'],'predicted_ratio':ratio,
                'error':None if ratio is None else ratio-target['ratio'],'disconnected':stop,
                'initial_a':float(A*z0),'effective_a':float(A*z),'integration_warnings':w1+wr+w2})
    return out


def residuals(rows):
    return np.array([100. if r['error'] is None else r['error'] for r in rows])


def classify(rows):
    out={}
    for label in ['train','check']:
        rr=[r for r in rows if r['subset']==label]
        complete=all(not r['disconnected'] for r in rr)
        e=residuals(rr)
        out[label]={'n':len(rr),'sse':float(np.sum(e*e)),
            'rmse':float(np.sqrt(np.mean(e*e))) if complete else None,
            'max_abs_error':float(max(abs(e))) if complete else None,
            'screen_pass':bool(complete and max(abs(e))<=.03)}
    return out


def task_id(t):
    return f"m{t['m']:g}_nu{t['nu']}_{t['timing']}_{t['extraction']}_{t['seed']}"


def fit(task):
    started=time.perf_counter()
    checkpoint_guard()
    rstar,protocols=data(task['extraction'])
    train=[p for p in protocols if p['subset']=='train']
    count,errors,disconnects=0,0,0
    failures={}
    def residual(x):
        nonlocal count,errors,disconnects
        count+=1
        if count%250==0:checkpoint_guard()
        try:
            rows=predictions(x,task['m'],task['nu'],task['timing'],rstar,train)
            disconnects+=any(r['disconnected'] for r in rows)
            return residuals(rows)
        except (RuntimeError,FloatingPointError,OverflowError,ValueError) as e:
            errors+=1
            failures[type(e).__name__+': '+str(e)[:180]]=failures.get(type(e).__name__+': '+str(e)[:180],0)+1
            return np.ones(8)*100.
    g=differential_evolution(lambda x:float(np.sum(residual(x)**2)),BOUNDS,seed=task['seed'],
        popsize=8,maxiter=100,tol=1e-7,polish=False,workers=1)
    l=least_squares(residual,g.x,bounds=np.array(BOUNDS).T,max_nfev=600,diff_step=1e-3,
        ftol=1e-10,xtol=1e-10,gtol=1e-10)
    x=min([g.x,l.x],key=lambda v:float(np.sum(residual(v)**2)))
    calculated=predictions(x,task['m'],task['nu'],task['timing'],rstar,protocols)
    direct=predictions(x,task['m'],task['nu'],task['timing'],rstar,protocols,True)
    agreement=[]
    for a,b in zip(calculated,direct):
        same=a['disconnected']==b['disconnected']
        diff=None if a['disconnected'] or b['disconnected'] else abs(a['predicted_ratio']-b['predicted_ratio'])
        agreement.append({'cathode':a['cathode'],'rate_c':a['rate_c'],'stage':a['stage'],
            'ratio_difference':diff,'status_matches':same,'pass':same and (diff is None or diff<=1e-4)})
    screens=classify(direct)
    out={'id':task_id(task),'task':task,'x':x.tolist(),'parameters':parameters(x,task['m'],task['nu'],rstar),
        'rstar':rstar,'points':direct,'optimizer_predictions':calculated,'subsets':screens,
        'numerical_checks':agreement,'numerical_checks_pass':all(r['pass'] for r in agreement),
        'both_screens_pass':all(r['screen_pass'] for r in screens.values()),
        'boundary_indices':[i for i,(v,(lo,hi)) in enumerate(zip(x,BOUNDS)) if min(v-lo,hi-v)<.001*(hi-lo)],
        'global_search':{'success':bool(g.success),'message':str(g.message),'nfev':int(g.nfev),'nit':int(g.nit)},
        'refinement':{'success':bool(l.success),'message':str(l.message),'nfev':int(l.nfev),'optimality':float(l.optimality)},
        'evaluations':count,'evaluation_numerical_failures':errors,'failure_reasons':failures,
        'evaluations_with_disconnection':int(disconnects),'wall_seconds':time.perf_counter()-started}
    checkpoint_guard()
    save(HERE/'fits'/(task_id(task)+'.json'),out)
    return {'id':out['id'],'seconds':out['wall_seconds'],'train':screens['train']['max_abs_error'],
        'check':screens['check']['max_abs_error'],'numerical_pass':out['numerical_checks_pass']}


def pilot():
    started=time.perf_counter()
    fixtures=[]
    # Explicit controls and finite extreme states in the declared normalized coordinates.
    for m in [0.,6.6]:
        for nu in [0,1]:
            for z,dt,H,A,D in [(.8,.5,0.,.5,.1),(.8,.5,1.,.5,0.),(.8,2.,.1,.5,.1),
                (.02,.1,1e5,.01,480.),(.5,2.,1e-5,1.,480.),(.9,.043,1e3,1.,0.)]:
                a,sa,wa=advance(z,dt,H,A,D,m,nu)
                b,sb,wb=advance(z,dt,H,A,D,m,nu,True)
                err=abs(a-b)/max(1.,abs(b))
                fixtures.append({'m':m,'nu':nu,'z':z,'dt':dt,'H':H,'A':A,'D':D,'computed':a,
                    'reference':b,'status_matches':sa==sb,'relative_state_error':err,
                    'pass':sa==sb and err<=1e-6,'warnings':wa+wb})
                checkpoint_guard()
    equilibria=[]
    for nu in [0,1]:
        for H,A,D in [(.1,.5,.5),(1e-5,.01,480.),(1e3,1.,.1)]:
            root=brentq(lambda z:H*(1-A*z)-D*z**(6.6-nu),1e-12,1/A,xtol=1e-13)
            value,stop,_=advance(root,2.,H,A,D,6.6,nu)
            err=abs(value-root)
            equilibria.append({'nu':nu,'H':H,'A':A,'D':D,'root':root,'error':err,'pass':not stop and err<=1e-7})
    reparameterization=[]
    rstar,protocols=data('published')
    for file in sorted((OLD/'fits').glob('*.json')):
        old=read(file)
        x=old['x'].copy()
        x[1]-=math.log10(x[4])
        x[2]-=math.log10(x[4])
        new=predictions(x,0.,old['nu'],old['timing'],rstar,protocols)
        error=max(abs(a['predicted_ratio']-b['predicted_ratio']) for a,b in zip(new,old['points']))
        reparameterization.append({'old_fit':file.name,'max_ratio_difference':error,'pass':error<=1e-6})
    x=np.array([-.7,-1.,-.2,.8,.7])
    complete=[]
    timings=[]
    for extraction in EXTRACTIONS:
        rstar,protocols=data(extraction)
        for nu in [0,1]:
            a=predictions(x,6.6,nu,'early',rstar,protocols)
            b=predictions(x,6.6,nu,'early',rstar,protocols,True)
            maximum=max(abs(i['predicted_ratio']-j['predicted_ratio']) for i,j in zip(a,b))
            complete.append({'extraction':extraction,'nu':nu,'max_ratio_difference':maximum,'pass':maximum<=1e-4})
        train=[p for p in protocols if p['subset']=='train']
        start=time.perf_counter()
        for j in range(20):
            probe=x.copy();probe[0]+=.005*j
            predictions(probe,6.6,1,'early',rstar,train)
        timings.append({'extraction':extraction,'seconds_per_training_evaluation':(time.perf_counter()-start)/20})
        checkpoint_guard()
    result={'fixtures':fixtures,'equilibria':equilibria,'reparameterization':reparameterization,
        'complete_protocol_checks':complete,'timings':timings,'wall_seconds':time.perf_counter()-started,
        'numerical_pass':all(r['pass'] for r in fixtures+equilibria+reparameterization+complete)}
    save(HERE/'pilot.json',result)
    print(json.dumps({'pass':result['numerical_pass'],'timings':timings,'seconds':result['wall_seconds']}),flush=True)
    assert result['numerical_pass']


def summarize():
    fits=[read(p) for p in sorted((HERE/'fits').glob('*.json'))]
    assert len(fits)==48
    selected=[]
    eligible=[]
    for m in [0.,6.6]:
        for nu in [0,1]:
            for timing in ['early','late']:
                family=[]
                for extraction in EXTRACTIONS:
                    group=[r for r in fits if r['task']['m']==m and r['task']['nu']==nu and r['task']['timing']==timing and r['task']['extraction']==extraction]
                    assert len(group)==2
                    best=min(group,key=lambda r:r['subsets']['train']['sse'])
                    row={'m':m,'nu':nu,'timing':timing,'extraction':extraction,'selected_id':best['id'],
                        'subsets':best['subsets'],'numerical_pass':best['numerical_checks_pass'],
                        'both_screens_pass':best['both_screens_pass'],'parameters':best['parameters'],
                        'boundary_indices':best['boundary_indices']}
                    selected.append(row);family.append(row)
                passed=all(r['both_screens_pass'] and r['numerical_pass'] for r in family)
                eligible.append({'m':m,'nu':nu,'timing':timing,'passes_all_extractions':passed})
    result={'selected':selected,'families':eligible,'eligible_family_count':sum(r['passes_all_extractions'] for r in eligible),
        'fit_tasks':48,'all_selected_numerical_pass':all(r['numerical_pass'] for r in selected),
        'selected_both_screen_pass_count':sum(r['both_screens_pass'] for r in selected),
        'prospective_prediction_generated':False,'claim_changed':False}
    save(HERE/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    checkpoint_guard()
    frozen=json.loads(json.dumps(manifest()))
    mp=HERE/'manifest.json'
    if mp.exists():assert read(mp)==frozen,'Exact manifest required'
    else:save(mp,frozen)
    if args.pilot:
        pilot();return
    assert read(HERE/'pilot.json')['numerical_pass']
    started=time.perf_counter()
    tasks=[{'m':m,'nu':nu,'timing':timing,'extraction':e,'seed':s}
        for m in [0.,6.6] for nu in [0,1] for timing in ['early','late'] for e in EXTRACTIONS for s in [1729,2718]]
    todo=[t for t in tasks if not (HERE/'fits'/(task_id(t)+'.json')).exists()]
    if len(todo)!=len(tasks):assert args.resume
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'))
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fit,t) for t in todo]
        for future in as_completed(futures):
            result=future.result();records.append(result)
            print(json.dumps(result),flush=True);checkpoint_guard()
    result=summarize()
    save(HERE/'batch.json',{'tasks':len(tasks),'new_tasks':len(todo),'workers':workers,
        'wall_seconds':time.perf_counter()-started,'completed':records,'manifest_sha256':sha(mp)})
    print(json.dumps({'eligible_families':result['eligible_family_count'],'selected_passes':result['selected_both_screen_pass_count'],
        'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
