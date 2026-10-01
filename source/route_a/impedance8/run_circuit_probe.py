"""Conditional full-spectrum circuit fits, with frozen inputs and all points."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import time
import numpy as np
from scipy.optimize import least_squares
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
OUT = HERE / 'fits'
LOW = np.array([-4., -4., .2, -4., -4., .2, -5., .2])
HIGH = np.array([3., 10., 1., 3., 10., 1., 3., 1.])
STARTS = [(6,1,.55), (7,30,.8), (8,1000,.95), (6,1000,.8), (7,1,.95), (8,30,.55)]
FSCALE = .02


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def model(x, f, rs=0., free=False, jac=False):
    z = np.zeros(f.size, complex)
    derivatives = []
    ln10 = np.log(10.)
    for offset in (0,3):
        r, fc, n = 10**x[offset], 10**x[offset+1], x[offset+2]
        logarithm = np.log(f/fc) + .5j*np.pi
        q = np.exp(n*logarithm)
        branch = r/(1+q)
        z += branch
        derivatives += [ln10*branch, r*n*ln10*q/(1+q)**2, -r*q*logarithm/(1+q)**2]
    logarithm = np.log(f) + .5j*np.pi
    tail = 10**x[6] * np.exp(-x[7]*logarithm)
    z += tail + (x[8] if free else rs)
    derivatives += [ln10*tail, -tail*logarithm]
    if free:
        derivatives += [np.ones(f.size, complex)]
    return (z, np.asarray(derivatives).T) if jac else z


def residual(x, f, zobs, weights, rs, free):
    e = (model(x, f, rs, free)-zobs)/weights
    return np.r_[e.real, e.imag]


def jacobian(x, f, zobs, weights, rs, free):
    _, j = model(x, f, rs, free, True)
    j = j/weights[:,None]
    return np.r_[j.real, j.imag]


def parameters(x, rs, free):
    branches = [{'r_ohm_cm2': float(10**x[o]), 'fc_hz': float(10**x[o+1]), 'n': float(x[o+2]),
        'q_angular_convention': float(1/(10**x[o]*(2*np.pi*10**x[o+1])**x[o+2]))} for o in (0,3)]
    branches.sort(key=lambda b: b['fc_hz'], reverse=True)
    rseries = float(x[8] if free else rs)
    return {'fast':branches[0], 'slow':branches[1], 'rs_ohm_cm2':rseries,
        'rs_plus_fast_r':rseries+branches[0]['r_ohm_cm2'],
        'tail_d_ohm_cm2_at_1_hz':float(10**x[6]), 'tail_n':float(x[7]),
        'branch_log10_separation':float(np.log10(branches[0]['fc_hz']/branches[1]['fc_hz']))}


def task_id(task):
    return f"{task['spectrum_id']}__{task['topology']}__{task['loss']}__{task['fraction']:g}"


def fit_one(task, spectrum, f):
    tic = time.perf_counter()
    free = task['topology'] == 'series'
    rs = task['fraction'] * min(spectrum['re_ohm_cm2']) if task['topology'] == 'fixed' else 0.
    zobs = np.asarray(spectrum['re_ohm_cm2']) - 1j*np.asarray(spectrum['minus_im_ohm_cm2'])
    weights = np.abs(zobs)
    upper_rs = .99*min(spectrum['re_ohm_cm2'])
    lo = np.r_[LOW, 0.] if free else LOW.copy()
    hi = np.r_[HIGH, upper_rs] if free else HIGH.copy()
    starts = []
    for i, (logfast, mid, n) in enumerate(STARTS):
        checkpoint_guard()
        start_rs = upper_rs*[.05,.5,.9,.5,.9,.05][i] if free else rs
        rfast = max(.01, spectrum['source_bulk_li_r']-start_rs)
        rmid = max(.01, spectrum['source_cathode_r'])
        d = max(.02, abs(zobs[-1].imag)*np.sqrt(f[-1])/np.sin(np.pi/4))
        x0 = np.array([np.log10(rfast),logfast,n,np.log10(rmid),np.log10(mid),n,np.log10(d),.5])
        if free:
            x0 = np.r_[x0,start_rs]
        x0 = np.minimum(hi-1e-8,np.maximum(lo+1e-8,x0))
        args = (f,zobs,weights,rs,free)
        ans = least_squares(residual, x0, jac=jacobian, args=args, bounds=(lo,hi),
            loss=task['loss'], f_scale=FSCALE, max_nfev=2500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        starts.append({'start_index':i, 'x':ans.x.tolist(), 'objective':float(ans.cost),
            'nfev':int(ans.nfev), 'status':int(ans.status), 'success':bool(ans.success),
            'termination':ans.message, 'optimality':float(ans.optimality),
            'bound_indices':np.flatnonzero(np.minimum(ans.x-lo,hi-ans.x)/(hi-lo)<1e-5).tolist(),
            'parameters':parameters(ans.x,rs,free)})
    best = min(starts, key=lambda r:r['objective'])
    x = np.array(best['x'])
    prediction = model(x,f,rs,free)
    e = (prediction-zobs)/weights
    j = jacobian(x, f, zobs, weights, rs, free)
    singular = np.linalg.svd(j,compute_uv=False)
    k = int(np.argmax(np.abs(e)))
    result = {'task':task, 'id':task_id(task), 'starts':starts, 'best_start':best['start_index'],
        'best':best, 'complex_relative_rms':float(np.sqrt(np.mean(np.abs(e)**2))),
        'component_relative_rms':float(np.sqrt(np.mean(residual(x,f,zobs,weights,rs,free)**2))),
        'complex_relative_max':float(np.max(np.abs(e))),
        'worst_point_index':k, 'worst_workbook_row':k+3, 'worst_frequency_hz':float(f[k]),
        'max_complex_absolute_ohm_cm2':float(np.max(np.abs(prediction-zobs))),
        'source_bulk_li_r':spectrum['source_bulk_li_r'], 'source_cathode_r':spectrum['source_cathode_r'],
        'source_bulk_li_fractional_difference':best['parameters']['rs_plus_fast_r']/spectrum['source_bulk_li_r']-1,
        'source_cathode_fractional_difference':best['parameters']['slow']['r_ohm_cm2']/spectrum['source_cathode_r']-1,
        'jacobian_singular_values':singular.tolist(), 'jacobian_smallest_over_largest':float(singular[-1]/singular[0]),
        'predicted_re_ohm_cm2':prediction.real.tolist(), 'predicted_minus_im_ohm_cm2':(-prediction.imag).tolist(),
        'wall_seconds':time.perf_counter()-tic}
    checkpoint_guard()
    save(OUT/(task_id(task)+'.json'),result)
    return {'id':result['id'], 'seconds':result['wall_seconds'], 'rms':result['complex_relative_rms']}


def validation(f):
    checkpoint_guard()
    x = np.array([np.log10(17.),np.log10(5e5),.8,np.log10(14.),np.log10(40.),.7,np.log10(2.),.5,2.])
    zz, jj = model(x,f,free=True,jac=True)
    direct = np.zeros(f.size,complex)+x[8]
    for o in (0,3):
        r,fc,n = 10**x[o],10**x[o+1],x[o+2]
        q = 1/(r*(2*np.pi*fc)**n)
        direct += 1/(1/r+q*(2j*np.pi*f)**n)
    qd = 1/(10**x[6]*(2*np.pi)**x[7])
    direct += 1/(qd*(2j*np.pi*f)**x[7])
    algebra_error = float(np.max(np.abs(zz-direct)))
    finite = []
    for k in range(len(x)):
        h = 1e-6
        xp,xm = x.copy(),x.copy()
        xp[k] += h
        xm[k] -= h
        fd = (model(xp,f,free=True)-model(xm,f,free=True))/(2*h)
        finite.append(float(np.max(np.abs(fd-jj[:,k]))/max(1.,np.max(np.abs(jj[:,k])))))
    synthetic = []
    for free in (False,True):
        truth = x if free else x[:8]
        z = model(truth,f,free=free)
        x0 = truth.copy()+np.array([.05,.1,-.04,-.05,-.15,.05,.1,-.02]+([.2] if free else []))
        lo,hi = (np.r_[LOW,0.],np.r_[HIGH,15.]) if free else (LOW,HIGH)
        args = (f,z,np.abs(z),0.,free)
        a = least_squares(residual,x0,jac=jacobian,args=args,bounds=(lo,hi),
            max_nfev=2500,ftol=1e-12,xtol=1e-12,gtol=1e-12)
        err = float(np.max(np.abs(model(a.x,f,free=free)-z)/np.abs(z)))
        synthetic.append({'free_series_r':free, 'max_relative_recovery_error':err, 'success':bool(a.success)})
    assert algebra_error<1e-10 and max(finite)<1e-7
    assert all(s['max_relative_recovery_error']<1e-8 for s in synthetic)
    result = {'direct_admittance_max_absolute_error':algebra_error, 'jacobian_fd_relative_errors':finite,
        'synthetic':synthetic, 'numerical_checks_pass':True,
        'meaning':'Checks implementation, not real-data validity or component identifiability.'}
    save(HERE/'numerical_checks.json',result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['pilot','all','profiles'],default='pilot')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    checkpoint_guard()
    manifest = {'script_sha256':sha(Path(__file__)), 'plan_sha256':sha(HERE/'PLAN.md'),
        'spectra_sha256':sha(HERE/'spectra.json'), 'starts':STARTS, 'fscale':FSCALE,
        'max_nfev':2500, 'ftol_xtol_gtol':1e-10, 'numpy':np.__version__}
    manifest = json.loads(json.dumps(manifest))
    mp = HERE/'fit_manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text())==manifest, 'Exact-source resume required'
    else:
        save(mp,manifest)
    data = json.loads((HERE/'spectra.json').read_text())
    f = 10**np.array(data['frequency_log10_hz'])
    spectra = {s['id']:s for s in data['spectra']}
    if args.mode=='pilot':
        validation(f)
        ids = ['N_0.2C_Ch','P_0.2C_Ch']
    elif args.mode=='all':
        ids = list(spectra)
    else:
        ids = ['N_0.2C_Ch','N_0.2C_dCh2','P_0.2C_Ch','P_0.2C_dCh2']
    if args.mode=='profiles':
        tasks = [{'spectrum_id':s,'topology':'fixed','loss':loss,'fraction':fraction}
            for s in ids for loss in ['linear','soft_l1'] for fraction in [.25,.5,.75,.95]]
    else:
        tasks = [{'spectrum_id':s,'topology':topology,'loss':loss,'fraction':0.}
            for s in ids for topology in ['source','series'] for loss in ['linear','soft_l1']]
    pending = [t for t in tasks if not (OUT/(task_id(t)+'.json')).exists()]
    if len(pending)!=len(tasks):
        assert args.resume, 'Use exact-source --resume or a fresh output copy'
    workers = int(os.environ.get('AJ_COMPUTE_WORKERS','1'))
    completed = []
    if workers==1:
        for task in pending:
            out = fit_one(task,spectra[task['spectrum_id']],f)
            completed.append(out)
            print(json.dumps(out),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fit_one,t,spectra[t['spectrum_id']],f) for t in pending]
            for future in as_completed(futures):
                out = future.result()
                completed.append(out)
                print(json.dumps(out),flush=True)
                checkpoint_guard()
    save(HERE/(args.mode+'_batch.json'), {'mode':args.mode,'requested_tasks':len(tasks),
        'previously_saved_tasks':len(tasks)-len(pending),'newly_completed_tasks':len(completed),
        'workers':workers,'wall_seconds':time.perf_counter()-started,
        'task_ids':[task_id(t) for t in tasks],'new_results':completed,'manifest_sha256':sha(mp)})
    print(json.dumps({'mode':args.mode,'completed':len(completed),'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':
    main()
