"""Synthetic sequential EIS: fixed kinetics and explicitly adiabatic observation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import importlib.util
import json
import os
import time

import numpy as np
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec); spec.loader.exec_module(out)
    return out


rest = module('sweep_frozen_rest', HERE.parent/'readout34/run_readout.py')
circuit = module('sweep_frozen_circuit', HERE.parent/'impedance8/run_circuit_probe.py')
circuit.OUT = HERE/'fits'
# The unchanged source fitter's stationary boundary solutions stopped above
# our newly frozen reconstruction tolerance. Retain its objectives, starts,
# bounds and maximum evaluations; change only these convergence tolerances.
# The failed original controls and complete inputs are archived in v1.
original_least_squares = circuit.least_squares


def precise_least_squares(*args, **kwargs):
    kwargs.update(ftol=1e-13, xtol=1e-13, gtol=1e-13)
    return original_least_squares(*args, **kwargs)


circuit.least_squares = precise_least_squares
read, save, sha = rest.read, rest.save, rest.sha
CATHODES = ['P-LCO', 'N-LCO']
LAWS = ['fixed_q', 'fixed_fc']
ORDERS = ['high_to_low', 'low_to_high']


def identifier(t):
    return '_'.join([t['witness_id'], t['law'], t['order']])


def loss_for(extraction):
    return 'soft_l1' if extraction == 'robust' else 'linear'


def template_path(extraction, cathode):
    return HERE.parent/'impedance8/fits'/f'{cathode[0]}_2C_Ch__source__{loss_for(extraction)}__0.json'


def frequency():
    return 10**np.array(read(HERE.parent/'impedance8/spectra.json')['frequency_log10_hz'])


def schedule(f, order):
    indices = list(range(len(f))) if order == 'high_to_low' else list(reversed(range(len(f))))
    rows = [None]*len(f); clock = 0.
    for rank, k in enumerate(indices):
        start = clock + .5; end = start + 3/f[k]
        rows[k] = dict(frequency_index=k, acquisition_rank=rank, frequency_hz=float(f[k]),
                       settle_start_s=clock, measure_start_s=start, measure_end_s=end)
        clock = end
    return rows, clock


def freeze():
    old = read(HERE.parent/'readout34/manifest.json')
    tasks = [dict(witness_id=w['id'], law=law, order=order)
             for w in old['witnesses'] for law in LAWS for order in ORDERS]
    paths = [Path(__file__).resolve(), HERE/'PLAN.md', HERE/'PRECISION_REPAIR.md', HERE.parent/'readout34/manifest.json',
             HERE.parent/'readout34/summary.json', HERE.parent/'impedance8/run_circuit_probe.py']
    paths += [HERE.parent/'readout34/cases'/f"{w['id']}.json" for w in old['witnesses']]
    paths += [template_path(e,c) for e in ['ordinary','published','robust'] for c in CATHODES]
    hashes = old['sha256'].copy()
    for p in paths: hashes[str(p.relative_to(ROOT))] = sha(p)
    for path, digest in hashes.items(): assert sha(ROOT/path) == digest, path
    out = dict(sha256=hashes, witnesses=old['witnesses'], tasks=tasks,
               quadrature_orders=[8,16], cycles_per_point=3, settling_seconds=.5,
               source_frequency_count=68, optimizer_ftol_xtol_gtol=1e-13,
               observation='adiabatic average of instantaneous complex impedance',
               kinetic_parameters_refitted=False, central_claim_changed=False,
               measured_noise_model=False, certified_parameter_bounds=False)
    p = HERE/'manifest.json'
    if p.exists(): assert read(p) == out
    else: save(p,out)
    return out


def make_template(extraction, cathode, Rref):
    p = read(template_path(extraction,cathode))['best']['parameters']
    b, s = p['fast'], p['slow']
    return dict(Rref=Rref, fc=b['fc_hz'], n=b['n'], slow_r=s['r_ohm_cm2'],
                slow_fc=s['fc_hz'], slow_n=s['n'], D=p['tail_d_ohm_cm2_at_1_hz'],
                tail_n=p['tail_n'], source=str(template_path(extraction,cathode).relative_to(ROOT)))


def forward(f, R, template, law, direct=False):
    t = template
    fc = t['fc']*(t['Rref']/R)**(1/t['n']) if law == 'fixed_q' else t['fc']
    if direct:
        Q = 1/(R*(2*np.pi*fc)**t['n'])
        slow_Q = 1/(t['slow_r']*(2*np.pi*t['slow_fc'])**t['slow_n'])
        tail_Q = 1/(t['D']*(2*np.pi)**t['tail_n'])
        return 1/(1/R+Q*(2j*np.pi*f)**t['n']) + 1/(1/t['slow_r']+slow_Q*(2j*np.pi*f)**t['slow_n']) + 1/(tail_Q*(2j*np.pi*f)**t['tail_n'])
    return R/(1+(1j*f/fc)**t['n']) + t['slow_r']/(1+(1j*f/t['slow_fc'])**t['slow_n']) + t['D']*(1j*f)**(-t['tail_n'])


def fit(sid, z, f, template, loss):
    task = dict(spectrum_id=sid, topology='source', loss=loss, fraction=0.)
    p = circuit.OUT/(circuit.task_id(task)+'.json')
    spectrum = dict(re_ohm_cm2=z.real.tolist(), minus_im_ohm_cm2=(-z.imag).tolist(),
                    source_bulk_li_r=template['Rref'], source_cathode_r=template['slow_r'])
    if not p.exists(): circuit.fit_one(task,spectrum,f)
    ans = read(p)
    return ans, str(p.relative_to(HERE))


def controls(manifest):
    p = HERE/'controls.json'
    if p.exists():
        out = read(p); assert out['manifest_sha256'] == sha(HERE/'manifest.json') and out['passed']; return out
    f = frequency(); records = []
    for extraction in ['ordinary','published','robust']:
        w = next(w for w in manifest['witnesses'] if w['setting']['extraction'] == extraction)
        for c in CATHODES:
            h = next(h for h in w['diagonal_histories'] if h['cathode'] == c)
            t = make_template(extraction,c,h['R_ch']); R = t['Rref']
            z = forward(f,np.full(len(f),R),t,'fixed_q')
            x = np.array([np.log10(R),np.log10(t['fc']),t['n'],np.log10(t['slow_r']),
                          np.log10(t['slow_fc']),t['slow_n'],np.log10(t['D']),t['tail_n']])
            errors = []
            for law in LAWS:
                for order in ORDERS:
                    sc,_ = schedule(f,order)
                    # Gamma=0 passes through the rest solver and point averaging.
                    a0=.6; C=.4*R*a0; rb=.6*R
                    values = []
                    nodes, weights = np.polynomial.legendre.leggauss(8)
                    for k,row in enumerate(sc):
                        times = (row['measure_start_s']+row['measure_end_s'])/2 + nodes*(row['measure_end_s']-row['measure_start_s'])/2
                        rr = np.array([rb+C/rest.implicit_contact(a0,0.,float(v)) for v in times])
                        values.append(np.dot(weights,forward(f[k],rr,t,law))/2)
                    errors.append(float(np.max(np.abs(np.array(values)-circuit.model(x,f))/np.abs(z))))
                    errors.append(float(np.max(np.abs(forward(f,np.full(len(f),R),t,law,True)-z)/np.abs(z))))
            ans,path = fit('reference_'+extraction+'_'+c[0],z,f,t,loss_for(extraction))
            Rfit = ans['best']['parameters']['fast']['r_ohm_cm2']
            relative = abs(Rfit/R-1.)
            passed = bool(max(errors)<=1e-12 and ans['complex_relative_max']<=1e-7 and relative<=1e-6 and ans['best']['success'])
            records.append(dict(extraction=extraction,cathode=c,template=t,fit_path=path,
                                fitted_reference=Rfit,normalized_R_recovery_error=relative,
                                zero_recovery_algebra_errors=errors,complex_relative_max=ans['complex_relative_max'],passed=passed))
            checkpoint_guard()
    out = dict(records=records,passed=all(r['passed'] for r in records),manifest_sha256=sha(HERE/'manifest.json'))
    save(p,out); assert out['passed'], 'Stationary implementation control failed'
    return out


def generate(w,h,t,law,sc,f,nodes_count,direct=False):
    b,A=w['x'][3:]; rb=b*w['rstar']; C=A*(1-b)*w['rstar']
    a0=h['rows'][0]['independent']['contact'][0]; gamma=h['gammas_per_hour'][1]
    nodes,weights=np.polynomial.legendre.leggauss(nodes_count)
    zz=[]; point_data=[]
    for k,row in enumerate(sc):
        u,v=row['measure_start_s'],row['measure_end_s']; times=(u+v)/2+nodes*(v-u)/2
        aa=np.array([rest.implicit_contact(a0,gamma,float(seconds)) for seconds in times])
        rr=rb+C/aa
        zz.append(np.dot(weights,forward(f[k],rr,t,law,direct))/2)
        endpoints=np.array([rest.implicit_contact(a0,gamma,float(seconds)) for seconds in [u,v]])
        r_end=rb+C/endpoints
        # Maximum fractional change per cycle is checked at interval endpoints.
        dr=C/endpoints**2 * gamma*(1-endpoints)*endpoints**(-rest.model.M)/3600
        point_data.append(dict(**row,contact_endpoints=endpoints.tolist(),resistance_endpoints=r_end.tolist(),
                               mean_instantaneous_resistance=float(np.dot(weights,rr)/2),
                               fractional_R_change_per_cycle_at_endpoints=(dr/r_end/f[k]).tolist()))
    return np.array(zz),point_data


def one(task):
    tic=time.perf_counter(); manifest=read(HERE/'manifest.json'); key=identifier(task)
    w=next(w for w in manifest['witnesses'] if w['id']==task['witness_id']); extraction=w['setting']['extraction']
    src=HERE.parent/'readout34/cases'/f"{w['id']}.json"; prior=read(src)
    ctrl=read(HERE/'controls.json');f=frequency();sc,duration=schedule(f,task['order']);rows=[]
    for c in CATHODES:
        checkpoint_guard();h=next(h for h in prior['histories'] if h['cathode']==c)
        ref=next(r for r in ctrl['records'] if r['extraction']==extraction and r['cathode']==c)
        t=make_template(extraction,c,h['R_ch']); assert t==ref['template']
        z,points=generate(w,h,t,task['law'],sc,f,8)
        zcheck,_=generate(w,h,t,task['law'],sc,f,16,True)
        error=float(np.max(np.abs(z-zcheck)/np.abs(zcheck)))
        assert error<=1e-8, 'Point quadrature/admittance check failed'
        ans,path=fit(key+'_'+c[0],z,f,t,loss_for(extraction))
        Rfit=ans['best']['parameters']['fast']['r_ohm_cm2']; normalized=Rfit/ref['fitted_reference']
        a0=h['rows'][0]['independent']['contact'][0]; aend=rest.implicit_contact(a0,h['gammas_per_hour'][1],duration)
        b,A=w['x'][3:];Rend=(b*w['rstar']+A*(1-b)*w['rstar']/aend)/h['R_ch']
        Rzero=h['rows'][0]['independent']['ratio'];slack=1e-6
        data=dict(cathode=c,template=t,reference_fit_path=ref['fit_path'],fit_path=path,
                  re_ohm_cm2=z.real.tolist(),minus_im_ohm_cm2=(-z.imag).tolist(),
                  direct16_re_ohm_cm2=zcheck.real.tolist(),direct16_minus_im_ohm_cm2=(-zcheck.imag).tolist(),
                  points=points,quadrature_direct_error=error,fitted_ratio=normalized,
                  fitted_ratio_true_reference=Rfit/h['R_ch'],true_ratio_t0=Rzero,true_ratio_tend=Rend,
                  fit_inside_instantaneous_extrema=bool(Rend-slack<=normalized<=Rzero+slack),
                  normalized_outside_extrema=max(0.,Rend-normalized,normalized-Rzero),
                  selected_fit_success=ans['best']['success'],branch_log10_separation=ans['best']['parameters']['branch_log10_separation'],
                  selected_bounds=ans['best']['bound_indices'],complex_relative_rms=ans['complex_relative_rms'],
                  max_fractional_R_change_per_cycle=max(max(p['fractional_R_change_per_cycle_at_endpoints']) for p in points))
        rows.append(data)
    P,N=rows;contrast=P['fitted_ratio']-N['fitted_ratio'];zero=P['true_ratio_t0']-N['true_ratio_t0']
    interval=[P['true_ratio_tend']-N['true_ratio_t0'],P['true_ratio_t0']-N['true_ratio_tend']]
    out=dict(task=task,setting=w['setting'],duration_s=duration,histories=rows,
             fitted_contrast=contrast,true_contrast_t0=zero,true_monotone_window_interval=interval,
             positive_contrast=bool(contrast>0),bias_from_t0=contrast-zero,
             instantaneous_budget_pass=bool(abs(contrast-zero)<=.001),
             pair_inside_true_window=bool(interval[0]-2e-6<=contrast<=interval[1]+2e-6),
             numerical_pass=all(r['selected_fit_success'] and r['quadrature_direct_error']<=1e-8 for r in rows),
             source=str(src.relative_to(ROOT)),manifest_sha256=sha(HERE/'manifest.json'),wall_seconds=time.perf_counter()-tic)
    save(HERE/'cases'/f'{key}.json',out)
    return dict(id=key,seconds=out['wall_seconds'],numerical_pass=out['numerical_pass'],
                contrast=contrast,bias=contrast-zero,contained=all(r['fit_inside_instantaneous_extrema'] for r in rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','all'],required=True);args=parser.parse_args()
    tic=time.perf_counter();checkpoint_guard();m=freeze();controls(m)
    tasks=[t for t in m['tasks'] if args.phase=='all' or t['witness_id'] in ['w026','w038']]
    pending=[];reused={}
    for t in tasks:
        p=HERE/'cases'/f'{identifier(t)}.json'
        if p.exists():
            r=read(p);assert r['manifest_sha256']==sha(HERE/'manifest.json') and r['task']==t
            reused[str(p.relative_to(HERE))]=sha(p)
        else:pending.append(t)
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,t) for t in pending]
        for future in as_completed(futures):
            r=future.result();records.append(r);print(json.dumps(r),flush=True);checkpoint_guard()
    out=dict(phase=args.phase,requested=len(tasks),reused_sha256=reused,new_results=records,
             workers=workers,wall_seconds=time.perf_counter()-tic,manifest_sha256=sha(HERE/'manifest.json'))
    save(HERE/(args.phase+'_execution.json'),out)
    print(json.dumps(dict(phase=args.phase,completed=len(records),seconds=out['wall_seconds'])),flush=True)


if __name__=='__main__':main()
