"""Frozen readout of existing 2x2 trajectories; no new kinetic fits or ODEs."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import os
import time

import numpy as np
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
REV = HERE.parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location('retained_sweep35', REV/'sweep35/run_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)
sweep.circuit.OUT = HERE/'fits'
read, save, sha = sweep.read, sweep.save, sweep.sha
CATS = ['P-LCO', 'N-LCO']
PILOT = ['w026', 'w038']


def freeze():
    old = read(REV/'sweep35/manifest.json')
    assert read(REV/'sweep35/summary.json')['verification_passed']
    assert read(REV/'weak33/summary.json')['verification_passed']
    assert read(REV/'decomposition29/summary.json')['passed']
    assert len(old['witnesses']) == 50
    hashes = dict(old['sha256'])
    paths = [Path(__file__), HERE/'PLAN.md', REV/'sweep35/manifest.json',
             REV/'sweep35/controls.json', REV/'sweep35/summary.json',
             REV/'weak33/manifest.json', REV/'weak33/summary.json',
             REV/'decomposition29/summary.json']
    for w in old['witnesses']:
        paths += [REV/'weak33/cases'/('linear_'+w['id']+'.json'),
                  REV/'sweep35/cases'/(w['id']+'_fixed_q_high_to_low.json')]
    for ref in read(REV/'sweep35/controls.json')['records']:
        paths.append(REV/'sweep35'/ref['fit_path'])
    for p in paths:
        hashes[str(p.relative_to(ROOT))] = sha(p)
    for name, digest in hashes.items():
        assert sha(ROOT/name) == digest, name
    out = dict(sha256=hashes, witnesses=old['witnesses'], pilot=PILOT,
               reused_discharge_histories=200, new_crossed_spectra=100,
               repeated_diagonal_adapter_controls=4, law='fixed_q', order='high_to_low',
               kinetic_refit=False, discharge_integrations=0,
               history_includes_q0=True, initial_includes_spectral_template=True,
               previously_inspected_instantaneous_results=True)
    p = HERE/'manifest.json'
    if p.exists():
        assert read(p) == out, 'Exact-source manifest changed'
    else:
        save(p, out)
    return out


def loaded_cells(w):
    p = REV/'weak33/cases'/('linear_'+w['id']+'.json')
    old = read(p)
    assert old['available'] and old['witness'] == w
    assert old['manifest_sha256'] == sha(REV/'weak33/manifest.json')
    cells = old['result']['histories']
    assert len(cells) == 4
    for h in cells:
        assert h['available'] and all(r['physical_pass'] and r['numerical_pass'] for r in h['rows'])
        ini = next(x for x in w['diagonal_histories'] if x['cathode'] == h['initial_cathode'])
        hist = next(x for x in w['diagonal_histories'] if x['cathode'] == h['history_cathode'])
        assert h['R_ch'] == ini['R_ch'] and h['q0'] == hist['q0']
        assert h['rows'][0]['stage'] == 'end_discharge'
        assert not h['rows'][0]['independent']['stopped']
    return cells


def adapt(w, cell):
    x = np.array(w['x']); H = 10**x[1]; A = x[4]
    pressures = sweep.rest.pressures(w, cell['history_cathode'], cell['q0']+.3)
    gammas = [H*A**(sweep.rest.model.M+1)*(p/2)**sweep.rest.model.M for p in pressures]
    src = cell['rows'][0]['independent']
    return dict(R_ch=cell['R_ch'], q0=cell['q0'], gammas_per_hour=gammas,
                rows=[dict(independent=dict(contact=[src['a']], ratio=src['ratio']))])


def new_readout(w, cell, label):
    extraction = w['setting']['extraction']; initial = cell['initial_cathode']
    f = sweep.frequency(); schedule, duration = sweep.schedule(f, 'high_to_low')
    h = adapt(w, cell)
    ref = next(r for r in read(REV/'sweep35/controls.json')['records']
               if r['extraction'] == extraction and r['cathode'] == initial)
    template = sweep.make_template(extraction, initial, cell['R_ch'])
    assert template == ref['template'] and ref['passed']
    z, points = sweep.generate(w, h, template, 'fixed_q', schedule, f, 8)
    zd, _ = sweep.generate(w, h, template, 'fixed_q', schedule, f, 16, True)
    quad_error = float(np.max(np.abs(z-zd)/np.abs(zd)))
    task = dict(spectrum_id=w['id']+'_'+label, topology='source',
                loss=sweep.loss_for(extraction), fraction=0.)
    fit_path = HERE/'fits'/(sweep.circuit.task_id(task)+'.json')
    spectrum = dict(re_ohm_cm2=z.real.tolist(), minus_im_ohm_cm2=(-z.imag).tolist(),
                    source_bulk_li_r=template['Rref'], source_cathode_r=template['slow_r'])
    if not fit_path.exists():
        sweep.circuit.fit_one(task, spectrum, f)
    fit = read(fit_path)
    rfit = fit['best']['parameters']['fast']['r_ohm_cm2']
    ratio = float(rfit/ref['fitted_reference'])
    return dict(initial_cathode=initial, history_cathode=cell['history_cathode'],
                R_ch=cell['R_ch'], q0=cell['q0'], template=template,
                gammas_per_hour=h['gammas_per_hour'], duration_seconds=duration,
                fitted_reference=ref['fitted_reference'], fitted_ratio=ratio,
                true_ratio_t0=cell['rows'][0]['independent']['ratio'],
                fit_path=str(fit_path.relative_to(HERE)), fit_sha256=sha(fit_path),
                re_ohm_cm2=z.real.tolist(), minus_im_ohm_cm2=(-z.imag).tolist(),
                direct16_re_ohm_cm2=zd.real.tolist(), direct16_minus_im_ohm_cm2=(-zd.imag).tolist(),
                points=points, quadrature_direct_error=quad_error,
                selected_fit_success=fit['best']['success'],
                numerical_pass=bool(quad_error<=1e-8 and fit['best']['success']),
                reused=False)


def split(y):
    pp, pn, np_, nn = [y[key] for key in ['PP','PN','NP','NN']]
    d = pp-nn; hn = np_-nn; sn = pn-nn; k = pp-pn-np_+nn
    hp = pp-pn; sp = pp-np_; hs = (hn+hp)/2; ss = (sn+sp)/2
    out = dict(total=d, history_at_N_initial=hn, initial_at_N_history=sn,
               interaction=k, history_at_P_initial=hp, initial_at_P_history=sp,
               history_symmetric=hs, initial_symmetric=ss)
    return dict(components_pp={name:100*v for name,v in out.items()},
                baseline_closure_pp=100*(d-hn-sn-k), symmetric_closure_pp=100*(d-hs-ss),
                symmetric_history_fraction=None if d==0 else hs/d)


def one(wid):
    checkpoint_guard(); tic = time.perf_counter(); m=read(HERE/'manifest.json')
    path=HERE/'cases'/(wid+'.json')
    if path.exists():
        out=read(path); assert out['manifest_sha256']==sha(HERE/'manifest.json'); return out
    w=next(w for w in m['witnesses'] if w['id']==wid)
    source=read(REV/'sweep35/cases'/(wid+'_fixed_q_high_to_low.json'))
    assert source['manifest_sha256']==sha(REV/'sweep35/manifest.json') and source['numerical_pass']
    cells={}; controls=[]
    for cell in loaded_cells(w):
        ini,hist=cell['initial_cathode'],cell['history_cathode'];key=ini[0]+hist[0]
        if ini==hist:
            old=next(h for h in source['histories'] if h['cathode']==ini)
            assert abs(old['true_ratio_t0']-cell['rows'][0]['independent']['ratio'])<=1e-12
            row=dict(initial_cathode=ini,history_cathode=hist,R_ch=cell['R_ch'],q0=cell['q0'],
                     fitted_ratio=old['fitted_ratio'],true_ratio_t0=old['true_ratio_t0'],
                     reused=True,source=str((REV/'sweep35/cases'/(wid+'_fixed_q_high_to_low.json')).relative_to(ROOT)),
                     numerical_pass=bool(old['selected_fit_success'] and old['quadrature_direct_error']<=1e-8))
            if wid in PILOT:
                probe=new_readout(w,cell,'adapter_'+key)
                oldz=np.array(old['re_ohm_cm2'])-1j*np.array(old['minus_im_ohm_cm2'])
                newz=np.array(probe['re_ohm_cm2'])-1j*np.array(probe['minus_im_ohm_cm2'])
                err=float(np.max(np.abs(newz-oldz)/np.abs(oldz)))
                diff=abs(probe['fitted_ratio']-old['fitted_ratio'])
                controls.append(dict(key=key,spectral_relative_error=err,ratio_difference=diff,
                                     passed=bool(probe['numerical_pass'] and err<=1e-10 and diff<=1e-6),readout=probe))
        else:
            row=new_readout(w,cell,key)
        cells[key]=row; checkpoint_guard()
    decompositions={field:split({key:r[field] for key,r in cells.items()}) for field in ['fitted_ratio','true_ratio_t0']}
    closure=max(abs(d[k]) for d in decompositions.values() for k in ['baseline_closure_pp','symmetric_closure_pp'])
    total_error=abs(decompositions['fitted_ratio']['components_pp']['total']/100-source['fitted_contrast'])
    passed=all(r['numerical_pass'] for r in cells.values()) and all(c['passed'] for c in controls) and closure<=1e-10 and total_error<=1e-6
    out=dict(witness_id=wid,setting=w['setting'],x=w['x'],rstar=w['rstar'],cells=cells,
             decompositions=decompositions,adapter_controls=controls,closure_pp=closure,
             diagonal_total_difference=total_error,numerical_pass=bool(passed),
             wall_seconds=time.perf_counter()-tic,manifest_sha256=sha(HERE/'manifest.json'))
    save(path,out);return out


def statistics(rows):
    metrics=list(rows[0]['decompositions']['fitted_ratio']['components_pp'])
    out={}
    for quantity in ['fitted_ratio','true_ratio_t0']:
        out[quantity]={}
        for metric in metrics:
            values=np.array([r['decompositions'][quantity]['components_pp'][metric] for r in rows])
            out[quantity][metric]=dict(count=len(values),minimum=float(min(values)),median=float(np.median(values)),
                                      maximum=float(max(values)),positive=int(sum(values>0)),negative=int(sum(values<0)),zero=int(sum(values==0)))
        values=np.array([r['decompositions'][quantity]['symmetric_history_fraction'] for r in rows])
        out[quantity]['symmetric_history_fraction']=dict(minimum=float(min(values)),median=float(np.median(values)),maximum=float(max(values)))
    return out


def summarize(m):
    rows=[read(HERE/'cases'/(w['id']+'.json')) for w in m['witnesses']]
    assert all(r['manifest_sha256']==sha(HERE/'manifest.json') for r in rows)
    groups={e:statistics([r for r in rows if r['setting']['extraction']==e]) for e in ['ordinary','published','robust']}
    out=dict(witnesses=len(rows),all_numerical_pass=all(r['numerical_pass'] for r in rows),
             failed_ids=[r['witness_id'] for r in rows if not r['numerical_pass']],
             all_witnesses=statistics(rows),by_extraction=groups,
             maximum_closure_pp=max(r['closure_pp'] for r in rows),
             maximum_diagonal_total_difference=max(r['diagonal_total_difference'] for r in rows),
             manifest_sha256=sha(HERE/'manifest.json'))
    save(HERE/'summary.json',out)
    table=[]
    for r in rows:
        item=dict(witness=r['witness_id'],**r['setting'])
        for key,c in r['cells'].items():
            item[key+'_fitted_ratio']=c['fitted_ratio'];item[key+'_true_ratio']=c['true_ratio_t0']
        for q,d in r['decompositions'].items():
            item.update({q+'_'+k+'_pp':v for k,v in d['components_pp'].items()})
        table.append(item)
    with (HERE/'per_witness.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['preflight','pilot','all'],required=True)
    args=parser.parse_args();checkpoint_guard();tic=time.perf_counter();m=freeze()
    for w in m['witnesses']:loaded_cells(w)
    if args.phase=='preflight':
        save(HERE/'preflight.json',dict(status='GO',witnesses=50,existing_four_cell_histories=200,
                                      new_discharge_integrations=0,manifest_sha256=sha(HERE/'manifest.json')))
        print('GO: 50 witnesses; 200 qualified source histories; frozen source hashes verified.');return
    ids=PILOT if args.phase=='pilot' else [w['id'] for w in m['witnesses']]
    workers=int(os.environ.get('AJ_COMPUTE_WORKERS','1'));records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,wid) for wid in ids]
        for future in as_completed(futures):
            r=future.result();records.append(r);checkpoint_guard()
            print(json.dumps(dict(id=r['witness_id'],passed=r['numerical_pass'],seconds=r['wall_seconds'])),flush=True)
    save(HERE/(args.phase+'_execution.json'),dict(completed=len(records),passed=all(r['numerical_pass'] for r in records),
                                               workers=workers,wall_seconds=time.perf_counter()-tic,manifest_sha256=sha(HERE/'manifest.json')))
    if args.phase=='all':summarize(m)
    assert all(r['numerical_pass'] for r in records),'A readout failed; retained without dropping the witness'


if __name__=='__main__':main()
