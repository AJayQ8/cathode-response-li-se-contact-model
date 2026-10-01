"""Bounded projection/readout of terminal, independently qualified stage47 fits.

Historical functions are called with explicit new inputs; their drivers and
output directories are never reused. One worker, no kinetic fitting or launch.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import time

import numpy as np
import scipy
from shared_compute import checkpoint_guard, PoolError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = HERE.parent
SETTING = dict(registration='balance_aligned', timing='late', extraction='ordinary')
CATHODES = ['P-LCO', 'N-LCO']
J, Q, REST = 1.2, .30, .5
FORWARD_LIMIT, SPECTRUM_LIMIT, BUDGET = 1e-7, 1e-8, .001


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite durable output')
    return value


def save(path, value, guard=True):
    if guard:
        checkpoint_guard()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(plain(value), indent=2, allow_nan=False)+'\n'
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def artifact(path, value):
    value = dict(value, manifest_sha256=sha(HERE/'manifest.json'))
    save(path, value)
    save(HERE/'receipts'/str(path.relative_to(HERE)).replace('/', '__'),
         dict(path=str(path.relative_to(HERE)), sha256=sha(path),
              manifest_sha256=sha(HERE/'manifest.json')))
    return value


def verified(path, stage=HERE):
    relative = str(path.relative_to(stage))
    receipt = read(stage/'receipts'/relative.replace('/', '__'))
    require(receipt['path'] == relative and receipt['sha256'] == sha(path) and
            receipt['manifest_sha256'] == sha(stage/'manifest.json'), 'Receipt mismatch: '+relative)
    value = read(path)
    require(value['manifest_sha256'] == receipt['manifest_sha256'], 'Manifest mismatch: '+relative)
    return value


def add_hashes(destination, values):
    for name, digest in values.items():
        require(name not in destination or destination[name] == digest, 'Conflicting digest: '+name)
        destination[name] = digest


def check_candidate(c, setting):
    """Recheck saved qualification evidence without fitting or changing a value."""
    require(c['qualified_witness'] and c['numerical_pass'] and c['physical_observation_pass'] and
            c['all_known_screen_pass'], 'Candidate is not independently qualified')
    f, a = c['forward'], c['physical_coordinate_audit']
    rows = f['tight']
    require(len(rows) == 14 and f['row_order_matches'] and f['statuses_match'] and
            f['physical_statuses_match'] and all(v['passed'] for v in f['comparisons']),
            'Incomplete forward qualification')
    require(a['passed'] and a['physical_observation_pass'] and a['audited_row_count'] == 14,
            'Incomplete physical-coordinate qualification')
    require(all(r.get('predicted_ratio') is not None and not r.get('disconnected', False) and
                not r.get('observation_invalid', False) for r in rows), 'Unavailable known observation')
    error = max(abs(r['predicted_ratio']-r['observed_ratio']) for r in rows)
    noise = max(f['max_ratio_difference'], a['maxima']['ratio_difference'])
    require(error == c['all_known_max_error'] and error <= .03 and noise <= FORWARD_LIMIT and
            .03-error > noise and c['screen_margin'] == .03-error and
            c['observed_numerical_disagreement'] == noise, 'Qualification margin mismatch')
    require(f['x'] == c['x'] and all(f['task'][k] == v for k, v in setting.items()) and
            float(c['x'][2]) == c['eta'],
            'Candidate coordinates/settings mismatch')
    return error, noise


def freeze():
    stage = BASE/'transfer47'
    require((stage/'execution_pilot.json').is_file(), 'Stage47 pilot must be terminal before selection')
    m47 = read(stage/'manifest.json')
    completed = verified(stage/'execution_pilot.json', stage)
    controls47 = verified(stage/'controls.json', stage)
    require(completed['completed'] == 2 and len(completed['tasks']) == 2 and controls47['passed'],
            'Stage47 pilot/controls incomplete')
    require(m47['screen'] == .03 and m47['known_observations'] == 14 and
            m47['forward_limit'] == FORWARD_LIMIT and len(m47['tasks']) == 2,
            'Stage47 scientific contract changed')
    threshold = min(s['x'][2] for s in m47['seeds'])
    selected, decisions, inputs = [], [], []
    for task in m47['tasks']:
        key = task['id']; result_path = stage/'results'/(key+'.json')
        raw_path = stage/'optimizer_returns'/(key+'.json')
        result, raw = verified(result_path, stage), verified(raw_path, stage)
        execution = next(v for v in completed['tasks'] if v['task'] == task)
        require(execution['result_sha256'] == sha(result_path) and result['task'] == raw['task'] == task and
                result['optimizer_return_sha256'] == sha(raw_path) and
                result['seed'] == raw['seed'] == m47['seeds'][task['seed']] and
                raw['status'] in ['optimizer_returned', 'evaluation_cap'] and
                result['optimizer_status'] == raw['status'], 'Stage47 terminal result mismatch')
        setting = {k: task[k] for k in SETTING}
        require(setting == SETTING, 'This follow-up freezes one interpretation only')
        candidates = [c for c in result['candidates'] if c['qualified_witness']]
        for c in candidates:
            check_candidate(c, setting)
        best = min(candidates, key=lambda c: c['eta']) if candidates else None
        require(result['qualified_witness_found'] == (best is not None) and
                result['smallest_witnessed_eta'] == (None if best is None else best['eta']) and
                result['selected_label'] == (None if best is None else best['label']),
                'Stage47 selected candidate mismatch')
        eligible = best is not None and best['eta'] < threshold
        duplicate = bool(eligible and any(w['x'] == best['x'] for w in selected))
        decisions.append(dict(task_id=key, selected_label=None if best is None else best['label'],
                              eta=None if best is None else best['eta'], lower_than_frozen_seed_minimum=eligible,
                              exact_duplicate_of_prior=duplicate, admitted=eligible and not duplicate))
        if eligible and not duplicate:
            selected.append(dict(id='s47_seed'+str(task['seed']), x=best['x'], setting=setting,
                                 rstar=best['forward']['rstar'], eta=best['eta'], source=str(result_path.relative_to(ROOT)),
                                 source_sha256=sha(result_path), label=best['label'],
                                 all_known_max_error=best['all_known_max_error'],
                                 observed_numerical_disagreement=best['observed_numerical_disagreement']))
        inputs += [result_path, raw_path, stage/'receipts'/('results__'+key+'.json'),
                   stage/'receipts'/('optimizer_returns__'+key+'.json')]
    require(len(selected) <= 2, 'Bounded selection exceeded')
    m35 = read(BASE/'sweep35/manifest.json')
    hashes = m47['sha256'].copy(); add_hashes(hashes, m35['sha256'])
    inputs += [stage/'manifest.json', stage/'execution_pilot.json', stage/'controls.json',
               stage/'receipts/execution_pilot.json', stage/'receipts/controls.json',
               BASE/'sweep35/manifest.json', BASE/'sweep35/controls.json', BASE/'sweep35/summary.json',
               BASE/'sweep35/verify_and_summarize.py', BASE/'sweep35/PRECISION_REPAIR.md',
               BASE/'readout34/manifest.json', BASE/'readout34/controls.json',
               BASE/'readout34/summary.json', BASE/'readout34/verify_and_summarize.py',
               BASE/'prospective27/cases/w026.json', BASE/'readout34/cases/w026.json',
               BASE/'decomposition29/manifest.json', BASE/'decomposition29/controls.json',
               BASE/'decomposition29/run_decomposition.py', BASE/'decomposition29/verify_and_summarize.py',
               BASE/'decomposition29/cases/w026.json',
               BASE/'sweep35/cases/w026_fixed_q_high_to_low.json',
               HERE/'PLAN.md', HERE/'run_prediction.py', HERE/'audit_prediction.py', HERE/'job_prediction.json']
    c35, c34 = read(BASE/'sweep35/controls.json'), read(BASE/'readout34/controls.json')
    c29 = read(BASE/'decomposition29/controls.json')
    require(c35['passed'] and c35['manifest_sha256'] == sha(BASE/'sweep35/manifest.json') and
            c34['passed'] and c34['manifest_sha256'] == sha(BASE/'readout34/manifest.json') and
            read(BASE/'sweep35/summary.json')['verification_passed'] and
            read(BASE/'readout34/summary.json')['verification_passed'] and
            c29['passed'] and c29['manifest_sha256'] == sha(BASE/'decomposition29/manifest.json'),
            'Historical controls no longer qualify')
    oldcase = read(BASE/'sweep35/cases/w026_fixed_q_high_to_low.json')
    for c in CATHODES:
        reference = next(r for r in c35['records'] if r['extraction'] == 'ordinary' and r['cathode'] == c)
        require(reference['passed'], 'Ordinary stationary reference failed')
        inputs.append(BASE/'sweep35'/reference['fit_path'])
        oldh = next(h for h in oldcase['histories'] if h['cathode'] == c)
        inputs.append(BASE/'sweep35'/oldh['fit_path'])
    for path in inputs:
        add_hashes(hashes, {str(path.relative_to(ROOT)): sha(path)})
    for name, digest in hashes.items():
        require(sha(ROOT/name) == digest, 'Frozen dependency changed: '+name)
    frozen = dict(schema=1, sha256=hashes, witnesses=selected, selection=decisions,
                  lower_eta_threshold=threshold, selection_uses_future_response=False,
                  current_mA_cm2=J, discharge_mAh_cm2=Q, discharge_h=Q/J, rest_h=REST,
                  law='fixed_q', order='high_to_low', source_frequencies=68,
                  settling_seconds=.5, cycles_per_point=3, quadrature_orders=[8, 16],
                  fit_tolerances=1e-13, forward_limit=FORWARD_LIMIT, spectrum_limit=SPECTRUM_LIMIT,
                  instantaneous_bias_budget=BUDGET, reference_witness='w026',
                  off_diagonal_pairs=[['P-LCO', 'N-LCO'], ['N-LCO', 'P-LCO']],
                  history_exchange_includes_q0=True, off_diagonal_spectral_fitting=False,
                  worker_count=1, kinetic_refitting=False, central_claim_changed=False,
                  global_bounds=False, experimental_validation=False,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    path = HERE/'manifest.json'
    if path.exists():
        require(read(path) == frozen, 'Frozen stage49 inputs changed; retain the original attempt')
    else:
        require(not any((HERE/p).exists() for p in ['cases', 'controls.json', 'spectral_fits']),
                'Outputs exist without stage49 manifest')
        save(path, frozen)
    return frozen


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def references():
    out = []
    for c in CATHODES:
        r = deepcopy(next(r for r in read(BASE/'sweep35/controls.json')['records']
                          if r['extraction'] == 'ordinary' and r['cathode'] == c))
        r['fit'] = read(BASE/'sweep35'/r['fit_path'])
        out.append(r)
    return out


def rest_input(w, h, sweep):
    x = w['x']; L, H = 10**np.asarray(x[:2]); A = x[4]
    pfast, pdirect = sweep.rest.pressures(w, h['cathode'], h['q0']+Q)
    gamma = H*A**(sweep.rest.model.M+1)*(pdirect/2)**sweep.rest.model.M
    a0 = 1-h['rows'][0]['recession_um']/L
    return dict(cathode=h['cathode'], R_ch=h['R_ch'], q0=h['q0'],
                pressures_mpa=[pfast, pdirect], gammas_per_hour=[gamma, gamma],
                rows=[dict(independent=dict(contact=[a0], ratio=h['rows'][0]['independent_ratio']))])


def synthesize(w, h, sweep):
    checkpoint_guard()
    info = rest_input(w, h, sweep)
    t = sweep.make_template('ordinary', h['cathode'], h['R_ch'])
    f = sweep.frequency(); schedule, duration = sweep.schedule(f, 'high_to_low')
    require(len(f) == 68 and abs(duration-106.9561255903888) <= 1e-10, 'Frozen readout schedule changed')
    z, points = sweep.generate(w, info, t, 'fixed_q', schedule, f, 8)
    zz, _ = sweep.generate(w, info, t, 'fixed_q', schedule, f, 16, True)
    error = float(np.max(np.abs(z-zz)/np.abs(zz)))
    b, A = w['x'][3:]; C = A*(1-b)*w['rstar']; rb = b*w['rstar']
    aend = sweep.rest.implicit_contact(info['rows'][0]['independent']['contact'][0],
                                     info['gammas_per_hour'][1], duration)
    return dict(cathode=h['cathode'], template=t, points=points,
                re_ohm_cm2=z.real.tolist(), minus_im_ohm_cm2=(-z.imag).tolist(),
                direct16_re_ohm_cm2=zz.real.tolist(), direct16_minus_im_ohm_cm2=(-zz.imag).tolist(),
                quadrature_direct_error=error, quadrature_pass=error <= SPECTRUM_LIMIT,
                true_ratio_t0=h['rows'][0]['independent_ratio'],
                true_ratio_tend=float((rb+C/aend)/h['R_ch']), duration_s=duration,
                max_fractional_R_change_per_cycle=max(max(p['fractional_R_change_per_cycle_at_endpoints']) for p in points))


def fit_spectrum(w, row, reference, sweep):
    """Unchanged precision-configured fitter; exact inputs and new output paths."""
    sid = w['id']+'_'+row['cathode'][0]
    task = dict(spectrum_id=sid, topology='source', loss='linear', fraction=0.)
    spectrum = dict(re_ohm_cm2=row['re_ohm_cm2'], minus_im_ohm_cm2=row['minus_im_ohm_cm2'],
                    source_bulk_li_r=row['template']['Rref'], source_cathode_r=row['template']['slow_r'])
    inputs = dict(task=task, spectrum=spectrum, frequencies_hz=sweep.frequency().tolist(),
                  witness=w, template=row['template'], fit_tolerances=1e-13,
                  source_fitter_sha256=sha(BASE/'impedance8/run_circuit_probe.py'),
                  precision_wrapper_sha256=sha(BASE/'sweep35/run_sweep.py'))
    digest = content_sha(inputs)
    target = HERE/'spectral_fits'/(sid+'.json')
    if target.exists():
        saved = verified(target)
        require(saved['input_sha256'] == digest and saved['inputs'] == inputs, 'Spectral fit input mismatch')
        fitted = saved['fit']
    else:
        # Never trust a pre-existing raw fitter output without a completed receipt.
        rawdir = HERE/'spectral_work'/sid
        rawpath = rawdir/(sweep.circuit.task_id(task)+'.json')
        require(not rawpath.exists(), 'Unreceipted raw fit preserved; inspect before any retry: '+str(rawpath))
        input_path = HERE/'fit_inputs'/(sid+'.json')
        if input_path.exists():
            saved_input = verified(input_path)
            require(saved_input['input_sha256'] == digest and saved_input['inputs'] == inputs,
                    'Preserved fit intent differs from current inputs')
        else:
            artifact(input_path, dict(input_sha256=digest, inputs=inputs))
        previous_out = sweep.circuit.OUT
        try:
            sweep.circuit.OUT = rawdir
            sweep.circuit.fit_one(task, spectrum, sweep.frequency())
        finally:
            sweep.circuit.OUT = previous_out
        fitted = read(rawpath)
        require(fitted['task'] == task and len(fitted['starts']) == 6, 'Unexpected fitter result')
        artifact(target, dict(input_sha256=digest, inputs=inputs, fit=fitted, raw_sha256=sha(rawpath)))
    best = fitted['best']; Rfit = best['parameters']['fast']['r_ohm_cm2']
    ratio = Rfit/reference['fitted_reference']; lo, hi = row['true_ratio_tend'], row['true_ratio_t0']
    return dict(row, fit=fitted, fitted_ratio=ratio,
                fitted_ratio_true_reference=Rfit/row['template']['Rref'],
                fit_inside_instantaneous_extrema=bool(lo-1e-6 <= ratio <= hi+1e-6),
                normalized_outside_extrema=max(0., lo-ratio, ratio-hi),
                selected_fit_success=best['success'], selected_bounds=best['bound_indices'],
                all_start_failures=sum(not v['success'] for v in fitted['starts']),
                branch_log10_separation=best['parameters']['branch_log10_separation'],
                complex_relative_rms=fitted['complex_relative_rms'], fit_path=str(target.relative_to(HERE)))


def controls(frozen, projection, sweep, auditor, refs):
    path = HERE/'controls.json'
    require(not (HERE/'controls_failure.json').exists(), 'Earlier control exception preserved; inspect before retry')
    if path.exists():
        saved = verified(path); require(saved['passed'], 'Preserved stage49 controls failed'); return saved
    old = read(BASE/'prospective27/cases/w026.json'); w = old['witness']
    require(w['setting'] == SETTING and w['x'] == read(BASE/'transfer47/manifest.json')['seeds'][1]['x'],
            'Frozen w026 comparison witness mismatch')
    oldread = read(BASE/'sweep35/cases/w026_fixed_q_high_to_low.json')
    tests, histories, readouts = [], [], []
    for c in CATHODES:
        checkpoint_guard()
        h = projection.project(w, c); histories.append(h)
        before = next(v for v in old['histories'] if v['cathode'] == c)
        require(h['available'] and before['available'], 'Historical trajectory control unavailable')
        error = max(abs(x['ratio']-y['ratio']) for x, y in zip(h['rows'], before['rows']))
        generated = synthesize(w, h, sweep)
        original = deepcopy(next(v for v in oldread['histories'] if v['cathode'] == c))
        z = np.array(generated['re_ohm_cm2'])-1j*np.array(generated['minus_im_ohm_cm2'])
        zz = np.array(original['re_ohm_cm2'])-1j*np.array(original['minus_im_ohm_cm2'])
        ze = float(np.max(np.abs(z-zz)/np.abs(zz)))
        original['fit'] = read(BASE/'sweep35'/original['fit_path']); readouts.append(original)
        reference = next(v for v in refs if v['cathode'] == c)
        require(generated['template'] == reference['template'], 'Stationary template changed')
        t = generated['template']; f = sweep.frequency()
        static = sweep.forward(f, np.full(len(f), t['Rref']), t, 'fixed_q')
        direct = sweep.forward(f, np.full(len(f), t['Rref']), t, 'fixed_q', True)
        algebra = float(np.max(np.abs(static-direct)/np.abs(direct)))
        tests.append(dict(cathode=c, saved_projection_ratio_difference=error,
                          saved_primary_spectrum_difference=ze, zero_recovery_algebra_error=algebra,
                          quadrature_error=generated['quadrature_direct_error'],
                          passed=bool(error <= FORWARD_LIMIT and ze <= FORWARD_LIMIT and
                                      algebra <= 1e-12 and generated['quadrature_pass'])))
    audit = auditor.audit_case(w, histories, readouts, refs)
    oldcross = read(BASE/'decomposition29/cases/w026.json')
    require(oldcross['witness']['x'] == w['x'] and oldcross['witness']['setting'] == w['setting'],
            'Historical crossed-history control witness mismatch')
    swap_audit = auditor.audit_swaps(w, histories, oldcross['crossed_histories'])
    passed = all(t['passed'] for t in tests) and audit['passed'] and swap_audit['passed']
    saved = artifact(path, dict(tests=tests, audit=audit, swap_audit=swap_audit, passed=passed,
                                historical_controls_reused=True, new_kinetic_fitting=False))
    require(passed, 'Stage49 trajectory/readout adapter controls failed')
    return saved


def one(w, projection, decomposition, sweep, auditor, refs):
    path = HERE/'cases'/(w['id']+'.json')
    if path.exists():
        previous = verified(path); require(previous['witness'] == w, 'Case witness changed'); return previous
    checkpoint_guard(); started = time.perf_counter()
    histories = []
    for c in CATHODES:
        hp = HERE/'trajectories'/(w['id']+'_'+c[0]+'.json')
        if hp.exists():
            saved = verified(hp); require(saved['witness'] == w and saved['cathode'] == c, 'Trajectory inputs changed')
            h = saved['history']
        else:
            try:
                h = projection.project(w, c)
            except PoolError:
                raise
            except (RuntimeError, ValueError, FloatingPointError, AssertionError) as exc:
                h = dict(cathode=c, available=False, error=type(exc).__name__+': '+str(exc))
            artifact(hp, dict(witness=w, cathode=c, history=h))
        histories.append(h)
    if not all(h['available'] for h in histories):
        return artifact(path, dict(witness=w, histories=histories, readout_rows=[], available=False,
                                   status='prospective_trajectory_unavailable', wall_seconds=time.perf_counter()-started))
    projection_difference = max(r['ratio_difference'] for h in histories for r in h['rows'])
    if projection_difference > FORWARD_LIMIT:
        return artifact(path, dict(witness=w, histories=histories, readout_rows=[], available=False,
                                   status='prospective_coordinate_agreement_failed',
                                   max_coordinate_ratio_difference=projection_difference,
                                   wall_seconds=time.perf_counter()-started))
    crossed = []
    for initial, history in [('P-LCO', 'N-LCO'), ('N-LCO', 'P-LCO')]:
        checkpoint_guard()
        xp = HERE/'trajectories'/(w['id']+'_'+initial[0]+'_start_'+history[0]+'_history.json')
        if xp.exists():
            saved = verified(xp)
            require(saved['witness'] == w and saved['initial_cathode'] == initial and
                    saved['history_cathode'] == history, 'Crossed-history inputs changed')
            h = saved['history']
        else:
            try:
                h = decomposition.project(w, initial, history)
            except PoolError:
                raise
            except (RuntimeError, ValueError, FloatingPointError, AssertionError) as exc:
                h = dict(initial_cathode=initial, history_cathode=history, available=False,
                         error=type(exc).__name__+': '+str(exc))
            artifact(xp, dict(witness=w, initial_cathode=initial, history_cathode=history, history=h))
        crossed.append(h)
    swap_audit = auditor.audit_swaps(w, histories, crossed)
    rows = []
    for h in histories:
        row = synthesize(w, h, sweep)
        if not row['quadrature_pass']:
            return artifact(path, dict(witness=w, histories=histories, crossed_histories=crossed,
                                       history_decomposition_audit=swap_audit, readout_rows=rows+[row], available=False,
                                       status='quadrature_failed', wall_seconds=time.perf_counter()-started))
        reference = next(v for v in refs if v['cathode'] == h['cathode'])
        require(row['template'] == reference['template'], 'New witness template differs from frozen charged state')
        rows.append(fit_spectrum(w, row, reference, sweep))
        checkpoint_guard()
    audit = auditor.audit_case(w, histories, rows, refs)
    p, n = rows; contrast = p['fitted_ratio']-n['fitted_ratio']
    start = p['true_ratio_t0']-n['true_ratio_t0']
    stop = histories[0]['rows'][1]['independent_ratio']-histories[1]['rows'][1]['independent_ratio']
    interval = [p['true_ratio_tend']-n['true_ratio_t0'], p['true_ratio_t0']-n['true_ratio_tend']]
    return artifact(path, dict(witness=w, histories=histories, crossed_histories=crossed,
                               history_decomposition_audit=swap_audit, readout_rows=rows, audit=audit,
                               status='completed', available=audit['passed'] and swap_audit['passed'],
                               readout_available=audit['passed'], history_effect_available=swap_audit['passed'],
                               max_coordinate_ratio_difference=projection_difference,
                               fitted_contrast=contrast,
                               true_contrast_t0=start, true_contrast_after_30_min_rest=stop,
                               positive_contrast=contrast > 0, bias_from_t0=contrast-start,
                               instantaneous_budget_pass=abs(contrast-start) <= BUDGET,
                               true_monotone_window_interval=interval,
                               pair_inside_true_window=interval[0]-2e-6 <= contrast <= interval[1]+2e-6,
                               wall_seconds=time.perf_counter()-started))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['controls', 'all'], required=True)
    args = parser.parse_args()
    require(int(os.environ.get('AJ_COMPUTE_WORKERS', '1')) == 1, 'Stage49 requires exactly one worker')
    checkpoint_guard(); frozen = freeze()
    if not frozen['witnesses']:
        artifact(HERE/'execution.json', dict(status='no_qualified_lower_eta_candidate', selected=0,
                                           cases=[], scientific_conclusion='No forecast attempted'))
        print(json.dumps(dict(status='no_qualified_lower_eta_candidate')), flush=True); return
    projection = module('transfer49_projection', BASE/'prospective27/run_projection.py')
    decomposition = module('transfer49_decomposition', BASE/'decomposition29/run_decomposition.py')
    sweep = module('transfer49_sweep', BASE/'sweep35/run_sweep.py')
    auditor = module('transfer49_independent_auditor', HERE/'audit_prediction.py')
    require((projection.J, projection.Q, projection.REST) == (J, Q, REST), 'Projection protocol changed')
    require((decomposition.J, decomposition.Q, decomposition.REST) == (J, Q, REST),
            'Crossed-history protocol changed')
    refs = references()
    try:
        controls(frozen, projection, sweep, auditor, refs)
    except PoolError:
        raise
    except (RuntimeError, ValueError, FloatingPointError, AssertionError, KeyError) as exc:
        failure = HERE/'controls_failure.json'
        if not failure.exists():
            save(failure, dict(error=type(exc).__name__+': '+str(exc),
                               manifest_sha256=sha(HERE/'manifest.json')), guard=False)
        raise
    if args.mode == 'controls':
        print(json.dumps(dict(mode='controls', passed=True)), flush=True); return
    cases = []
    for witness in frozen['witnesses']:
        result = one(witness, projection, decomposition, sweep, auditor, refs)
        record = dict(id=witness['id'], eta=witness['eta'], status=result['status'], available=result['available'],
                      result_sha256=sha(HERE/'cases'/(witness['id']+'.json')),
                      fitted_contrast=result.get('fitted_contrast'), positive_contrast=result.get('positive_contrast'),
                      instantaneous_budget_pass=result.get('instantaneous_budget_pass'))
        cases.append(record); print(json.dumps(record), flush=True)
    old = read(BASE/'sweep35/cases/w026_fixed_q_high_to_low.json')
    artifact(HERE/'execution.json', dict(status='completed', selected=len(cases), cases=cases,
                                       old_w026_fitted_contrast=old['fitted_contrast'],
                                       old_w026_case_sha256=sha(BASE/'sweep35/cases/w026_fixed_q_high_to_low.json'),
                                       central_claim_changed=False, experimental_validation=False))


if __name__ == '__main__':
    main()
