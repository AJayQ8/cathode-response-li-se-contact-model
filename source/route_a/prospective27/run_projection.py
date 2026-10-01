"""Project all jointly compatible witnesses into one declared new protocol."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import importlib.util
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location('bounded_assessment', HERE.parent/'robust23/resume_startup_repair.py')
assessment = importlib.util.module_from_spec(spec); spec.loader.exec_module(assessment)
model = assessment.model
read, save, sha = model.read, model.save, model.sha
J = 1.2
Q = .30
REST = .5


def freeze():
    evidence = read(HERE.parent/'uncertainty25/evidence.json')
    assert read(HERE.parent/'uncertainty25/verification.json')['passed']
    witnesses = []
    hashes = read(HERE.parent/'uncertainty25/manifest.json')['sha256'].copy()
    for p in [Path(__file__).resolve(), HERE/'PLAN.md', HERE.parent/'uncertainty25/evidence.json',
              HERE.parent/'uncertainty25/verification.json', HERE.parent/'uncertainty25/summarize_evidence.py']:
        hashes[str(p.relative_to(ROOT))] = sha(p)
    for group in evidence['groups']:
        for witness in group['retrospective_joint_compatible_witnesses']:
            path = ROOT/witness['source'] if witness['source'].startswith('science_lab/') else HERE.parent/'uncertainty25'/witness['source']
            fit = read(path)
            candidate = next(c for c in fit['candidates'] if c['label'] == witness['label'])
            assert candidate['x'] == witness['x'] and candidate['numerical_pass'] and candidate['physical_observation_pass']
            assert all(candidate['subsets'][s]['screen_pass'] for s in ['train', 'check'])
            item = dict(id='w%03d'%len(witnesses), source=str(path.relative_to(ROOT)),
                        source_sha256=sha(path), label=witness['label'], x=witness['x'],
                        setting={k:group[k] for k in ['registration', 'timing', 'extraction']},
                        rstar=fit['rstar'], original_subsets=candidate['subsets'])
            witnesses.append(item); hashes[item['source']] = item['source_sha256']
    assert len(witnesses) == 37
    for path, digest in hashes.items():
        assert sha(ROOT/path) == digest, path
    frozen = dict(sha256=hashes, witnesses=witnesses, current_mA_cm2=J,
                  discharge_mAh_cm2=Q, duration_h=Q/J, rest_h=REST,
                  all_14_known_points_used_for_admission=True,
                  fitting_performed=False, prospective_prediction_qualified=False,
                  central_claim_changed=False, intervals_are_certified=False)
    path = HERE/'manifest.json'
    if path.exists():
        assert read(path) == frozen
    else:
        save(path, frozen)
    return frozen


def original_wave(cathode):
    root = HERE.parent/'mechanics3'
    rows = list(csv.DictReader((root/'waveforms.csv').open()))
    meta = next(r for r in read(root/'waveform_summary.json') if r['cathode'] == cathode)
    points = [r for r in rows if r['cathode'] == cathode and r['quantity'] == 'pressure_kpa']
    tt = np.array([float(r['time_h']) for r in points])
    pp = np.array([float(r['value']) for r in points])
    t0 = meta['charge_end_voltage_max']['time_h']
    return tt, pp, t0, float(np.interp(t0, tt, pp)), meta['common_end_h']


def project(witness, cathode):
    x = np.array(witness['x']); L, H = 10**x[:2]; eta, b, A = x[2:]
    rstar, protocols = model.data(witness['setting']['extraction'], witness['setting']['registration'])
    assert rstar == witness['rstar']
    original = next(p for p in protocols if p['cathode'] == cathode and p['rate_c'] == 2.)
    p = dict(original, J=J, t1=Q/J, t2=0., rest=REST)
    loading = model.domain(p, eta)
    state, z0, a0 = model.initial_state(x, rstar, p['R_ch'])
    if not loading['valid'] or state is None:
        return dict(cathode=cathode, available=False, reason=loading['reason'] or 'Initial contact outside domain', loading=loading)
    fast = []; stopped = False; stripped = 0.
    for label, duration, current, qstart in [('end_discharge', Q/J, J, p['q0']),
                                             ('after_30_min_rest', REST, 0., p['q0']+Q)]:
        if not stopped:
            state, stopped, elapsed, projection = assessment.advance(state, duration, x, cathode, qstart, current, True)
        else:
            elapsed, projection = 0., 0.
        stripped += current*elapsed
        a = A*np.sqrt(state[0]); resistance = b*rstar+A*(1-b)*rstar/a
        fast.append(dict(stage=label, stopped=stopped, elapsed_h=elapsed, a=float(a),
                         resistance=float(resistance), ratio=None if stopped else float(resistance/p['R_ch']),
                         projection=float(projection), stripped_charge_mAh_cm2=float(stripped)))
    tt, pp, t0, p0, tend = original_wave(cathode)
    assert t0+(p['q0']+Q)/.12 <= tend+1e-10
    def load(q):
        return 2+eta*(float(np.interp(t0+q/.12, tt, pp))-p0)/1000
    velocity = H*L*A**(model.M+1)
    volume0 = L*a0*a0/2
    physical = np.array([L*(1-a0), 0.]); physical_stopped = False; independent_stripped = 0.
    rows = []
    for recorded, (duration, current, qstart) in zip(fast, [(Q/J, J, p['q0']), (REST, 0., p['q0']+Q)]):
        if not physical_stopped:
            def rhs(t, y):
                a = max(min(1-y[0]/L, 1.), A*model.ZMIN/4)
                pressure = load(qstart+current*t)
                assert pressure > 0
                recovery = velocity*(pressure/2)**model.M
                return [model.U*current/a-recovery*(1-a)/a**model.M,
                        recovery*(1-a)*a**(1-model.M)]
            def event(t, y):
                return y[0]-L*(1-A*model.ZMIN)
            event.terminal = True; event.direction = 1
            solution = solve_ivp(rhs, (0., duration), physical, method='Radau',
                                 rtol=1e-11, atol=1e-13, events=event, first_step=min(duration, 1e-4))
            assert solution.success, solution.message
            physical = solution.y[:, -1]
            physical_stopped = bool(len(solution.t_events[0]))
            independent_stripped += current*float(solution.t[-1])
        assert physical_stopped == recorded['stopped']
        a = max(0., min(1., 1-physical[0]/L)); volume = L*a*a/2
        ratio = None if physical_stopped else (b*rstar+A*(1-b)*rstar/a)/p['R_ch']
        discrepancy = None if ratio is None else abs(ratio-recorded['ratio'])
        ledger = abs(volume-volume0+model.U*independent_stripped-physical[1])
        # Any event is retained as unavailable, using its actual stripped charge.
        numeric = (discrepancy is None or discrepancy <= 1e-4) and ledger <= 1e-6*max(1., volume, abs(physical[1]))
        numeric = numeric and abs(independent_stripped-recorded['stripped_charge_mAh_cm2']) <= 1e-6
        numeric = numeric and recorded['projection'] <= 1e-7*max(1., 1/(A*A))
        physical_pass = (not physical_stopped and 0 < a <= 1 and L*a <= 40+1e-10
                         and volume/model.U+independent_stripped <= 40/model.U+1e-7 and physical[1] >= -1e-7)
        rows.append(dict(**recorded, independent_ratio=ratio, ratio_difference=discrepancy,
                         recession_um=float(physical[0]), replenished_volume_um=float(physical[1]),
                         remaining_volume_um=float(volume), initial_volume_um=float(volume0),
                         independent_stripped_charge_mAh_cm2=float(independent_stripped),
                         volume_ledger_error_um=float(ledger), numerical_pass=bool(numeric),
                         physical_pass=bool(physical_pass)))
    return dict(cathode=cathode, available=all(r['numerical_pass'] and r['physical_pass'] for r in rows),
                R_ch=p['R_ch'], q0=p['q0'], source_charged_rate_c=2., loading=loading, rows=rows)


def run(witness):
    checkpoint_guard(); tic = time.perf_counter()
    path = HERE/'cases'/(witness['id']+'.json')
    if path.exists():
        old = read(path)
        assert old['witness'] == witness and old['manifest_sha256'] == sha(HERE/'manifest.json')
        return old
    histories = []
    for cathode in ['P-LCO', 'N-LCO']:
        try:
            histories.append(project(witness, cathode))
        except (RuntimeError, ValueError, FloatingPointError, AssertionError) as exc:
            histories.append(dict(cathode=cathode, available=False, numerical_error=type(exc).__name__+': '+str(exc)))
        checkpoint_guard()
    output = dict(witness=witness, histories=histories, wall_seconds=time.perf_counter()-tic,
                  manifest_sha256=sha(HERE/'manifest.json'))
    save(path, output)
    return output


def main():
    checkpoint_guard(); tic = time.perf_counter(); frozen = freeze()
    workers = int(os.environ.get('AJ_COMPUTE_WORKERS', '1')); results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run, w) for w in frozen['witnesses']]):
            item = future.result(); results.append(item)
            print(json.dumps(dict(id=item['witness']['id'], available=all(h['available'] for h in item['histories']))), flush=True)
    save(HERE/'execution.json', dict(completed=len(results), workers=workers, wall_seconds=time.perf_counter()-tic,
                                     manifest_sha256=sha(HERE/'manifest.json')))


if __name__ == '__main__':
    main()
