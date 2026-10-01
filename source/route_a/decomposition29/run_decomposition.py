"""Cross initial contact states and pressure inputs without refitting."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import importlib.util
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location('frozen_projection', HERE.parent/'prospective27/run_projection.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
model, assessment = prior.model, prior.assessment
read, save, sha = prior.read, prior.save, prior.sha
J, Q, REST = prior.J, prior.Q, prior.REST
CATHODES = ['P-LCO', 'N-LCO']
STAGES = ['end_discharge', 'after_30_min_rest']


def freeze():
    old = read(HERE.parent/'prospective27/manifest.json')
    later = read(HERE.parent/'counterexample28/manifest.json')
    audited = read(HERE.parent/'counterexample28/summary.json')
    assert audited['passed'] and audited['retained_distinct_witnesses'] == 50
    hashes = old['sha256'].copy()
    for path, digest in later['sha256'].items():
        assert path not in hashes or hashes[path] == digest
        hashes[path] = digest
    for name, digest in audited['sha256'].items():
        path = HERE.parent/'counterexample28'/name
        assert sha(path) == digest
        hashes[str(path.relative_to(ROOT))] = digest
    records = []
    for w in old['witnesses']:
        path = HERE.parent/'prospective27/cases'/(w['id']+'.json')
        c = read(path)
        assert c['witness'] == w and c['manifest_sha256'] == sha(HERE.parent/'prospective27/manifest.json')
        records.append(dict(source=str(path.relative_to(ROOT)), source_label=None,
                            original_id='stage27/'+w['id'], setting=w['setting'], x=w['x'],
                            rstar=w['rstar'], diagonal_histories=c['histories']))
    for task in later['tasks']:
        fid = task['extraction']+'_'+task['direction']+'_seed'+str(task['seed'])
        path = HERE.parent/'counterexample28/searches'/(fid+'.json')
        fit = read(path)
        for c in fit['candidates']:
            if c['all_known_admitted']:
                records.append(dict(source=str(path.relative_to(ROOT)), source_label=c['label'],
                                    original_id=fid+'/'+c['label'],
                                    setting={k:task[k] for k in ['registration', 'timing', 'extraction']},
                                    x=c['x'], rstar=fit['rstar'], diagonal_histories=c['new_protocol']))
    distinct = {}
    for r in records:
        key = tuple(r['setting'][k] for k in ['registration', 'timing', 'extraction'])+tuple(r['x'])
        distinct.setdefault(key, r)
    witnesses = list(distinct.values())
    assert len(witnesses) == 50
    for i, w in enumerate(witnesses):
        w['id'] = 'w%03d'%i
        hashes[w['source']] = sha(ROOT/w['source'])
        assert len(w['diagonal_histories']) == 2 and all(h['available'] for h in w['diagonal_histories'])
    controls = [next(w['id'] for w in witnesses if w['setting'] == dict(registration='balance_aligned', timing='late', extraction=e))
                for e in ['published', 'ordinary', 'robust']]
    for path in [Path(__file__).resolve(), HERE/'PLAN.md', HERE.parent/'prospective27/manifest.json',
                 HERE.parent/'counterexample28/summary.json', HERE.parent/'counterexample28/verify_and_summarize.py']:
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for name, digest in hashes.items():
        assert sha(ROOT/name) == digest, name
    frozen = dict(sha256=hashes, witnesses=witnesses, control_ids=controls,
                  current_mA_cm2=J, discharge_mAh_cm2=Q, duration_h=Q/J, rest_h=REST,
                  loading_exchange_includes_q0=True, initial_state_keeps_own_normalization=True,
                  fitting_performed=False, all_14_known_points_used_for_admission=True,
                  central_claim_changed=False, prospective_prediction_qualified=False)
    path = HERE/'manifest.json'
    if path.exists():
        assert read(path) == frozen, 'Exact manifest required'
    else:
        save(path, frozen)
    return frozen


def project(witness, initial_cathode, history_cathode):
    x = np.array(witness['x'])
    L, H = 10**x[:2]
    eta, b, A = x[2:]
    rstar, protocols = model.data(witness['setting']['extraction'], witness['setting']['registration'])
    assert rstar == witness['rstar']
    initial = next(p for p in protocols if p['cathode'] == initial_cathode and p['rate_c'] == 2.)
    history = next(p for p in protocols if p['cathode'] == history_cathode and p['rate_c'] == 2.)
    p = dict(history, J=J, t1=Q/J, t2=0., rest=REST)
    loading = model.domain(p, eta)
    state, z0, a0 = model.initial_state(x, rstar, initial['R_ch'])
    result = dict(initial_cathode=initial_cathode, history_cathode=history_cathode,
                  R_ch=initial['R_ch'], q0=history['q0'], initial_a=a0, initial_z=z0,
                  source_charged_rate_c=2., loading=loading)
    if not loading['valid'] or state is None:
        return dict(**result, available=False, reason=loading['reason'] or 'Initial contact outside domain')
    segments = [(Q/J, J, p['q0']), (REST, 0., p['q0']+Q)]
    fast = []
    stopped, stripped = False, 0.
    for label, (duration, current, qstart) in zip(STAGES, segments):
        if not stopped:
            state, stopped, elapsed, projection = assessment.advance(state, duration, x, history_cathode, qstart, current, True)
        else:
            elapsed, projection = 0., 0.
        stripped += current*elapsed
        a = A*np.sqrt(state[0])
        resistance = b*rstar+A*(1-b)*rstar/a
        fast.append(dict(stage=label, stopped=stopped, elapsed_h=elapsed, a=float(a),
                         resistance=float(resistance), ratio=None if stopped else float(resistance/initial['R_ch']),
                         projection=float(projection), stripped_charge_mAh_cm2=float(stripped)))
    # Independent loading interpolation uses the original source-time arrays.
    tt, pp, t0, p0, tend = prior.original_wave(history_cathode)
    assert t0+(p['q0']+Q)/.12 <= tend+1e-10

    def load(q):
        return 2+eta*(float(np.interp(t0+q/.12, tt, pp))-p0)/1000

    velocity = H*L*A**(model.M+1)
    volume0 = L*a0*a0/2
    physical = np.array([L*(1-a0), 0.])
    physical_stopped, independent_stripped = False, 0.
    rows = []
    for recorded, (duration, current, qstart) in zip(fast, segments):
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

            event.terminal = True
            event.direction = 1
            solution = solve_ivp(rhs, (0., duration), physical, method='Radau',
                                 rtol=1e-11, atol=1e-13, events=event, first_step=min(duration, 1e-4))
            assert solution.success, solution.message
            physical = solution.y[:, -1]
            physical_stopped = bool(len(solution.t_events[0]))
            independent_stripped += current*float(solution.t[-1])
        a = max(0., min(1., 1-physical[0]/L))
        volume = L*a*a/2
        ratio = None if physical_stopped else (b*rstar+A*(1-b)*rstar/a)/initial['R_ch']
        discrepancy = None if ratio is None or recorded['ratio'] is None else abs(ratio-recorded['ratio'])
        ledger = abs(volume-volume0+model.U*independent_stripped-physical[1])
        numeric = (physical_stopped == recorded['stopped'] and
                   (discrepancy is None or discrepancy <= 1e-4) and
                   ledger <= 1e-6*max(1., volume, abs(physical[1])) and
                   abs(independent_stripped-recorded['stripped_charge_mAh_cm2']) <= 1e-6 and
                   recorded['projection'] <= 1e-7*max(1., 1/(A*A)))
        physical_pass = (not physical_stopped and not recorded['stopped'] and 0 < a <= 1 and
                         L*a <= 40+1e-10 and volume/model.U+independent_stripped <= 40/model.U+1e-7 and
                         physical[1] >= -1e-7)
        rows.append(dict(**recorded, independent_stopped=physical_stopped, independent_ratio=ratio,
                         ratio_difference=discrepancy, recession_um=float(physical[0]),
                         replenished_volume_um=float(physical[1]), remaining_volume_um=float(volume),
                         initial_volume_um=float(volume0), independent_stripped_charge_mAh_cm2=float(independent_stripped),
                         volume_ledger_error_um=float(ledger), numerical_pass=bool(numeric), physical_pass=bool(physical_pass)))
    return dict(**result, available=all(r['numerical_pass'] and r['physical_pass'] for r in rows), rows=rows)


def controls(frozen):
    path = HERE/'controls.json'
    if path.exists():
        old = read(path)
        assert old['manifest_sha256'] == sha(HERE/'manifest.json') and old['passed']
        return old
    tests = []
    for wid in frozen['control_ids']:
        w = next(w for w in frozen['witnesses'] if w['id'] == wid)
        diagonals = []
        for c in CATHODES:
            actual = project(w, c, c)
            reference = next(h for h in w['diagonal_histories'] if h['cathode'] == c)
            differences = [abs(a['ratio']-b['ratio']) for a, b in zip(actual['rows'], reference['rows'])]
            diagonals.append(dict(cathode=c, actual=actual, ratio_differences=differences,
                                  passed=actual['available'] == reference['available'] and max(differences) <= 1e-4))
            checkpoint_guard()
        zero = dict(w, x=w['x'].copy())
        zero['x'][2] = 0.
        nulls = [project(zero, 'P-LCO', c) for c in CATHODES]
        differences = [abs(a['ratio']-b['ratio']) for a, b in zip(nulls[0]['rows'], nulls[1]['rows'])]
        null_pass = all(h['available'] for h in nulls) and max(differences) <= 1e-7
        tests.append(dict(id=wid, diagonals=diagonals, eta_zero_histories=nulls,
                          eta_zero_differences=differences, eta_zero_passed=null_pass,
                          passed=all(d['passed'] for d in diagonals) and null_pass))
        checkpoint_guard()
    result = dict(tests=tests, passed=all(t['passed'] for t in tests), manifest_sha256=sha(HERE/'manifest.json'))
    save(path, result)
    assert result['passed'], 'Input-exchange controls must pass before the batch'
    return result


def run(witness):
    checkpoint_guard()
    tic = time.perf_counter()
    path = HERE/'cases'/(witness['id']+'.json')
    if path.exists():
        old = read(path)
        assert old['witness'] == witness and old['manifest_sha256'] == sha(HERE/'manifest.json')
        return old
    crossed = []
    for initial, history in [('P-LCO', 'N-LCO'), ('N-LCO', 'P-LCO')]:
        try:
            crossed.append(project(witness, initial, history))
        except (RuntimeError, ValueError, FloatingPointError, AssertionError) as exc:
            crossed.append(dict(initial_cathode=initial, history_cathode=history, available=False,
                                numerical_error=type(exc).__name__+': '+str(exc)))
        checkpoint_guard()
    output = dict(witness=witness, crossed_histories=crossed, wall_seconds=time.perf_counter()-tic,
                  manifest_sha256=sha(HERE/'manifest.json'))
    save(path, output)
    return output


def main():
    checkpoint_guard()
    tic = time.perf_counter()
    frozen = freeze()
    controls(frozen)
    workers = int(os.environ.get('AJ_COMPUTE_WORKERS', '1'))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run, w) for w in frozen['witnesses']]):
            item = future.result()
            results.append(item)
            print(json.dumps(dict(id=item['witness']['id'], available=all(h['available'] for h in item['crossed_histories']))), flush=True)
            checkpoint_guard()
    save(HERE/'execution.json', dict(completed=len(results), workers=workers, wall_seconds=time.perf_counter()-tic,
                                     manifest_sha256=sha(HERE/'manifest.json')))


if __name__ == '__main__':
    main()
