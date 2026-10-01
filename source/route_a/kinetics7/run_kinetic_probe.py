"""Bounded contact-kinetics diagnostic; not the paper's spatial-field model."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, differential_evolution, least_squares
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = HERE.parent / 'calibration2'
ZMIN = 1e-6
BOUNDS = [(-5., 2.), (-5., 3.), (-5., 3.), (0., .99), (.01, 1.)]
SEEDS = [1729, 2718]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def manifest():
    paths = [Path(__file__), HERE/'PLAN.md', OLD/'resistance_rows.json',
             OLD/'time_segments.json']
    return {'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths}, 'numpy': np.__version__, 'scipy': scipy.__version__}


def data():
    rows = [r for r in json.loads((OLD/'resistance_rows.json').read_text())
            if r['sheet'] == 'Fig 4d']
    times = json.loads((OLD/'time_segments.json').read_text())
    rstar = min(r['resistance_ohm_cm2'] for r in rows if r['stage'] == 'Ch')
    protocols = []
    for cathode, sheet in [('P-LCO', 'Fig 4b'), ('N-LCO', 'Fig 4a')]:
        for rate in [.2, .5, 2., 4.]:
            rr = [r for r in rows if r['workbook_cathode'] == cathode
                  and r['rate_c_label'] == rate]
            ss = [r for r in times if r['sheet'] == sheet and r['group'] == f'{rate:g}C']
            assert len(rr) == 3
            charged = next(r for r in rr if r['stage'] == 'Ch')
            targets = [next(r for r in rr if r['stage'] == stage)
                       for stage in ['dCh1', 'dCh2']]
            excluded = cathode == 'N-LCO' and rate == .5
            assert len(ss) == (4 if excluded else 3)
            row = {'cathode': cathode, 'rate_c': rate, 'J': 1.2*rate,
                   'R_ch': charged['resistance_ohm_cm2'],
                   'charged_cell': charged['value_cell'], 'source_sheet': sheet,
                   'segments': ss, 'excluded_ambiguous': excluded,
                   'targets': [{'cell': r['value_cell'], 'stage': r['stage'],
                                'ratio': r['normalized_to_same_cycle_charge']}
                               for r in targets],
                   'subset': 'excluded' if excluded else
                             ('train' if rate in [.2, 2.] else 'check')}
            if not excluded:
                row.update(t1=ss[1]['duration_h'], t2=ss[2]['duration_h'],
                           rest=ss[2]['start_h']-ss[1]['end_h'])
                assert row['rest'] >= 0
            protocols.append(row)
    return rstar, protocols


def reference(z0, duration, h, ceiling, drive, nu):
    """Independent direct Radau integration with a numerical guard event."""
    if duration == 0:
        return z0, False
    def rhs(t, y):
        return [h*(ceiling-y[0])-drive/max(y[0], ZMIN/4)**nu]
    def stop(t, y):
        return y[0]-ZMIN
    stop.terminal = True
    stop.direction = -1
    sol = solve_ivp(rhs, (0., duration), [z0], method='Radau',
                    rtol=2e-10, atol=2e-12, events=stop)
    if not sol.success:
        raise RuntimeError(sol.message)
    return float(sol.y[0, -1]), bool(len(sol.t_events[0]))


def advance(z0, duration, h, ceiling, drive, nu):
    """Exact constant-input flow, with scalar root finding for nu=1."""
    if z0 <= ZMIN:
        return ZMIN, True
    if duration == 0:
        return z0, False
    if drive == 0:
        return z0+(ceiling-z0)*(-math.expm1(-h*duration)), False
    if h == 0:
        power = z0**(nu+1)-(nu+1)*drive*duration
        if power <= ZMIN**(nu+1):
            return ZMIN, True
        return power**(1/(nu+1)), False
    if nu == 0:
        z = z0+(ceiling-z0-drive/h)*(-math.expm1(-h*duration))
        return max(ZMIN, z), z <= ZMIN
    c = drive/h
    discriminant = ceiling*ceiling-4*c
    if abs(discriminant) < 1e-11*ceiling*ceiling:
        # Coalescing roots make the primitive ill conditioned. This fallback is
        # uncommon in fitting and is separately identified in the pilot fixtures.
        return reference(z0, duration, h, ceiling, drive, nu)
    f0 = h*(ceiling-z0)-drive/z0
    if f0 == 0:
        return z0, False
    if discriminant > 0:
        gap = math.sqrt(discriminant)
        hi = .5*(ceiling+gap)
        lo = c/hi
        def elapsed(z):
            if z == z0:
                return 0.
            upper = math.log(abs(z-hi))-math.log(abs(z0-hi))
            lower = math.log(abs(z-lo))-math.log(abs(z0-lo))
            return (-hi*upper+lo*lower)/(gap*h)
        endpoint = hi if z0 > lo else ZMIN
        if z0 == lo or z0 == hi:
            return z0, False
    else:
        gap = math.sqrt(-discriminant)
        d0 = (z0-ceiling/2)**2-discriminant/4
        angle0 = math.atan2(2*z0-ceiling, gap)
        def elapsed(z):
            if z == z0:
                return 0.
            dd = (z-ceiling/2)**2-discriminant/4
            return (-.5*math.log(dd/d0)-ceiling/gap*
                    (math.atan2(2*z-ceiling, gap)-angle0))/h
        endpoint = ZMIN
    disconnected = endpoint == ZMIN
    inner = endpoint if disconnected else float(np.nextafter(endpoint, z0))
    if inner == z0:
        return endpoint, disconnected
    try:
        end_time = elapsed(inner)
    except (ValueError, ZeroDivisionError):
        end_time = math.inf
    if duration >= end_time:
        return endpoint, disconnected
    low, high = sorted((z0, inner))
    z = brentq(lambda zz: elapsed(zz)-duration, low, high,
               xtol=1e-12, rtol=1e-13, maxiter=150)
    return z, False


def parameters(x, nu, rstar):
    alpha, hp, hn = (10**float(t) for t in x[:3])
    rb, astar = float(x[3])*rstar, float(x[4])
    K = astar*(rstar-rb)
    return {'alpha': alpha, 'h_P_per_hour': hp, 'h_N_per_hour': hn,
            'R_background_ohm_cm2': rb, 'reference_contact_fraction': astar,
            'K_ohm_cm2': K, 'k_per_mah_cm2': alpha*astar**(nu+1)}


def predictions(x, nu, timing, rstar, protocols, independent=False):
    par = parameters(x, nu, rstar)
    rb, alpha = par['R_background_ohm_cm2'], par['alpha']
    ceiling = 1/par['reference_contact_fraction']
    method = reference if independent else advance
    out = []
    for p in protocols:
        if p['excluded_ambiguous']:
            continue
        h = par['h_P_per_hour' if p['cathode'] == 'P-LCO' else 'h_N_per_hour']
        z0 = (rstar-rb)/(p['R_ch']-rb)
        z1, stop1 = method(z0, p['t1'], h, ceiling, alpha*p['J'], nu)
        if stop1:
            zrest, stoprest = ZMIN, True
        else:
            zrest, stoprest = method(z1, p['rest'], h, ceiling, 0., nu)
        if stoprest:
            z2, stop2 = ZMIN, True
        else:
            z2, stop2 = method(zrest, p['t2'], h, ceiling, alpha*p['J'], nu)
        for target, z, stopped in zip(p['targets'],
                                     [z1 if timing == 'early' else zrest, z2],
                                     [stop1, stop2]):
            ratio = None if stopped else (rb+(rstar-rb)/z)/p['R_ch']
            out.append({'cathode': p['cathode'], 'rate_c': p['rate_c'],
                        'source_cell': target['cell'], 'stage': target['stage'],
                        'subset': p['subset'], 'observed_ratio': target['ratio'],
                        'predicted_ratio': ratio, 'disconnected': stopped,
                        'error': None if ratio is None else ratio-target['ratio'],
                        'contact_fraction': z/ceiling,
                        'initial_contact_fraction': z0/ceiling})
    return out


def errors(rows):
    return np.array([100. if r['error'] is None else r['error'] for r in rows])


def fit(task):
    nu, timing, seed = task
    tick = time.perf_counter()
    rstar, protocols = data()
    train = [p for p in protocols if p['subset'] == 'train']
    evaluations = 0
    disconnections = 0
    def residual(x):
        nonlocal evaluations, disconnections
        evaluations += 1
        rows = predictions(x, nu, timing, rstar, train)
        disconnections += int(any(r['disconnected'] for r in rows))
        if evaluations % 250 == 0:
            checkpoint_guard()
        return errors(rows)
    result = differential_evolution(lambda x: float(np.sum(residual(x)**2)),
                                    BOUNDS, seed=seed, popsize=8, maxiter=100,
                                    tol=1e-7, polish=False, workers=1)
    refined = least_squares(residual, result.x, bounds=np.array(BOUNDS).T,
                            max_nfev=600, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    candidates = [result.x, refined.x]
    x = min(candidates, key=lambda xx: float(np.sum(residual(xx)**2)))
    rows = predictions(x, nu, timing, rstar, protocols)
    direct = predictions(x, nu, timing, rstar, protocols, independent=True)
    agreement = []
    for a, b in zip(rows, direct):
        status = a['disconnected'] == b['disconnected']
        difference = None if a['disconnected'] or b['disconnected'] else abs(a['predicted_ratio']-b['predicted_ratio'])
        agreement.append({'cell': a['source_cell'], 'status_matches': status,
                          'ratio_difference': difference,
                          'pass': status and (difference is None or difference <= 1e-6)})
    subsets = {}
    for label in ['train', 'check']:
        subset = [r for r in rows if r['subset'] == label]
        complete = all(not r['disconnected'] for r in subset)
        residuals = errors(subset)
        subsets[label] = {'n': len(subset), 'complete': complete,
                          'sse': float(np.sum(residuals**2)),
                          'rmse': float(np.sqrt(np.mean(residuals**2))) if complete else None,
                          'max_abs_error': float(max(abs(residuals))) if complete else None,
                          'screen_pass': bool(complete and max(abs(residuals)) <= .03)}
    return {'nu': nu, 'timing': timing, 'seed': seed, 'x': list(map(float, x)),
            'parameters': parameters(x, nu, rstar), 'subsets': subsets, 'points': rows,
            'parameter_near_bound': [i for i, (v, (lo, hi)) in enumerate(zip(x, BOUNDS))
                                     if min(v-lo, hi-v) <= .001*(hi-lo)],
            'global_search': {'success': bool(result.success), 'message': str(result.message),
                              'nfev': int(result.nfev), 'nit': int(result.nit)},
            'refinement': {'success': bool(refined.success), 'message': str(refined.message),
                           'nfev': int(refined.nfev), 'optimality': float(refined.optimality)},
            'prediction_screen_pass': all(v['screen_pass'] for v in subsets.values()),
            'independent_ode_checks': agreement,
            'numerical_checks_pass': all(r['pass'] for r in agreement),
            'evaluations': evaluations, 'evaluations_with_disconnection': disconnections,
            'seconds': time.perf_counter()-tick}


def pilot():
    rng = np.random.default_rng(65017)
    fixtures = []
    for nu in [0, 1]:
        for z0, duration, h, ceiling, drive in [
            (.8, .5, 0., 1., .1), (.8, 5., 0., 1., 1.),
            (.8, .5, 2., 1., 0.), (.8, .1, 1., 1., .25),
            (.4, .1, 1., 1., .25), (.8, 4., 1000., 1., .2),
            (.8, 4., .00001, 100., .3), (.8, .1, 1., 1., .16),
            (.2, .1, 1., 1., .16)]:
            fixtures.append((nu, z0, duration, h, ceiling, drive))
        for _ in range(32):
            fixtures.append((nu, float(rng.uniform(.2, 1.)),
                             10**float(rng.uniform(-3, .7)),
                             10**float(rng.uniform(-4, 2)),
                             10**float(rng.uniform(0, 2)),
                             10**float(rng.uniform(-4, 1))))
    checks = []
    for nu, z0, duration, h, ceiling, drive in fixtures:
        computed, stopped = advance(z0, duration, h, ceiling, drive, nu)
        expected, refstopped = reference(z0, duration, h, ceiling, drive, nu)
        err = abs(computed-expected)
        checks.append({'nu': nu, 'z0': z0, 'duration': duration, 'h': h,
                       'ceiling': ceiling, 'drive': drive, 'value': computed,
                       'reference': expected, 'absolute_error': err,
                       'disconnected': stopped, 'reference_disconnected': refstopped,
                       'pass': bool(stopped == refstopped and err <= 1e-7*max(1, abs(expected)))})
        checkpoint_guard()
    rstar, protocols = data()
    x = [-1., -.5, .5, .4, .8]
    tick = time.perf_counter()
    for _ in range(100):
        for nu in [0, 1]:
            predictions(x, nu, 'early', rstar, protocols)
    elapsed = time.perf_counter()-tick
    result = {'checks': checks, 'all_pass': all(r['pass'] for r in checks),
              'timing': {'full_evaluations': 200, 'seconds': elapsed},
              'rstar': rstar, 'protocols': protocols,
              'spatial_field_solves': 0}
    save(HERE/'pilot.json', result)
    print(json.dumps({'all_pass': result['all_pass'], 'checks': len(checks),
                      'timing_seconds': elapsed}), flush=True)
    assert result['all_pass']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    current = manifest()
    if args.pilot:
        assert not (HERE/'pilot.json').exists(), 'Preserve the previous attempt.'
        save(HERE/'manifest.json', current)
        pilot()
        return
    assert json.loads((HERE/'manifest.json').read_text()) == current
    assert json.loads((HERE/'pilot.json').read_text())['all_pass']
    output = HERE/'fits'
    output.mkdir(exist_ok=True)
    tasks = [(nu, timing, seed) for nu in [0, 1] for timing in ['early', 'late']
             for seed in SEEDS]
    completed = []
    pending = []
    for task in tasks:
        name = f'nu{task[0]}_{task[1]}_{task[2]}.json'
        if (output/name).exists():
            assert args.resume
            completed.append(json.loads((output/name).read_text()))
        else:
            pending.append(task)
    tick = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(os.environ.get('AJ_COMPUTE_WORKERS', '1'))) as pool:
        futures = {pool.submit(fit, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            save(output/f'nu{task[0]}_{task[1]}_{task[2]}.json', result)
            completed.append(result)
            checkpoint_guard()
            print(json.dumps({'completed': len(completed), 'nu': task[0],
                              'timing': task[1], 'seed': task[2],
                              'train_rmse': result['subsets']['train']['rmse'],
                              'check_rmse': result['subsets']['check']['rmse']}), flush=True)
    selected = []
    for nu in [0, 1]:
        for timing in ['early', 'late']:
            rows = [r for r in completed if r['nu'] == nu and r['timing'] == timing]
            selected.append(min(rows, key=lambda r: r['subsets']['train']['sse']))
    result = {'fit_tasks': len(completed), 'selection': 'training SSE only',
              'selected': selected,
              'passing_selected': sum(r['prediction_screen_pass'] for r in selected),
              'numerical_checks_pass': all(r['numerical_checks_pass'] for r in completed),
              'new_tasks': len(pending), 'wall_seconds': time.perf_counter()-tick,
              'experimental_prediction_adopted': False,
              'original_central_hypothesis_retained_for_testing': True}
    save(HERE/'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'selected'}), flush=True)


if __name__ == '__main__':
    main()
