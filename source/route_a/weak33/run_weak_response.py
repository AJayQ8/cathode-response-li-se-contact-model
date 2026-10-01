"""Finite-time weak spatial response of the frozen conserving contact law."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


spatial = module('frozen_patch_law', HERE.parent/'spatial32/run_spatial_probe.py')
projection, model = spatial.projection, spatial.model
electrical = spatial.electrical
read, save, sha = spatial.read, spatial.save, spatial.sha
CATHODES = ['P-LCO', 'N-LCO']
STAGES = ['end_discharge', 'after_30_min_rest']
VARIANTS = [dict(fraction=f, mode=k, grid=n) for f in [.1, .5, 1.] for k in [1, 4] for n in [16, 32, 'continuum']]


def identifier(t):
    if t['kind'] == 'linear':
        return 'linear_'+t['witness_id']
    return (t['kind']+'_'+t['witness_id']+'_'+t['cathode']+'_c'+str(t['rate_c'])+
            '_f'+str(t['fraction'])+'_n'+str(t['n'])+'_e'+str(t['amplitude'])).replace('.', 'p')


def freeze():
    prior = read(HERE.parent/'decomposition29/manifest.json')
    hashes = read(HERE.parent/'spatial32/manifest.json')['sha256'].copy()
    witnesses = prior['witnesses']
    assert len(witnesses) == 50
    paths = [Path(__file__).resolve(), HERE/'PLAN.md', HERE.parent/'spatial32/run_spatial_probe.py',
             HERE.parent/'spatial32/manifest.json', HERE.parent/'spatial32/summary.json',
             HERE.parent/'spatial32/GEOMETRY_DIAGNOSIS.md', HERE.parent/'spatial32/geometry_diagnosis.json',
             HERE.parent/'decomposition29/manifest.json', HERE.parent/'decomposition29/summary.json']
    paths += [HERE.parent/'decomposition29/cases'/(w['id']+'.json') for w in witnesses]
    for p in paths:
        hashes[str(p.relative_to(ROOT))] = sha(p)
    for name, digest in hashes.items():
        assert sha(ROOT/name) == digest, name
    tasks = [dict(kind='linear', witness_id=w['id']) for w in witnesses]
    ordinary = next(w for w in witnesses if w['id'] == 'w026')
    assert ordinary['setting'] == dict(extraction='ordinary', registration='balance_aligned', timing='late')
    for cathode in CATHODES:
        for fraction in [0., 1.]:
            for n in [16, 32]:
                for epsilon in [.02, .01]:
                    tasks.append(dict(kind='future', witness_id='w026', cathode=cathode, rate_c=2.,
                                      fraction=fraction, n=n, amplitude=epsilon))
    _, protocols = model.data('ordinary', 'balance_aligned')
    for p in protocols:
        if p['excluded_ambiguous']:
            continue
        for fraction in [0., 1.]:
            tasks.append(dict(kind='known', witness_id='w026', cathode=p['cathode'], rate_c=p['rate_c'],
                              fraction=fraction, n=32, amplitude=.02))
    assert len(tasks) == 80 and len(set(map(identifier, tasks))) == 80
    frozen = dict(sha256=hashes, witnesses=witnesses, variants=VARIANTS, tasks=tasks,
                  representative_ids=read(HERE.parent/'bridge31/manifest_v2.json')['representative_ids'],
                  fitted_parameters_changed=False, central_claim_changed=False,
                  all_known_data_retrospective=True, prospective_prediction_qualified=False)
    path = HERE/'manifest.json'
    if path.exists():
        assert read(path) == frozen
    else:
        save(path, frozen)
    return frozen


@lru_cache(None)
def impedance(grid, mode):
    if grid == 'continuum':
        k = 2*np.pi*mode
        return float(1/(k*np.tanh(k)))
    v = np.cos(2*np.pi*mode*(np.arange(grid)+.5)/grid)
    z = electrical.boundary_operator(grid)@v
    value = float(np.dot(v, z)/np.dot(v, v))
    assert value > 0 and np.max(np.abs(z-value*v)) <= 1e-12
    return value


def gain(log_value):
    return math.exp(log_value) if -700 <= log_value <= 700 else None


def coefficients(x, a, pressure, current, variants=VARIANTS):
    L, H = 10**np.asarray(x[:2]); _, b, A = x[2:]
    gamma = H*A**(model.M+1)*(pressure/2)**model.M
    h = (1-a)*a**(-model.M)
    hp = -a**(-model.M)-model.M*(1-a)*a**(-model.M-1)
    hpp = model.M*a**(-model.M-2)*((model.M+1)-(model.M-1)*a)
    bz = np.asarray([v['fraction']*b/(A*(1-b))*impedance(v['grid'], v['mode']) for v in variants])
    feedback = model.U*current/L*bz/(a*(1+bz*a))
    feedback_derivative = -model.U*current/L*bz*(1+2*bz*a)/(a*a*(1+bz*a)**2)
    return gamma, h, hp, hpp, feedback, feedback_derivative


class Augmented:
    def __init__(self, x, cathode, q0, current, coordinate):
        self.x = np.asarray(x); self.L, self.H = 10**self.x[:2]
        self.eta, self.b, self.A = self.x[2:]
        self.cathode, self.q0, self.current, self.coordinate = cathode, q0, current, coordinate
        self.tt, self.pp, self.t0, self.p0, _ = projection.original_wave(cathode)
        self.evaluations = 0

    def calculate(self, t, y, jacobian=False):
        self.evaluations += 1
        if self.evaluations % 2500 == 0:
            checkpoint_guard()
        raw = self.A*np.sqrt(max(0., y[0])) if self.coordinate == 'w' else 1-y[0]/self.L
        a = float(np.clip(raw, self.A*model.ZMIN/4, 1.))
        q = self.q0+self.current*t
        pressure = (2+self.eta*model.pressure_increment(self.cathode, q) if self.coordinate == 'w' else
                    2+self.eta*(float(np.interp(self.t0+q/.12, self.tt, self.pp))-self.p0)/1000)
        assert pressure > 0
        gamma, h, hp, hpp, feedback, feedback_derivative = coefficients(self.x, a, pressure, self.current)
        F = gamma*h-model.U*self.current/(self.L*a)
        Fp = gamma*hp+model.U*self.current/(self.L*a*a)
        first = 2*a*F/(self.A*self.A) if self.coordinate == 'w' else -self.L*F
        K = self.L*a*gamma*h
        if not jacobian:
            return np.r_[first, self.current, K, gamma*hp, feedback]
        da = (self.A*self.A/(2*a) if self.coordinate == 'w' else -1/self.L) if self.A*model.ZMIN/4 < raw < 1 else 0.
        top = 2*(F+a*Fp)/(self.A*self.A) if self.coordinate == 'w' else -self.L*Fp
        column = np.r_[top, 0., self.L*gamma*(h+a*hp), gamma*hpp, feedback_derivative]*da
        out = np.zeros((len(y), len(y))); out[:, 0] = column
        return out

    def rhs(self, t, y):
        return self.calculate(t, y)

    def jac(self, t, y):
        return self.calculate(t, y, True)


def scalar_reference(w, initial, history):
    if initial == history:
        return next(h for h in w['diagonal_histories'] if h['cathode'] == initial)
    source = read(HERE.parent/'decomposition29/cases'/(w['id']+'.json'))
    assert source['witness'] == w
    return next(h for h in source['crossed_histories'] if h['initial_cathode'] == initial and h['history_cathode'] == history)


def linear_history(w, initial, history):
    x = np.array(w['x']); L, H = 10**x[:2]; eta, b, A = x[2:]
    rstar, protocols = model.data(w['setting']['extraction'], w['setting']['registration'])
    first = next(p for p in protocols if p['cathode'] == initial and p['rate_c'] == 2.)
    load = next(p for p in protocols if p['cathode'] == history and p['rate_c'] == 2.)
    p = dict(load, J=projection.J, t1=projection.Q/projection.J, t2=0., rest=projection.REST)
    assert model.domain(p, eta)['valid']
    _, _, a0 = model.initial_state(x, rstar, first['R_ch'])
    assert 0 < a0 <= 1
    routes = {}
    for coordinate in ['w', 'g']:
        state = np.r_[(a0/A)**2 if coordinate == 'w' else L*(1-a0), np.zeros(3+len(VARIANTS))]
        stopped = False; stripped = 0.; qstart = p['q0']; max_projection = 0.; rows = []; calls = 0
        for label, duration, current in zip(STAGES, [p['t1'], p['rest']], [p['J'], 0.]):
            system = Augmented(x, history, qstart, current, coordinate)
            def event(t, y):
                return y[0]-model.ZMIN**2 if coordinate == 'w' else L*(1-A*model.ZMIN)-y[0]
            event.terminal = True; event.direction = -1
            elapsed = 0.
            if not stopped:
                sol = solve_ivp(system.rhs, (0., duration), state, jac=system.jac, method='Radau',
                                rtol=1e-9 if coordinate == 'w' else 1e-11, atol=1e-11 if coordinate == 'w' else 1e-13,
                                first_step=min(duration, 1e-4), events=event)
                assert sol.success and np.all(np.isfinite(sol.y[:, -1])), sol.message
                state = sol.y[:, -1]; stopped = bool(len(sol.t_events[0])); elapsed = float(sol.t[-1])
            correction = max(0., float(state[0]-1/(A*A))) if coordinate == 'w' else max(0., float(-state[0]))*2/(L*A*A)
            assert correction <= 1e-7*max(1., 1/(A*A))
            max_projection = max(max_projection, correction)
            state[0] = np.clip(state[0], model.ZMIN**2, 1/(A*A)) if coordinate == 'w' else np.clip(state[0], 0., L*(1-A*model.ZMIN))
            a = A*np.sqrt(state[0]) if coordinate == 'w' else 1-state[0]/L
            stripped += current*elapsed; qstart += current*elapsed; calls += system.evaluations
            V, V0 = L*a*a/2, L*a0*a0/2
            ledger = abs(V-V0+model.U*state[1]-state[2])
            physical = bool(not stopped and 0 < a <= 1+1e-12 and L*a <= 40+1e-10 and V+model.U*state[1] <= 40+model.U*1e-7 and state[2] >= -1e-7)
            numerical = bool(ledger <= 1e-6*max(1., V, abs(state[2])) and abs(state[1]-stripped) <= 1e-6)
            ratio = None if stopped else float((b*rstar+A*(1-b)*rstar/a)/first['R_ch'])
            rows.append(dict(stage=label, state=state.tolist(), a=float(a), ratio=ratio, stopped=stopped,
                             stripped_charge=float(state[1]), nominal_charge=stripped, volume=float(V), initial_volume=float(V0),
                             replenishment=float(state[2]), ledger=float(ledger), projection=max_projection,
                             log_gain_f0=float(state[3]), log_enhancement=state[4:].tolist(),
                             physical_pass=physical, numerical_pass=numerical))
            checkpoint_guard()
        routes[coordinate] = dict(rows=rows, evaluations=calls)
    reference = scalar_reference(w, initial, history)
    rows = []
    for fast, direct, ref in zip(routes['w']['rows'], routes['g']['rows'], reference['rows']):
        same = fast['stopped'] == direct['stopped'] == ref['stopped']
        re = None if fast['ratio'] is None or direct['ratio'] is None else abs(fast['ratio']-direct['ratio'])
        old = None if direct['ratio'] is None or ref['ratio'] is None else abs(direct['ratio']-ref['ratio'])
        logs = max(abs(fast['log_gain_f0']-direct['log_gain_f0']), max(abs(a-b) for a, b in zip(fast['log_enhancement'], direct['log_enhancement'])))
        numerical = same and (re is None or re <= 1e-4) and (old is None or old <= 1e-4) and logs <= 1e-4
        numerical = numerical and fast['numerical_pass'] and direct['numerical_pass']
        rows.append(dict(stage=fast['stage'], primary=fast, independent=direct, coordinate_ratio_difference=re,
                         reference_ratio_difference=old, maximum_log_difference=logs,
                         numerical_pass=bool(numerical), physical_pass=fast['physical_pass'] and direct['physical_pass']))
    return dict(initial_cathode=initial, history_cathode=history, R_ch=first['R_ch'], q0=p['q0'], initial_a=a0,
                rows=rows, available=all(r['numerical_pass'] and r['physical_pass'] for r in rows),
                evaluations={k: v['evaluations'] for k, v in routes.items()})


def finite_history(w, task):
    x = np.array(w['x']); L = 10**x[0]; b, A = x[3:]
    rstar, protocols = model.data(w['setting']['extraction'], w['setting']['registration'])
    source = next(p for p in protocols if p['cathode'] == task['cathode'] and p['rate_c'] == task['rate_c'])
    p = dict(source)
    new = task['kind'] == 'future'
    if new:
        p.update(J=projection.J, t1=projection.Q/projection.J, t2=0., rest=projection.REST)
    assert model.domain(p, x[2])['valid'] and not p['excluded_ambiguous']
    n = task['n']; cosine = np.cos(2*np.pi*(np.arange(n)+.5)/n)
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
                       mean_contact=float(np.mean(a)), second_harmonic=float(2*np.mean(a*np.cos(4*np.pi*(np.arange(n)+.5)/n))))
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


def controls(frozen):
    path = HERE/'controls.json'
    if path.exists():
        old = read(path); assert old['passed'] and old['manifest_sha256'] == sha(HERE/'manifest.json')
        return
    tests = []
    for wid in frozen['representative_ids']:
        w = next(w for w in frozen['witnesses'] if w['id'] == wid)
        x = np.array(w['x']); L = 10**x[0]; A = x[4]
        rstar, protocols = model.data(w['setting']['extraction'], w['setting']['registration'])
        for cathode in CATHODES:
            p = next(p for p in protocols if p['cathode'] == cathode and p['rate_c'] == 2.)
            _, _, a = model.initial_state(x, rstar, p['R_ch'])
            pressure = 2+x[2]*model.pressure_increment(cathode, p['q0'])
            for n in [16, 32]:
                for fraction in [0., .1, .5, 1.]:
                    system = spatial.System(x, rstar, fraction, n, cathode, p['q0'], projection.J, 'g')
                    y = np.r_[np.full(n, L*(1-a)), np.zeros(2*n)]
                    jac = system.jac(0., y)[:n, :n]
                    for mode in [1, 4]:
                        v = np.cos(2*np.pi*mode*(np.arange(n)+.5)/n)
                        gamma, _, hp, _, feedback, _ = coefficients(x, a, pressure, projection.J, [dict(fraction=fraction, mode=mode, grid=n)])
                        expected = float(gamma*hp+feedback[0])
                        action = jac@v
                        actual = float(np.dot(v, action)/np.dot(v, v))
                        step = 1e-5*min(a, 1-a)
                        d = np.r_[-L*step*v, np.zeros(2*n)]
                        fd = -(system.rhs(0., y+d)[:n]-system.rhs(0., y-d)[:n])/(2*L*step)
                        finite = float(np.dot(v, fd)/np.dot(v, v))
                        error = max(abs(actual-expected), abs(finite-expected))/max(1., abs(expected))
                        leakage = float(np.linalg.norm(action-expected*v)/max(1., np.linalg.norm(action)))
                        mean_expected = gamma*hp+model.U*projection.J/(L*a*a)
                        mean_error = float(np.max(np.abs(jac@np.ones(n)-mean_expected))/max(1., abs(mean_expected)))
                        rest = coefficients(x, a, pressure, 0., [dict(fraction=fraction, mode=mode, grid=n)])[4][0]
                        tests.append(dict(witness_id=wid, cathode=cathode, n=n, fraction=fraction, mode=mode,
                                          formula=expected, spatial_jacobian=actual, finite_difference=finite,
                                          scaled_error=error, leakage=leakage, mean_mode_error=mean_error,
                                          zero_current_feedback=float(rest), passed=bool(error <= 1e-5 and leakage <= 1e-9 and mean_error <= 1e-9 and rest == 0.)))
            checkpoint_guard()
    w = next(w for w in frozen['witnesses'] if w['id'] == 'w026')
    x = np.array(w['x']); L = 10**x[0]; A = x[4]
    _, protocols = model.data('ordinary', 'balance_aligned')
    p = next(p for p in protocols if p['cathode'] == 'P-LCO' and p['rate_c'] == 2.)
    _, _, a0 = model.initial_state(x, w['rstar'], p['R_ch'])
    augmented = []
    for coordinate in ['w', 'g']:
        system = Augmented(x, 'P-LCO', p['q0'], projection.J, coordinate)
        y = np.r_[(a0/A)**2 if coordinate == 'w' else L*(1-a0), np.zeros(3+len(VARIANTS))]
        step = 1e-5*max(1., abs(y[0])); d = np.zeros(len(y)); d[0] = step
        fd = (system.rhs(0., y+d)-system.rhs(0., y-d))/(2*step)
        jac = system.jac(0., y)
        error = float(np.max(np.abs(fd-jac[:, 0]))/max(1., np.max(np.abs(fd))))
        assert np.all(jac[:, 1:] == 0.)
        augmented.append(dict(coordinate=coordinate, scaled_error=error, passed=error <= .001))
    output = dict(tests=tests, augmented=augmented, passed=all(t['passed'] for t in tests+augmented),
                  manifest_sha256=sha(HERE/'manifest.json'))
    save(path, output)
    assert output['passed'], 'Weak-response derivative controls failed'


def run(task, witness):
    checkpoint_guard(); tic = time.perf_counter()
    path = HERE/'cases'/(identifier(task)+'.json')
    if path.exists():
        old = read(path)
        assert old['task'] == task and old['witness'] == witness and old['manifest_sha256'] == sha(HERE/'manifest.json')
        return old
    if task['kind'] == 'linear':
        result = dict(histories=[linear_history(witness, i, h) for i in CATHODES for h in CATHODES])
        available = all(h['available'] for h in result['histories'])
    else:
        result = finite_history(witness, task)
        available = result['available']
    out = dict(task=task, witness=witness, result=result, available=available, wall_seconds=time.perf_counter()-tic,
               manifest_sha256=sha(HERE/'manifest.json'))
    save(path, out)
    return out


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['pilot', 'all'], required=True)
    args = parser.parse_args(); checkpoint_guard(); tic = time.perf_counter()
    frozen = freeze(); controls(frozen)
    def pilot(t):
        return t['witness_id'] == 'w026' and (t['kind'] == 'linear' or (t['kind'] == 'future' and t['cathode'] == 'P-LCO' and t['n'] == 16))
    tasks = [t for t in frozen['tasks'] if args.phase == 'all' or pilot(t)]
    workers = int(os.environ.get('AJ_COMPUTE_WORKERS', '1')); done = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, t, next(w for w in frozen['witnesses'] if w['id'] == t['witness_id'])) for t in tasks]
        for future in as_completed(futures):
            item = future.result(); name = identifier(item['task']); done.append(name)
            print(json.dumps(dict(id=name, available=item['available'], seconds=item['wall_seconds'])), flush=True)
            checkpoint_guard()
    save(HERE/(args.phase+'_execution.json'), dict(completed=sorted(done), workers=workers,
         wall_seconds=time.perf_counter()-tic, manifest_sha256=sha(HERE/'manifest.json')))


if __name__ == '__main__':
    main()
