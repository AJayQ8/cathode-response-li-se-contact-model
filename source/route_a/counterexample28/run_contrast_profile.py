"""Bounded all-known-data contrast profiles for the fixed new 1 C protocol."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import os
import time

import numpy as np
from scipy.optimize import minimize
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location('projection', HERE.parent/'prospective27/run_projection.py')
projection = importlib.util.module_from_spec(spec); spec.loader.exec_module(projection)
model, assessment = projection.model, projection.assessment
read, save, sha = model.read, model.save, model.sha
WIDTH = model.HIGH-model.LOW
LIMIT = .03-2e-6


def identifier(task):
    return task['extraction']+'_'+task['direction']+'_seed'+str(task['seed'])


def freeze():
    previous = read(HERE.parent/'prospective27/manifest.json')
    pools = {}
    for extraction in ['published', 'ordinary', 'robust']:
        pool = [w for w in previous['witnesses'] if w['setting'] == dict(registration='balance_aligned', timing='late', extraction=extraction)]
        first = min(pool, key=lambda w:sum(s['sse'] for s in w['original_subsets'].values()))
        second = max(pool, key=lambda w:float(np.linalg.norm((np.array(w['x'])-first['x'])/WIDTH)))
        seeds = [first] if first['x'] == second['x'] else [first, second]
        pools[extraction] = dict(pool=pool, seeds=seeds)
    tasks = [dict(registration='balance_aligned', timing='late', extraction=e, direction=d, seed=i)
             for e, p in pools.items() for d in ['min', 'max'] for i in range(len(p['seeds']))]
    assert len(tasks) == 10
    hashes = previous['sha256'].copy()
    for path in [Path(__file__).resolve(), HERE/'PLAN.md', HERE.parent/'prospective27/manifest.json',
                 HERE.parent/'prospective27/summary.json', HERE.parent/'uncertainty25/controls.json']:
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for path, digest in hashes.items():
        assert sha(ROOT/path) == digest, path
    assert read(HERE.parent/'uncertainty25/controls.json')['passed']
    frozen = dict(sha256=hashes, pools=pools, tasks=tasks, maxiter=30, ftol=1e-9,
                  optimization_limit=LIMIT, known_targets_used=14, new_observed_targets_exist=False,
                  all_known_targets_used_in_constraints=True, central_claim_changed=False)
    path = HERE/'manifest.json'
    if path.exists():
        assert read(path) == frozen
    else:
        save(path, frozen)
    return frozen


def contrast(x, task, independent=False):
    rstar, protocols = model.data(task['extraction'], task['registration'])
    values = []; gradients = []; valid = True
    for cathode in ['P-LCO', 'N-LCO']:
        p = next(p for p in protocols if p['cathode'] == cathode and p['rate_c'] == 2.)
        probe = dict(p, J=projection.J, t1=projection.Q/projection.J, t2=0., rest=projection.REST)
        initial, _, _ = model.initial_state(x, rstar, p['R_ch'])
        if initial is None or not model.domain(probe, x[2])['valid']:
            return None, np.zeros(5), False
        advance = assessment.advance if independent else model.advance
        state, stopped, elapsed, correction = advance(initial, projection.Q/projection.J, x, cathode,
                                                       p['q0'], projection.J, independent)
        if stopped:
            return None, np.zeros(5), False
        z = float(np.sqrt(state[0])); a = x[4]*z; L = 10**x[0]; rb = x[3]*rstar; n = rstar-rb
        value = (rb+n/z)/p['R_ch']
        derivative = -n/(2*p['R_ch']*z**3)*state[1:]
        derivative[3] += (rstar-rstar/z)/p['R_ch']
        D = model.U/(L*x[4]**2)
        valid = valid and 0 < a <= 1 and L*a <= 40+1e-10 and L*a*a/2/model.U+projection.Q <= 40/model.U+1e-7
        valid = valid and state[0]-initial[0]+2*D*projection.Q >= -1e-7 and correction <= 1e-7*max(1., 1/x[4]**2)
        values.append(value); gradients.append(derivative)
    return values[0]-values[1], gradients[0]-gradients[1], bool(valid)


class Problem:
    def __init__(self, task):
        self.task = task
        self.rstar, self.protocols = model.data(task['extraction'], task['registration'])
        self.sign = 1. if task['direction'] == 'min' else -1.
        self.last_y = None; self.evaluations = 0; self.failures = {}; self.best = None; self.trace = []
        self.started = time.perf_counter()

    def calculate(self, y):
        if self.last_y is not None and np.array_equal(y, self.last_y):
            return
        self.evaluations += 1
        if self.evaluations % 10 == 0:
            checkpoint_guard()
        x = model.LOW+WIDTH*np.asarray(y)
        progress = dict(task=self.task, evaluations=self.evaluations, x=x.tolist(), evaluation_complete=False,
                        elapsed_seconds=time.perf_counter()-self.started)
        save(HERE/'progress'/(identifier(self.task)+'.json'), progress)
        try:
            rows = model.predictions(x, self.task, self.rstar, self.protocols)
            assert len(rows) == 14
            residual = np.array([100. if model.base.unavailable(r) else r['error'] for r in rows])
            jac = np.array([r['derivative'] for r in rows])*WIDTH[None, :]
            known_valid = all(model.physical(r) and not model.base.unavailable(r) for r in rows)
            value, derivative, target_valid = contrast(x, self.task)
            objective = self.sign*value if target_valid else 1e6
            gradient = self.sign*derivative*WIDTH if target_valid else np.zeros(5)
            if not all(np.all(np.isfinite(v)) for v in [residual, jac, objective, gradient]):
                raise FloatingPointError('Nonfinite objective or constraint')
            if known_valid and target_valid and np.max(np.abs(residual)) <= LIMIT:
                if self.best is None or objective < self.best['objective']:
                    self.best = dict(x=x.tolist(), objective=float(objective), max_known_error=float(np.max(np.abs(residual))))
                    self.trace.append(dict(evaluation=self.evaluations, **self.best))
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            key = type(exc).__name__+': '+str(exc)[:180]
            self.failures[key] = self.failures.get(key, 0)+1
            residual = np.full(14, 100.); jac = np.zeros((14, 5)); objective = 1e6; gradient = np.zeros(5)
        self.last_y = np.array(y).copy(); self.r = residual; self.j = jac; self.f = float(objective); self.g = gradient
        progress.update(evaluation_complete=True, best_feasible=self.best,
                        max_constraint_violation=float(max(0., np.max(np.abs(residual))-LIMIT)),
                        elapsed_seconds=time.perf_counter()-self.started)
        save(HERE/'progress'/(identifier(self.task)+'.json'), progress)

    def objective(self, y): self.calculate(y); return self.f
    def gradient(self, y): self.calculate(y); return self.g.copy()
    def constraint(self, y): self.calculate(y); return np.r_[LIMIT-self.r, LIMIT+self.r]
    def constraint_jac(self, y): self.calculate(y); return np.r_[-self.j, self.j]


def controls(frozen):
    path = HERE/'controls.json'
    if path.exists():
        old = read(path)
        assert old['passed'] and old['manifest_sha256'] == sha(HERE/'manifest.json')
        return
    tests = []
    for extraction, pool in frozen['pools'].items():
        x = np.array(pool['seeds'][0]['x']); y = (x-model.LOW)/WIDTH
        task = dict(registration='balance_aligned', timing='late', extraction=extraction)
        value, derivative, valid = contrast(x, task)
        direct, _, direct_valid = contrast(x, task, True)
        assert valid and direct_valid
        errors = []; stencils = []
        def f(yy):
            val, _, good = contrast(model.LOW+WIDTH*yy, task, True)
            assert good
            return val
        for col in range(5):
            h = 1e-5; step = np.eye(5)[col]*h
            if y[col]-h >= 0 and y[col]+h <= 1:
                fd = (f(y+step)-f(y-step))/(2*h); stencil = 'central'
            elif y[col]+2*h <= 1:
                fd = (-3*direct+4*f(y+step)-f(y+2*step))/(2*h); stencil = 'forward_second_order'
            else:
                fd = (3*direct-4*f(y-step)+f(y-2*step))/(2*h); stencil = 'backward_second_order'
            errors.append(float(abs(fd-derivative[col]*WIDTH[col])/max(1., abs(fd))))
            stencils.append(stencil)
        tests.append(dict(extraction=extraction, x=x.tolist(), scaled_errors=errors, stencils=stencils,
                          value_difference=abs(value-direct), passed=bool(max(errors) <= .001 and abs(value-direct) <= 1e-4)))
        checkpoint_guard()
    report = dict(tests=tests, passed=all(t['passed'] for t in tests), manifest_sha256=sha(HERE/'manifest.json'))
    save(path, report)
    assert report['passed'], 'New contrast derivative controls failed'


def search(task, seeds):
    checkpoint_guard(); started = time.perf_counter(); problem = Problem(task)
    x0 = np.array(seeds[task['seed']]['x'])
    opt = minimize(problem.objective, (x0-model.LOW)/WIDTH, jac=problem.gradient,
                   method='SLSQP', bounds=[(0., 1.)]*5,
                   constraints=[dict(type='ineq', fun=problem.constraint, jac=problem.constraint_jac)],
                   options=dict(maxiter=30, ftol=1e-9))
    xopt = model.LOW+WIDTH*opt.x
    optimizer = dict(success=bool(opt.success), status=int(opt.status), message=str(opt.message), nit=int(opt.nit),
                     nfev=int(opt.nfev), njev=int(opt.njev), fun=float(opt.fun), x=xopt.tolist())
    output = dict(id=identifier(task), task=task, seed=seeds[task['seed']], optimizer=optimizer,
                  best_feasible=problem.best, improvement_trace=problem.trace, evaluations=problem.evaluations,
                  evaluation_failures=problem.failures, optimization_seconds=time.perf_counter()-started,
                  manifest_sha256=sha(HERE/'manifest.json'), prospective_prediction_qualified=False)
    save(HERE/'optimizer_returns'/(output['id']+'.json'), output)
    candidates = [('start', x0), ('optimizer', xopt)]
    if problem.best is not None:
        candidates.append(('best_feasible', np.array(problem.best['x'])))
    assessed = []; reused = {}
    known_problem = model.Problem(task)
    for label, x in candidates:
        key = tuple(x)
        if key in reused:
            item = deepcopy(reused[key]); item['label'] = label
        else:
            item = assessment.assess(known_problem, x, label)
            item['all_known_admitted'] = bool(item['numerical_pass'] and item['physical_observation_pass']
                and all(item['subsets'][subset]['screen_pass'] for subset in ['train', 'check']))
            witness = dict(x=x.tolist(), rstar=known_problem.rstar,
                           setting={k:task[k] for k in ['registration', 'timing', 'extraction']})
            new = []
            for cathode in ['P-LCO', 'N-LCO']:
                try:
                    new.append(projection.project(witness, cathode))
                except (RuntimeError, ValueError, FloatingPointError, AssertionError) as exc:
                    new.append(dict(cathode=cathode, available=False, numerical_error=type(exc).__name__+': '+str(exc)))
            item['new_protocol'] = new; reused[key] = deepcopy(item)
        assessed.append(item); checkpoint_guard()
    output.update(rstar=known_problem.rstar, candidates=assessed, wall_seconds=time.perf_counter()-started)
    save(HERE/'searches'/(output['id']+'.json'), output)
    return dict(id=output['id'], wall_seconds=output['wall_seconds'], optimizer_success=optimizer['success'],
                optimizer_status=optimizer['status'], admitted=sum(c['all_known_admitted'] for c in assessed))


def main():
    checkpoint_guard(); started = time.perf_counter(); frozen = freeze(); controls(frozen)
    workers = int(os.environ.get('AJ_COMPUTE_WORKERS', '1')); records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for task in frozen['tasks']:
            path = HERE/'searches'/(identifier(task)+'.json')
            if path.exists():
                fit = read(path)
                assert fit['manifest_sha256'] == sha(HERE/'manifest.json') and fit['task'] == task
                records.append(dict(id=fit['id'], reused_completed_search=True))
            else:
                futures.append(pool.submit(search, task, frozen['pools'][task['extraction']]['seeds']))
        for future in as_completed(futures):
            result = future.result(); records.append(result); print(json.dumps(result), flush=True)
            checkpoint_guard()
    save(HERE/'execution.json', dict(tasks=records, completed=len(records), workers=workers,
                                    wall_seconds=time.perf_counter()-started, manifest_sha256=sha(HERE/'manifest.json')))


if __name__ == '__main__':
    main()
