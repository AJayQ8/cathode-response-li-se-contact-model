"""Calibration-only extrema of the unchanged finite-height history model."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import argparse
import importlib.util
import json
import os
import time

import numpy as np
from scipy.optimize import minimize
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location('bounded_assessment', HERE.parent/'robust23/resume_startup_repair.py')
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)
model = assessment.model
read, save, sha = model.read, model.save, model.sha
WIDTH = model.HIGH-model.LOW
LIMIT = .03-2e-6
QUERIES = ['b', 'eta', 'P05_first', 'P4_first']


def setting_key(task):
    return tuple(task[k] for k in ['registration', 'timing', 'extraction'])


def identifier(task):
    return '_'.join(setting_key(task))+'_'+task['query']+'_'+task['direction']+'_seed'+str(task['seed'])


def settings():
    return [dict(registration=r, timing=t, extraction=e)
            for r in ['local', 'balance_aligned'] for t in ['early', 'late']
            for e in ['published', 'ordinary', 'robust']]


def tasks():
    return [dict(s, query=q, direction=d, seed=i) for s in settings()
            for q in QUERIES for d in ['min', 'max'] for i in range(2)]


def seed_pools():
    files = sorted((HERE.parent/'robust23/fits').glob('*.json'))+sorted((HERE.parent/'history24/fits').glob('*.json'))
    pools = {}
    for s in settings():
        candidates = []
        for path in files:
            f = read(path)
            if (f['task']['timing'], f['task']['extraction']) != (s['timing'], s['extraction']):
                continue
            is_null = path.parent.parent.name == 'history24'
            if not is_null and f['task']['registration'] != s['registration']:
                continue
            for c in f['candidates']:
                if not c['calibration_admissible'] or not c['subsets']['train']['screen_pass']:
                    continue
                if is_null:
                    assert c['x'][2] == 0.
                candidates.append(dict(x=c['x'], training_objective=c['training_objective'],
                                       source=str(path.relative_to(ROOT)), label=c['label']))
        unique = {}
        for c in sorted(candidates, key=lambda v:(v['training_objective'], v['source'], v['label'])):
            unique.setdefault(tuple(c['x']), c)
        pool = list(unique.values())
        assert pool
        first = min(pool, key=lambda c:c['training_objective'])
        second = max(pool, key=lambda c:float(np.linalg.norm((np.array(c['x'])-first['x'])/WIDTH)))
        pools['_'.join(setting_key(s))] = dict(pool=pool, seeds=[first, second],
            seed_scaled_distance=float(np.linalg.norm((np.array(first['x'])-second['x'])/WIDTH)))
    return pools


def manifest():
    hashes = read(HERE.parent/'history24/manifest.json')['sha256'].copy()
    for stage in ['robust23', 'history24']:
        report = read(HERE.parent/stage/'verification.json')
        assert report['passed']
        for name, digest in report['output_sha256'].items():
            path = HERE.parent/stage/name
            assert sha(path) == digest, str(path)
            hashes[str(path.relative_to(ROOT))] = digest
        path = HERE.parent/stage/'verification.json'
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for path in [Path(__file__), HERE/'PLAN.md']:
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for name, digest in hashes.items():
        assert sha(ROOT/name) == digest, name
    return dict(sha256=hashes, low=model.LOW.tolist(), high=model.HIGH.tolist(),
                tasks=tasks(), seed_pools=seed_pools(), optimization_limit=LIMIT,
                maxiter=120, ftol=1e-9, check_targets_used_in_constraints=False,
                global_bounds_certified=False, central_claim_changed=False)


class Envelope:
    def __init__(self, task):
        self.task = task
        self.rstar, protocols = model.data(task['extraction'], task['registration'])
        self.protocols = deepcopy(protocols)
        for p in self.protocols:
            if p['subset'] == 'check':
                for t in p['targets']:
                    t['ratio'] = 0.  # No measured check response reaches the objective.
        self.train = [p for p in self.protocols if p['subset'] == 'train']
        self.target = []
        if task['query'].startswith('P'):
            rate = .5 if task['query'] == 'P05_first' else 4.
            self.target = [p for p in self.protocols if p['cathode'] == 'P-LCO' and p['rate_c'] == rate]
            assert len(self.target) == 1
        self.last_y = None
        self.evaluations = 0
        self.failures = {}
        self.best = None
        self.trace = []
        self.sign = 1. if task['direction'] == 'min' else -1.

    def calculate(self, y):
        if self.last_y is not None and np.array_equal(y, self.last_y):
            return
        self.evaluations += 1
        if self.evaluations % 20 == 0:
            checkpoint_guard()
        x = model.LOW+WIDTH*np.asarray(y)
        try:
            rows = model.predictions(x, self.task, self.rstar, self.train+self.target)
            train = [r for r in rows if r['subset'] == 'train']
            residual = np.array([100. if model.base.unavailable(r) else r['error'] for r in train])
            jac = np.array([r['derivative'] for r in train])*WIDTH[None, :]
            valid = all(model.physical(r) and not model.base.unavailable(r) for r in train)
            if self.task['query'] in ['b', 'eta']:
                col = 3 if self.task['query'] == 'b' else 2
                value = y[col]
                gradient = np.eye(5)[col]
                target_valid = True
            else:
                row = next(r for r in rows if r['subset'] == 'check')
                target_valid = model.physical(row) and not model.base.unavailable(row)
                value = row['predicted_ratio'] if target_valid else 1e6
                gradient = np.array(row['derivative'])*WIDTH if target_valid else np.zeros(5)
            if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jac)) or not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError('Nonfinite objective or constraint')
            objective = self.sign*value if target_valid else 1e6
            objective_gradient = self.sign*gradient if target_valid else np.zeros(5)
            if valid and target_valid and np.max(np.abs(residual)) <= LIMIT:
                if self.best is None or objective < self.best['objective']:
                    self.best = dict(x=x.tolist(), objective=float(objective), max_training_error=float(np.max(np.abs(residual))))
                    self.trace.append(dict(evaluation=self.evaluations, **self.best))
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            key = type(exc).__name__+': '+str(exc)[:160]
            self.failures[key] = self.failures.get(key, 0)+1
            residual = np.full(8, 100.)
            jac = np.zeros((8, 5))
            objective = 1e6
            objective_gradient = np.zeros(5)
        self.last_y = np.array(y).copy()
        self.r, self.j, self.f, self.g = residual, jac, float(objective), objective_gradient

    def objective(self, y):
        self.calculate(y)
        return self.f

    def gradient(self, y):
        self.calculate(y)
        return self.g.copy()

    def constraint(self, y):
        self.calculate(y)
        return np.r_[LIMIT-self.r, LIMIT+self.r]

    def constraint_jac(self, y):
        self.calculate(y)
        return np.r_[-self.j, self.j]


def controls(frozen):
    path = HERE/'controls.json'
    if path.exists():
        old = read(path)
        assert old['passed'] and old['manifest_sha256'] == sha(HERE/'manifest.json')
        return
    tic = time.perf_counter()
    tests = []
    # Known extremal witnesses: .2 <= y0 <= .8, remaining coordinates irrelevant.
    for sign, expected in [(1., .2), (-1., .8)]:
        opt = minimize(lambda y:sign*y[0], np.full(5, .5), jac=lambda y:sign*np.eye(5)[0],
                       method='SLSQP', bounds=[(0., 1.)]*5,
                       constraints=[dict(type='ineq', fun=lambda y:np.array([y[0]-.2, .8-y[0]]),
                                         jac=lambda y:np.array([np.eye(5)[0], -np.eye(5)[0]]))],
                       options=dict(maxiter=120, ftol=1e-9))
        error = abs(opt.x[0]-expected)
        tests.append(dict(name='linear_extremum_'+str(sign), error=float(error), passed=bool(opt.success and error<1e-8)))
    key = 'balance_aligned_early_ordinary'
    for index, seed in enumerate(frozen['seed_pools'][key]['seeds']):
        task = dict(registration='balance_aligned', timing='early', extraction='ordinary', query='P4_first', direction='max', seed=index)
        env = Envelope(task)
        x = np.array(seed['x'])
        y = (x-model.LOW)/WIDTH
        env.calculate(y)
        jac = np.vstack([env.j, -env.g])
        def direct(v):
            xx = model.LOW+WIDTH*v
            rows = assessment.bounded_predictions(xx, task, env.rstar, env.train+env.target)
            assert all(not model.base.unavailable(r) for r in rows)
            train = [r['error'] for r in rows if r['subset'] == 'train']
            target = next(r['predicted_ratio'] for r in rows if r['subset'] == 'check')
            return np.r_[train, target]
        f0 = direct(y)
        errors, stencils = [], []
        for col in range(5):
            h = 1e-5
            d = np.eye(5)[col]*h
            if y[col]-h >= 0. and y[col]+h <= 1.:
                fd = (direct(y+d)-direct(y-d))/(2*h)
                stencil = 'central'
            elif y[col]+2*h <= 1.:
                fd = (-3*f0+4*direct(y+d)-direct(y+2*d))/(2*h)
                stencil = 'forward_second_order'
            else:
                fd = (3*f0-4*direct(y-d)+direct(y-2*d))/(2*h)
                stencil = 'backward_second_order'
            errors.append(float(np.max(np.abs(fd-jac[:, col]))/max(1., np.max(np.abs(fd)))))
            stencils.append(stencil)
        discrepancy = float(np.max(np.abs(f0-np.r_[env.r, -env.f])))
        constraints = env.constraint(y)
        cj = env.constraint_jac(y)
        assert np.array_equal(constraints, np.r_[LIMIT-env.r, LIMIT+env.r])
        assert np.array_equal(cj, np.r_[-env.j, env.j])
        assert all(t['ratio'] == 0. for p in env.protocols if p['subset'] == 'check' for t in p['targets'])
        tests.append(dict(name='normalized_derivatives_seed'+str(index), x=x.tolist(), scaled_errors=errors,
                          stencils=stencils, maximum_ratio_difference=discrepancy,
                          passed=bool(max(errors)<=.001 and discrepancy<=1e-4)))
        checkpoint_guard()
    result = dict(tests=tests, passed=all(t['passed'] for t in tests),
                  manifest_sha256=sha(HERE/'manifest.json'), wall_seconds=time.perf_counter()-tic)
    save(path, result)
    print(json.dumps(result), flush=True)
    assert result['passed'], 'Uncertainty controls must pass before search'


def search(task, seeds):
    checkpoint_guard()
    tic = time.perf_counter()
    env = Envelope(task)
    x0 = np.array(seeds[task['seed']]['x'])
    y0 = (x0-model.LOW)/WIDTH
    opt = minimize(env.objective, y0, jac=env.gradient, method='SLSQP', bounds=[(0., 1.)]*5,
                   constraints=[dict(type='ineq', fun=env.constraint, jac=env.constraint_jac)],
                   options=dict(maxiter=120, ftol=1e-9))
    xopt = model.LOW+WIDTH*opt.x
    optimizer = dict(success=bool(opt.success), status=int(opt.status), message=str(opt.message),
                     nit=int(opt.nit), nfev=int(opt.nfev), njev=int(opt.njev), fun=float(opt.fun), x=xopt.tolist())
    output = dict(id=identifier(task), task=task, seed=seeds[task['seed']], optimizer=optimizer,
                  best_feasible=env.best, improvement_trace=env.trace, evaluations=env.evaluations,
                  evaluation_failures=env.failures, manifest_sha256=sha(HERE/'manifest.json'),
                  check_targets_used_in_constraints=False, check_targets_used_in_objective=False,
                  prospective_prediction_qualified=False, optimization_wall_seconds=time.perf_counter()-tic)
    save(HERE/'optimizer_returns'/(output['id']+'.json'), output)
    candidates = [('start', x0), ('optimizer', xopt)]
    if env.best is not None:
        candidates.append(('best_feasible', np.array(env.best['x'])))
    problem = model.Problem(task)
    assessed = []
    for label, x in candidates:
        c = assessment.assess(problem, x, label)
        c['uncertainty_admitted'] = bool(c['calibration_admissible'] and c['subsets']['train']['screen_pass'])
        assessed.append(c)
    output.update(rstar=problem.rstar, candidates=assessed, wall_seconds=time.perf_counter()-tic)
    save(HERE/'searches'/(output['id']+'.json'), output)
    checkpoint_guard()
    return dict(id=output['id'], seconds=output['wall_seconds'], optimizer_success=optimizer['success'],
                optimizer_status=optimizer['status'], admitted=sum(c['uncertainty_admitted'] for c in assessed),
                best_objective=None if env.best is None else env.best['objective'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    checkpoint_guard()
    frozen = json.loads(json.dumps(manifest()))
    path = HERE/'manifest.json'
    if path.exists():
        assert read(path) == frozen, 'Exact manifest required'
    else:
        save(path, frozen)
    controls(frozen)
    chosen = tasks()
    if args.pilot:
        chosen = [t for t in chosen if setting_key(t) == ('balance_aligned', 'early', 'ordinary') and t['seed'] == 0
                  and (t['query'], t['direction']) in [('b', 'max'), ('eta', 'min'), ('P4_first', 'min'), ('P4_first', 'max')]]
    pending = []
    for task in chosen:
        path = HERE/'searches'/(identifier(task)+'.json')
        if args.resume and path.exists():
            old = read(path)
            assert old['manifest_sha256'] == sha(HERE/'manifest.json') and old['task'] == task
        else:
            assert not path.exists(), 'Use --resume for saved searches'
            pending.append(task)
    workers = int(os.environ.get('AJ_COMPUTE_WORKERS', '1'))
    tic = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(search, t, frozen['seed_pools']['_'.join(setting_key(t))]['seeds']) for t in pending]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print(json.dumps(r), flush=True)
            checkpoint_guard()
    save(HERE/('pilot.json' if args.pilot else 'execution.json'), dict(tasks=results, workers=workers,
         reused_tasks=len(chosen)-len(pending), wall_seconds=time.perf_counter()-tic))


if __name__ == '__main__':
    main()
