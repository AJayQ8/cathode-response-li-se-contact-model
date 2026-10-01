"""Frozen finite-height model across extractions, readouts and starting points."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import time

import numpy as np
from scipy.optimize import least_squares
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location('finite_height', HERE.parent/'finite22/run_finite_height.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
read, save, sha = model.read, model.save, model.sha
STARTS = [(20., .001, 20., .3, .5), (5., .01, 5., .6, .8), (35., .0005, 50., .9, .95)]


def tasks():
    return [dict(registration=r, timing=t, extraction=e, start=s)
            for r in ['local', 'balance_aligned'] for t in ['early', 'late']
            for e in ['published', 'ordinary', 'robust'] for s in range(3)]


def initial(task):
    x = np.array(STARTS[task['start']], dtype=float)
    x[:2] = np.log10(x[:2])
    return x


def pilot_paths():
    return [HERE.parent/'finite22/fits'/(model.task_id(t)+'.json') for t in model.tasks()]


def manifest():
    prior = read(HERE.parent/'finite22/manifest.json')['sha256']
    paths = [Path(__file__), HERE/'PLAN.md', HERE.parent/'finite22/controls.json',
             HERE.parent/'finite22/verify_outputs.py', HERE.parent/'finite22/verification.json', *pilot_paths()]
    hashes = {**prior, **{str(p.relative_to(ROOT)): sha(p) for p in paths}}
    for name, digest in hashes.items():
        assert sha(ROOT/name) == digest, name
    return dict(sha256=hashes, tasks=tasks(), low=model.LOW.tolist(), high=model.HIGH.tolist(),
                starts_physical=STARTS, pilot_reuse={p.name: sha(p) for p in pilot_paths()},
                central_claim_changed=False, check_targets_used_in_calibration=False,
                prospective_prediction_qualified=False)


def controls():
    prior = read(HERE.parent/'finite22/controls.json')
    assert prior['passed'] and read(HERE.parent/'finite22/verification.json')['passed']
    pilot = read(HERE.parent/'finite22/fits/balance_aligned_early_ordinary_start0.json')
    pilot_x = np.array(next(c['x'] for c in pilot['candidates'] if c['label']=='optimizer'))
    other = dict(registration='local', timing='late', extraction='published', start=2)
    tests = []
    for task, x in [(pilot['task'], pilot_x), (other, initial(other))]:
        checkpoint_guard()
        p = model.Problem(task)
        jac = p.jacobian(x)
        assessment = p.assess(x, 'hard_control')
        def residual_direct(v):
            rows = model.predictions(v, task, p.rstar, p.train, True)
            assert all(not model.base.unavailable(r) for r in rows), 'Training finite-difference domain'
            return np.array([r['error'] for r in rows])
        f0 = residual_direct(x)
        errors, methods, differences = [], [], []
        for i, h in enumerate([1e-4, 1e-4, .005, 1e-4, 1e-4]):
            direction = np.zeros(5)
            direction[i] = h
            if x[i]-h >= model.LOW[i] and x[i]+h <= model.HIGH[i]:
                fd = (residual_direct(x+direction)-residual_direct(x-direction))/(2*h)
                method = 'central'
            elif x[i]+2*h <= model.HIGH[i]:
                fd = (-3*f0+4*residual_direct(x+direction)-residual_direct(x+2*direction))/(2*h)
                method = 'forward_second_order'
            else:
                assert x[i]-2*h >= model.LOW[i]
                fd = (3*f0-4*residual_direct(x-direction)+residual_direct(x-2*direction))/(2*h)
                method = 'backward_second_order'
            errors.append(float(np.max(np.abs(fd-jac[:, i]))/max(1., np.max(np.abs(fd)))))
            methods.append(method)
            differences.append(fd.tolist())
        tests.append(dict(task=task, x=x.tolist(), analytic_jacobian=jac.tolist(),
                          independent_finite_difference_columns=differences, methods=methods,
                          scaled_errors=errors, calibration_admissible=assessment['calibration_admissible'],
                          numerical_pass=assessment['numerical_pass'],
                          maximum_ratio_difference=max(c['ratio_difference'] or 0 for c in assessment['numerical_checks']),
                          check_initial_domain_failures=[r for r in assessment['points']
                                                       if r['subset']=='check' and r.get('observation_invalid', False)],
                          passed=bool(max(errors)<=.001 and assessment['numerical_pass'] and assessment['calibration_admissible'])))
        print(json.dumps(dict(control=model.task_id(task), scaled_errors=errors, passed=tests[-1]['passed'])), flush=True)
    report = dict(passed=all(t['passed'] for t in tests), tests=tests,
                  inherited_controls_sha256=sha(HERE.parent/'finite22/controls.json'),
                  inherited_verification_sha256=sha(HERE.parent/'finite22/verification.json'))
    save(HERE/'controls.json', report)
    assert report['passed'], 'Hard-state derivative controls must pass before batch'


def fit(task):
    tic = time.perf_counter()
    checkpoint_guard()
    p = model.Problem(task)
    x0 = initial(task)
    opt = least_squares(p.residual, x0, jac=p.jacobian, bounds=(model.LOW, model.HIGH),
                        x_scale='jac', max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-8)
    candidates = [p.assess(x0, 'start'), p.assess(opt.x, 'optimizer')]
    allowed = [c for c in candidates if c['calibration_admissible']]
    chosen = min(allowed, key=lambda c: c['training_objective']) if allowed else None
    out = dict(id=model.task_id(task), task=task, rstar=p.rstar, candidates=candidates,
               selected_label=None if chosen is None else chosen['label'],
               optimizer=dict(success=bool(opt.success), status=int(opt.status), message=str(opt.message),
                              nfev=int(opt.nfev), njev=int(opt.njev), cost=float(opt.cost), optimality=float(opt.optimality)),
               evaluations=p.evaluations, evaluation_failures=p.failures, wall_seconds=time.perf_counter()-tic,
               prospective_prediction_qualified=False, check_targets_used_in_calibration=False,
               input_manifest_sha256=sha(HERE/'manifest.json'))
    save(HERE/'fits'/(out['id']+'.json'), out)
    checkpoint_guard()
    return dict(id=out['id'], seconds=out['wall_seconds'], optimizer_success=bool(opt.success),
                training_max=None if chosen is None else chosen['subsets']['train']['max_abs_error'],
                check_max=None if chosen is None else chosen['subsets']['check']['max_abs_error'],
                physical_pass=False if chosen is None else chosen['physical_observation_pass'])


def summarize():
    fits = [read(HERE/'fits'/(model.task_id(t)+'.json')) for t in tasks()]
    groups = []
    for registration in ['local', 'balance_aligned']:
        for timing in ['early', 'late']:
            for extraction in ['published', 'ordinary', 'robust']:
                matched = [f for f in fits if (f['task']['registration'], f['task']['timing'], f['task']['extraction'])
                           == (registration, timing, extraction)]
                assert len(matched) == 3
                pool = [(f, c) for f in matched for c in f['candidates'] if c['label']==f['selected_label']]
                group = dict(registration=registration, timing=timing, extraction=extraction, starts_completed=len(matched))
                if not pool:
                    group.update(selected_id=None, all_screens_pass=False)
                else:
                    f, c = min(pool, key=lambda pair: pair[1]['training_objective'])
                    group.update(selected_id=f['id'], selected_label=c['label'], x=c['x'], subsets=c['subsets'],
                                 numerical_pass=c['numerical_pass'], physical_observation_pass=c['physical_observation_pass'],
                                 training_objective=c['training_objective'], optimizer_success=f['optimizer']['success'],
                                 bound_indices=c['bound_indices'],
                                 all_screens_pass=bool(c['numerical_pass'] and c['physical_observation_pass']
                                                      and all(s['screen_pass'] for s in c['subsets'].values())))
                groups.append(group)
    out = dict(groups=groups, fit_count=len(fits), optimizer_success_count=sum(f['optimizer']['success'] for f in fits),
               selected_all_screens_pass=sum(g['all_screens_pass'] for g in groups),
               prospective_prediction_qualified=False, central_claim_changed=False, prior_failures_reclassified=False)
    save(HERE/'summary.json', out)
    print(json.dumps({k: v for k, v in out.items() if k!='groups'}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    checkpoint_guard()
    frozen = json.loads(json.dumps(manifest()))
    path = HERE/'manifest.json'
    if path.exists():
        assert read(path) == frozen, 'Exact manifest required'
    else:
        save(path, frozen)
    controls()
    (HERE/'fits').mkdir(exist_ok=True)
    for pilot in pilot_paths():
        target = HERE/'fits'/pilot.name
        if target.exists():
            assert sha(target) == sha(pilot), 'Pilot reuse must be byte-identical'
        else:
            shutil.copyfile(pilot, target)
    pending = []
    pilot_names = {p.name for p in pilot_paths()}
    progress = read(HERE/'progress.json') if (HERE/'progress.json').exists() else {}
    for task in tasks():
        path = HERE/'fits'/(model.task_id(task)+'.json')
        if path.name in pilot_names:
            continue
        if args.resume and path.exists():
            old = read(path)
            assert old['input_manifest_sha256']==sha(HERE/'manifest.json') and old['task']==task
            if path.name in progress:
                assert sha(path)==progress[path.name]
        else:
            assert not path.exists(), 'Existing output requires --resume'
            pending.append(task)
    workers = int(os.environ.get('AJ_COMPUTE_WORKERS', '1'))
    tic = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fit, t) for t in pending]):
            result = future.result()
            results.append(result)
            name = result['id']+'.json'
            progress[name] = sha(HERE/'fits'/name)
            save(HERE/'progress.json', progress)
            print(json.dumps(result), flush=True)
            checkpoint_guard()
    save(HERE/'execution.json', dict(tasks=results, workers=workers, reused_pilots=2,
                                    resumed_completed=len(tasks())-len(pending)-2,
                                    wall_seconds=time.perf_counter()-tic))
    summarize()


if __name__ == '__main__':
    main()
