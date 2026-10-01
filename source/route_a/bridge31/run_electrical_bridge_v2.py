"""Map calibrated scalar resistance to conserving spatial transport."""
from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE.parent/'transport4'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


electrical = module('existing_boundary', HERE.parent/'coupling5/electrical_boundary.py')
volume = module('existing_volume', HERE.parent/'transport4/run_finite_contact.py')
source = module('existing_io', HERE.parent/'forecast9/run_recovery_probe.py')
read, save, sha = source.read, source.save, source.sha
J = 1.2
FRACTIONS = [0., .1, .5, 1.]
SHAPES = ['uniform', 'cosine_one', 'cosine_four']


def freeze():
    old = read(HERE.parent/'decomposition29/manifest.json')
    null = read(HERE.parent/'jointnull30/manifest.json')
    reps = [next(s['original_witness_id'] for s in null['pools']['late_'+e]
                 if s['origin'] == 'stage29_aligned_late_minimum_all_point_sse_eta_zero')
            for e in ['published', 'ordinary', 'robust']]
    bounds = []
    for w in old['witnesses']:
        b, A = w['x'][3:]
        Rb, C = b*w['rstar'], A*(1-b)*w['rstar']
        assert 0 <= Rb and C > 0
        bounds.append(dict(id=w['id'], setting=w['setting'], Rb=Rb, C=C, beta_max=Rb/C))
    selected = list(dict.fromkeys(reps+[min(bounds, key=lambda r:r['beta_max'])['id'], max(bounds, key=lambda r:r['beta_max'])['id']]))
    paths = [Path(__file__).resolve(), HERE/'PLAN.md', HERE/'UNIFORM_SHAPE_REPAIR.md',
             HERE/'run_electrical_bridge.py', HERE/'manifest.json', HERE.parent/'decomposition29/manifest.json',
             HERE.parent/'jointnull30/manifest.json', HERE.parent/'coupling5/electrical_boundary.py',
             HERE.parent/'coupling5/prechecks.json', HERE.parent/'coupling5/precheck_manifest.json',
             HERE.parent/'transport4/run_finite_contact.py', HERE.parent/'transport4/run_transport.py']
    hashes = old['sha256'].copy()
    assert read(HERE.parent/'coupling5/prechecks.json')['all_pass']
    pre = read(HERE.parent/'coupling5/precheck_manifest.json')['sha256']
    for p in paths:
        key = str(p.relative_to(ROOT))
        if key in pre:
            assert sha(p) == pre[key], key
        hashes[key] = sha(p)
    for path, digest in hashes.items():
        assert sha(ROOT/path) == digest, path
    frozen = dict(sha256=hashes, all_witness_beta_bounds=bounds, representative_ids=reps,
                  witnesses=[w for wid in selected for w in old['witnesses'] if w['id'] == wid],
                  current_mA_cm2=J, bulk_fractions=FRACTIONS, shapes=SHAPES,
                  geometry_measured=False, new_evolution_adopted=False,
                  central_claim_changed=False, prospective_prediction_qualified=False)
    path = HERE/'manifest_v2.json'
    if path.exists():
        assert read(path) == frozen
    else:
        save(path, frozen)
    return frozen


def profile(n, shift, shape):
    x = (np.arange(n)+.5)/n
    mode = 0 if shape == 'uniform' else 1 if shape == 'cosine_one' else 4
    return np.full(n, float(expit(shift))) if mode == 0 else expit(shift+2*np.cos(2*np.pi*mode*x))


def solve(a, Rb, C, fraction):
    B, S = fraction*Rb, (1-fraction)*Rb
    if B == 0.:
        q = J*a/np.mean(a)
        qa = np.full(len(a), J/np.mean(a))
        R = Rb+C/np.mean(a)
        diag = dict(status='parallel_limit', current_relative_error=abs(float(np.mean(q))/J-1),
                    power_relative_error=0., boundary_law_relative_l2=0., zero_gap_current=True)
    else:
        diag, q, qa = electrical.solve(a, J, B/C)
        assert diag['status'] == 'solved'
        R = S+B*diag['resistance_over_bulk']
    assert np.all(a > 0) and np.all(a <= 1) and S >= 0 and B >= 0
    invariant = max(diag[k] for k in ['current_relative_error','power_relative_error','boundary_law_relative_l2'])
    relation = float(np.linalg.norm(q-a*qa)/max(np.linalg.norm(q), 1e-30))
    assert invariant <= 1e-9 and relation <= 1e-9
    assert np.min(q) >= -1e-12 and diag['zero_gap_current']
    mean_a = float(np.mean(a))
    flat = J/mean_a
    return dict(resistance=float(R), bulk_resistance=B, series_resistance=S, beta=B/C,
                contact=a.tolist(), nominal_current=q.tolist(), active_current=qa.tolist(), electrical=diag,
                maximum_invariant_error=invariant, current_area_relation_error=relation,
                mean_contact=mean_a, minimum_contact=float(np.min(a)), maximum_contact=float(np.max(a)),
                peak_nominal_current=float(np.max(q)), peak_active_current=float(np.max(qa)),
                uniform_active_current=flat, spatial_peak_factor=float(np.max(qa)/flat),
                weighted_active_current_cv=float(np.sqrt(np.mean(a*(qa-flat)**2)/mean_a)/flat))


def run(witness, frozen):
    path = HERE/'cases'/(witness['id']+'.json')
    if path.exists():
        output = read(path)
        assert output['witness'] == witness and output['manifest_sha256'] == sha(HERE/'manifest_v2.json')
        return output
    tic = time.perf_counter()
    b, A = witness['x'][3:]
    Rb, C = b*witness['rstar'], A*(1-b)*witness['rstar']
    cases = []
    for h in witness['diagonal_histories']:
        target = h['R_ch']
        scalar_a = C/(target-Rb)
        assert 0 < scalar_a < 1
        for fraction in FRACTIONS:
            for shape in SHAPES:
                checkpoint_guard()
                def residual(shift):
                    return solve(profile(96, shift, shape), Rb, C, fraction)['resistance']/target-1
                shift = brentq(residual, -35., 35., xtol=1e-12, rtol=1e-14)
                coarse = solve(profile(96, shift, shape), Rb, C, fraction)
                fine = solve(profile(192, shift, shape), Rb, C, fraction)
                matching = abs(coarse['resistance']/target-1)
                assert matching <= 1e-9
                if shape == 'uniform':
                    assert abs(coarse['mean_contact']-scalar_a) <= 1e-9
                    assert abs(coarse['spatial_peak_factor']-1.) <= 1e-9
                    assert abs(coarse['resistance']-(Rb+C/coarse['mean_contact']))/target <= 1e-9
                refinement = dict(resistance_relative_change=abs(fine['resistance']-coarse['resistance'])/fine['resistance'],
                                  peak_active_current_relative_change=abs(fine['peak_active_current']-coarse['peak_active_current'])/fine['peak_active_current'],
                                  refitted_shift=False)
                refinement['passed'] = bool(refinement['resistance_relative_change'] <= 1e-4 and refinement['peak_active_current_relative_change'] <= .01)
                independent = None
                if witness['id'] in frozen['representative_ids'] and fraction == 1. and shape == 'cosine_four':
                    check, fields = volume.solve(96, np.array(coarse['contact']), coarse['beta'], current=J)
                    value = coarse['series_resistance']+coarse['bulk_resistance']*check['resistance_over_bulk']
                    re = abs(value-coarse['resistance'])/coarse['resistance']
                    qe = float(np.linalg.norm(fields[1]-coarse['nominal_current'])/np.linalg.norm(fields[1]))
                    independent = dict(electrical=check, physical_resistance=value, resistance_relative_difference=re,
                                       nominal_current_relative_l2=qe, nominal_current=fields[1].tolist(),
                                       passed=bool(re <= 1e-8 and qe <= 1e-8 and volume.conservative(check)))
                cases.append(dict(cathode=h['cathode'], initial_resistance=target, scalar_initial_contact=scalar_a,
                                  bulk_fraction=fraction, shape=shape, shift=shift, coarse=coarse, fine=fine,
                                  matching_relative_error=matching, refinement=refinement, independent_volume=independent))
    output = dict(witness=witness, cases=cases, wall_seconds=time.perf_counter()-tic,
                  manifest_sha256=sha(HERE/'manifest_v2.json'))
    save(path, output)
    return output


def main():
    checkpoint_guard()
    tic = time.perf_counter()
    frozen = freeze()
    outputs = []
    for w in frozen['witnesses']:
        r = run(w, frozen)
        outputs.append(r)
        print(json.dumps(dict(id=w['id'], cases=len(r['cases']), refined=sum(c['refinement']['passed'] for c in r['cases']))), flush=True)
    save(HERE/'execution.json', dict(witnesses=len(outputs), static_cases=sum(len(r['cases']) for r in outputs),
                                    wall_seconds=time.perf_counter()-tic, workers=1, manifest_sha256=sha(HERE/'manifest_v2.json')))


if __name__ == '__main__':
    main()
