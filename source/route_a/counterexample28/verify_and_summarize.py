"""Audit retained all-data contrast profiles without rerunning trajectories."""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def unavailable(r): return r['disconnected'] or r.get('observation_invalid', False)


def physical(r):
    return (not r.get('observation_invalid', False) and 0 < r['initial_a'] < 1 and 0 < r['effective_a'] <= 1
            and r['replenishment_scaled'] >= -1e-7 and r['maximum_remaining_height_um'] <= 40+1e-10
            and r['required_initial_metal_stock_mAh_cm2'] <= r['nominal_available_metal_stock_mAh_cm2']+1e-7)


def main():
    frozen = read(HERE/'manifest.json')
    for name, digest in frozen['sha256'].items(): assert sha(ROOT/name) == digest, name
    controls = read(HERE/'controls.json'); execution = read(HERE/'execution.json')
    assert controls['passed'] and execution['completed'] == len(frozen['tasks']) == 10
    assert controls['manifest_sha256'] == execution['manifest_sha256'] == sha(HERE/'manifest.json')
    assert all(max(c['scaled_errors']) <= .001 and c['value_difference'] <= 1e-4 for c in controls['tests'])
    old = read(HERE.parent/'prospective27/manifest.json')
    U = read(HERE.parent/'mechanics3/mechanical_checks.json')['li_equivalent_um_per_mah_cm2']
    prior_evidence = read(HERE.parent/'uncertainty25/evidence.json')
    bound_reference = HERE.parent/'uncertainty25/manifest.json'
    assert sha(bound_reference) == prior_evidence['provenance']['manifest.json']
    bounds = read(bound_reference)
    low, high = bounds['low'], bounds['high']
    records = []
    for witness in old['witnesses']:
        path = HERE.parent/'prospective27/cases'/(witness['id']+'.json')
        c = read(path)
        records.append(dict(id='stage27/'+witness['id'], source=str(path.relative_to(ROOT)),
                            setting=witness['setting'], x=witness['x'], histories=c['histories']))
    statuses = {}; eval_failures = {}; known_rows = 0; projection_rows = 0
    max_known_difference = 0.; max_new_difference = 0.; max_ledger = 0.; candidates = 0
    ids = []
    for task in frozen['tasks']:
        fid = task['extraction']+'_'+task['direction']+'_seed'+str(task['seed']); ids.append(fid)
        path = HERE/'searches'/(fid+'.json'); fit = read(path)
        raw = read(HERE/'optimizer_returns'/(fid+'.json'))
        assert all(fit[k] == v for k, v in raw.items())
        assert fit['task'] == task and fit['manifest_sha256'] == sha(HERE/'manifest.json')
        assert fit['optimizer']['nit'] <= 30
        assert fit['seed'] == frozen['pools'][task['extraction']]['seeds'][task['seed']]
        assert next(c for c in fit['candidates'] if c['label'] == 'start')['x'] == fit['seed']['x']
        statuses[fid] = fit['optimizer']
        if fit['evaluation_failures']: eval_failures[fid] = fit['evaluation_failures']
        seedfit = read(ROOT/fit['seed']['source'])
        reference = next(c for c in seedfit['candidates'] if c['label'] == fit['seed']['label'])
        observed = {(r['cathode'], r['rate_c'], r['stage']):r['observed_ratio'] for r in reference['points']}
        for c in fit['candidates']:
            candidates += 1
            for value, lo, hi in zip(c['x'], low, high):
                assert min(lo, lo+(hi-lo)*0) <= value <= max(hi, lo+(hi-lo)*1)
            assert len(c['points']) == len(c['optimizer_predictions']) == len(c['numerical_checks']) == 14
            passes = []
            for r, fast, check in zip(c['points'], c['optimizer_predictions'], c['numerical_checks']):
                assert r['observed_ratio'] == observed[(r['cathode'], r['rate_c'], r['stage'])]
                same = r['disconnected'] == fast['disconnected'] and r.get('observation_invalid', False) == fast.get('observation_invalid', False)
                delta = None if unavailable(r) or unavailable(fast) else abs(r['predicted_ratio']-fast['predicted_ratio'])
                passed = same and (delta is None or delta <= 1e-4)
                assert check['passed'] == passed and check['status_matches'] == same
                if delta is not None:
                    assert math.isclose(delta, check['ratio_difference'], abs_tol=1e-15)
                    max_known_difference = max(max_known_difference, delta)
                    assert math.isclose(r['error'], r['predicted_ratio']-r['observed_ratio'], abs_tol=1e-15)
                passes.append(passed); known_rows += 1
            assert c['numerical_pass'] == all(passes)
            assert c['physical_observation_pass'] == all(physical(r) for r in c['points'])
            screens = []
            for subset in ['train', 'check']:
                rr = [r for r in c['points'] if r['subset'] == subset]
                errors = [100. if unavailable(r) else r['error'] for r in rr]
                screen = all(not unavailable(r) for r in rr) and max(abs(e) for e in errors) <= .03
                metrics = c['subsets'][subset]
                assert metrics['screen_pass'] == screen and metrics['n'] == len(rr)
                assert math.isclose(metrics['sse'], sum(e*e for e in errors), rel_tol=1e-10, abs_tol=1e-10)
                screens.append(screen)
            admitted = c['numerical_pass'] and c['physical_observation_pass'] and all(screens)
            assert admitted == c['all_known_admitted']
            x = c['x']; L, b, A = 10**x[0], x[3], x[4]; rstar = fit['rstar']
            for h in c['new_protocol']:
                if 'rows' not in h:
                    assert not h['available']; continue
                valid_rows = []
                for r in h['rows']:
                    if r['ratio'] is not None:
                        assert math.isclose(r['ratio'], (b*rstar+A*(1-b)*rstar/r['a'])/h['R_ch'], rel_tol=1e-12)
                        assert math.isclose(r['ratio_difference'], abs(r['ratio']-r['independent_ratio']), abs_tol=1e-15)
                        max_new_difference = max(max_new_difference, r['ratio_difference'])
                    a = max(0., min(1., 1-r['recession_um']/L)); volume = L*a*a/2
                    assert math.isclose(volume, r['remaining_volume_um'], rel_tol=1e-12, abs_tol=1e-12)
                    ledger = abs(volume-r['initial_volume_um']+U*r['independent_stripped_charge_mAh_cm2']-r['replenished_volume_um'])
                    assert math.isclose(ledger, r['volume_ledger_error_um'], abs_tol=1e-12)
                    max_ledger = max(max_ledger, ledger)
                    numeric = ((r['ratio_difference'] is None or r['ratio_difference'] <= 1e-4)
                               and ledger <= 1e-6*max(1., volume, abs(r['replenished_volume_um']))
                               and abs(r['independent_stripped_charge_mAh_cm2']-r['stripped_charge_mAh_cm2']) <= 1e-6
                               and r['projection'] <= 1e-7*max(1., 1/(A*A)))
                    phys = (not r['stopped'] and 0 < a <= 1 and L*a <= 40+1e-10
                            and volume/U+r['independent_stripped_charge_mAh_cm2'] <= 40/U+1e-7 and r['replenished_volume_um'] >= -1e-7)
                    assert r['numerical_pass'] == numeric and r['physical_pass'] == phys
                    valid_rows.append(numeric and phys); projection_rows += 1
                assert h['available'] == all(valid_rows)
            if admitted:
                records.append(dict(id=fid+'/'+c['label'], source=str(path.relative_to(ROOT)),
                                    setting={k:task[k] for k in ['registration', 'timing', 'extraction']},
                                    x=c['x'], histories=c['new_protocol']))
    assert sorted(p.stem for p in (HERE/'searches').glob('*.json')) == sorted(ids)
    assert sorted(p.stem for p in (HERE/'optimizer_returns').glob('*.json')) == sorted(ids)
    unique = {}
    for r in records:
        k = tuple(r['setting'][n] for n in ['registration', 'timing', 'extraction'])+tuple(r['x'])
        unique.setdefault(k, r)
    records = list(unique.values())

    def spread(selected, stage, quantity):
        values = []; unavailable_ids = []
        for r in selected:
            pair = {}
            for h in r['histories']:
                row = next((v for v in h.get('rows', []) if v['stage'] == stage), None)
                pair[h['cathode']] = row['ratio'] if row and row['numerical_pass'] and row['physical_pass'] else None
            value = (pair['P-LCO']-pair['N-LCO'] if all(v is not None for v in pair.values()) else None) if quantity == 'P_minus_N' else pair[quantity]
            if value is None: unavailable_ids.append(r['id'])
            else: values.append(dict(id=r['id'], value=value, x=r['x'], source=r['source']))
        lo = min(values, key=lambda v:v['value']) if values else None
        hi = max(values, key=lambda v:v['value']) if values else None
        return dict(stage=stage, quantity=quantity, minimum=lo, maximum=hi, available=len(values),
                    unavailable_ids=unavailable_ids, sampled_span=None if lo is None else hi['value']-lo['value'])

    scopes = []
    for name, selected in [('all_retained_families', records),
                           ('aligned_late', [r for r in records if r['setting']['registration'] == 'balance_aligned' and r['setting']['timing'] == 'late'])]+[
        (e, [r for r in records if r['setting'] == dict(registration='balance_aligned', timing='late', extraction=e)])
        for e in ['published', 'ordinary', 'robust']]:
        scopes.append(dict(name=name, witnesses=len(selected), spreads=[spread(selected, s, q)
            for s in ['end_discharge', 'after_30_min_rest'] for q in ['P-LCO', 'N-LCO', 'P_minus_N']]))
    report = dict(passed=True, search_count=len(ids), candidate_records=candidates, known_rows_checked=known_rows,
                  new_protocol_rows_checked=projection_rows, retained_distinct_witnesses=len(records),
                  max_known_LSODA_Radau_difference=max_known_difference,
                  max_new_independent_physical_difference=max_new_difference, max_new_volume_ledger_error_um=max_ledger,
                  optimizer_statuses=statuses, evaluation_failures=eval_failures, scopes=scopes,
                  central_claim_changed=False, prospective_prediction_qualified=False, global_bounds_certified=False,
                  analysis_sha256=sha(Path(__file__).resolve()),
                  sha256={str(p.relative_to(HERE)):sha(p) for p in [HERE/'manifest.json', HERE/'execution.json', HERE/'controls.json']
                          +sorted((HERE/'searches').glob('*.json'))+sorted((HERE/'optimizer_returns').glob('*.json'))})
    (HERE/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['sha256', 'scopes', 'optimizer_statuses']}, indent=2))


if __name__ == '__main__': main()
