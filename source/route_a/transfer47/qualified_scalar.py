"""Independent tighter scalar checks for the unchanged finite-height model.

The scalar RHS, domain, stopping event, projection rule and reported quantities
match robust23/resume_startup_repair.py. Only Radau tolerances are tightened.
Importing this module does not launch controls or predictions; no source module
is monkeypatched. The caller owns manifests, qualification scheduling and saves.
"""
from pathlib import Path
import importlib.util

import numpy as np
from scipy.integrate import solve_ivp
from shared_compute import checkpoint_guard


HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE.parent/'robust23/resume_startup_repair.py'
_spec = importlib.util.spec_from_file_location('transfer47_scalar_source', SOURCE_PATH)
assessment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assessment)
model = assessment.model
U, M, ZMIN = model.U, model.M, model.ZMIN
initial_state, domain, pressure_increment = model.initial_state, model.domain, model.pressure_increment
WIDTH = model.HIGH-model.LOW
RTOL, ATOL = 1e-12, 1e-14
FORWARD_LIMIT, DERIVATIVE_LIMIT = 1e-7, .001
SCALED_STEPS = (1e-3, 3e-4, 1e-4)


def _advance(state, duration, x, cathode, q_start, J):
    """Frozen scalar advance with tighter tolerances and the same startup rule."""
    L, H0 = 10**x[:2]
    eta, A = x[2], x[4]
    D = U/(L*A*A)
    floor, upper = ZMIN**2, 1/(A*A)
    if state[0] <= floor:
        return state.copy(), True, 0., 0.
    if duration == 0:
        return state.copy(), False, 0., 0.

    def rhs(t, y):
        assert -1e-14 <= t <= duration+1e-14, 'RHS outside integration interval'
        z = max(float(np.sqrt(max(y[0], 0.))), ZMIN/4)
        scale = (2+eta*pressure_increment(cathode, q_start+J*t))/2
        if scale <= 0:
            raise ValueError('Noncompressive pressure during integration')
        g = (1-A*z)*z**(1-M) if A*z < 1 else 0.
        return [2*H0*scale**M*g-2*D*J]

    def event(t, y):
        return y[0]-floor

    event.terminal, event.direction = True, -1
    result = solve_ivp(rhs, (0., duration), state[:1], method='Radau',
                       rtol=RTOL, atol=ATOL, events=event,
                       first_step=min(duration, 1e-4))
    if not result.success or not np.all(np.isfinite(result.y[:, -1])):
        raise RuntimeError(result.message)
    output = state.copy()
    output[0] = result.y[0, -1]
    correction = max(0., float(output[0]-upper))
    if correction > 1e-7*max(1., upper):
        raise RuntimeError('Full-contact projection exceeds numerical tolerance')
    output[0] = min(upper, max(floor, output[0]))
    return output, bool(len(result.t_events[0])), float(result.t[-1]), correction


def tight_predictions(x, task, rstar, protocols):
    """Return the original scalar prediction schema at rtol=1e-12/atol=1e-14."""
    x = np.asarray(x, dtype=float)
    L, H0 = 10**x[:2]
    b, A = x[3:]
    rb, D, out = b*rstar, U/(L*A*A), []
    for p in protocols:
        if p['excluded_ambiguous']:
            continue
        state, z0, a0 = initial_state(x, rstar, p['R_ch'])
        load = domain(p, x[2])
        if state is None or not load['valid']:
            for target in p['targets']:
                out.append(dict(cathode=p['cathode'], rate_c=p['rate_c'], stage=target['stage'],
                                subset=p['subset'], observed_ratio=target['ratio'],
                                predicted_ratio=None, error=None, disconnected=False,
                                observation_invalid=True,
                                invalid_reason='Initial contact outside finite-height domain'
                                if state is None else load['reason'],
                                initial_z=z0, initial_a=a0, loading=load, derivative=[0.]*5))
            continue
        start = state.copy()
        s1, stop1, t1, c1 = _advance(state, p['t1'], x, p['cathode'], p['q0'], p['J'])
        sr, stopr, tr, cr = ((s1, True, 0., 0.) if stop1 else
                            _advance(s1, p['rest'], x, p['cathode'], p['q0']+p['J']*t1, 0.))
        s2, stop2, t2, c2 = ((sr, True, 0., 0.) if stopr else
                            _advance(sr, p['t2'], x, p['cathode'], p['q0']+p['J']*t1, p['J']))
        first = ((s1, stop1, p['J']*t1) if task['timing'] == 'early' else
                 (sr, stopr, p['J']*t1))
        for target, (state, stop, Q) in zip(p['targets'], [first, (s2, stop2, p['J']*(t1+t2))]):
            w = float(state[0])
            z, n = float(np.sqrt(w)), rstar-rb
            a = A*z
            value = None if stop else float((rb+n/z)/p['R_ch'])
            out.append(dict(cathode=p['cathode'], rate_c=p['rate_c'], stage=target['stage'],
                            subset=p['subset'], observed_ratio=target['ratio'],
                            predicted_ratio=value, error=None if stop else value-target['ratio'],
                            disconnected=stop, initial_z=z0, initial_a=a0,
                            initial_w=float(start[0]), w=w, z=z, effective_a=a, loading=load,
                            height_spread_um=L, maximum_remaining_height_um=L*a,
                            mean_remaining_height_um=L*a/2, initial_mean_remaining_height_um=L*a0/2,
                            creep_velocity_scale_um_h=H0*L*A**(M+1), damage_coefficient=D,
                            stripped_charge_mAh_cm2=Q, replenishment_scaled=float(w-start[0]+2*D*Q),
                            required_initial_metal_stock_mAh_cm2=float(w/(2*D)+Q),
                            nominal_available_metal_stock_mAh_cm2=40/U,
                            maximum_w_projection=max(c1, cr, c2), derivative=[0.]*5))
    return out


def _key(row):
    return [row[name] for name in ['cathode', 'rate_c', 'stage', 'subset', 'observed_ratio']]


def _sample(x, rows):
    """Raw forward values and boundary diagnostics retained for each FD point."""
    margins = [1-r['initial_a'] for r in rows if r.get('initial_a') is not None]
    projections = [r['maximum_w_projection'] for r in rows if 'maximum_w_projection' in r]
    return dict(x=np.asarray(x).tolist(), row_keys=[_key(r) for r in rows],
                ratios=[r['predicted_ratio'] for r in rows], errors=[r['error'] for r in rows],
                available=[not model.base.unavailable(r) for r in rows],
                physical=[bool(model.physical(r)) for r in rows],
                disconnected=[r['disconnected'] for r in rows],
                observation_invalid=[r.get('observation_invalid', False) for r in rows],
                max_full_contact_projection=max(projections) if projections else None,
                min_initial_contact_margin=min(margins) if margins else None)


def forward_check(x, task, rstar, protocols):
    """Compare all 14 fast, normal scalar and tighter scalar predictions."""
    x = np.asarray(x, dtype=float)
    checkpoint_guard()
    rows = dict(fast=model.predictions(x, task, rstar, protocols))
    checkpoint_guard()
    rows['normal'] = assessment.bounded_predictions(x, task, rstar, protocols, True)
    checkpoint_guard()
    rows['tight'] = tight_predictions(x, task, rstar, protocols)
    expected = [[p['cathode'], p['rate_c'], t['stage'], p['subset'], t['ratio']]
                for p in protocols if not p['excluded_ambiguous'] for t in p['targets']]
    order_matches = len(expected) == 14 and all([_key(r) for r in rr] == expected for rr in rows.values())
    samples = {name: _sample(x, rr) for name, rr in rows.items()}
    comparisons = []
    for left, right in [('fast', 'normal'), ('fast', 'tight'), ('normal', 'tight')]:
        a, b = samples[left], samples[right]
        same = (a['row_keys'] == b['row_keys'] and a['disconnected'] == b['disconnected'] and
                a['observation_invalid'] == b['observation_invalid'])
        physical_same = a['physical'] == b['physical']
        differences = [None if av is None or bv is None else float(abs(av-bv))
                       for av, bv in zip(a['ratios'], b['ratios'])]
        available = all(a['available']) and all(b['available'])
        finite = available and all(v is not None and np.isfinite(v) for v in differences)
        delta = max(differences) if finite and differences else None
        comparisons.append(dict(left=left, right=right, statuses_match=bool(same),
                                physical_statuses_match=bool(physical_same),
                                ratio_differences=differences, max_ratio_difference=delta,
                                passed=bool(order_matches and same and physical_same and finite and
                                            delta <= FORWARD_LIMIT)))
    physical = all(all(s['physical']) for s in samples.values())
    return dict(x=x.tolist(), task=dict(task), rstar=float(rstar), **rows, samples=samples,
                forward_limit=FORWARD_LIMIT, tight_rtol=RTOL, tight_atol=ATOL,
                first_step_rule='min(segment_duration,1e-4)', row_order_matches=bool(order_matches),
                physical_pass=bool(physical), comparisons=comparisons,
                statuses_match=bool(order_matches and all(c['statuses_match'] for c in comparisons)),
                physical_statuses_match=bool(order_matches and all(c['physical_statuses_match']
                                                                  for c in comparisons)),
                max_ratio_difference=max((c['max_ratio_difference'] for c in comparisons
                                          if c['max_ratio_difference'] is not None), default=None),
                passed=bool(physical and all(c['passed'] for c in comparisons)))


def _domain_valid(x, rstar, protocols):
    if not np.all(np.isfinite(x)) or not np.all((x >= model.LOW) & (x <= model.HIGH)):
        return False
    return all(initial_state(x, rstar, p['R_ch'])[0] is not None and domain(p, x[2])['valid']
               for p in protocols if not p['excluded_ambiguous'])


def _stencil(x, column, h, rstar, protocols):
    d = np.eye(5)[column]*WIDTH[column]*h
    for name, offsets, weights in [('central', [-1, 1], [-1., 1.]),
                                   ('forward_second_order', [0, 1, 2], [-3., 4., -1.]),
                                   ('backward_second_order', [0, -1, -2], [3., -4., 1.])]:
        if all(_domain_valid(x+offset*d, rstar, protocols) for offset in offsets):
            return name, offsets, weights, d
    raise ValueError('No in-domain finite-difference stencil for column '+str(column)+' at scaled h='+str(h))


def derivative_check(x, task, rstar, protocols):
    """Qualify eta/A sensitivities using declared scaled steps and inward stencils.

    Return raw predictions, stencils and signed analytic/FD differences. Both
    smallest steps must agree with the analytic column at the .001 scaled
    criterion; those two FD estimates must also agree with one another.
    """
    x = np.asarray(x, dtype=float)
    forward = forward_check(x, task, rstar, protocols)
    result = dict(x=x.tolist(), task=dict(task), forward=forward,
                  scaled_steps=list(SCALED_STEPS), derivative_limit=DERIVATIVE_LIMIT,
                  columns=[], passed=False)
    if not forward['passed']:
        result['derivatives_skipped'] = 'Forward/physical qualification failed'
        return result
    base = forward['samples']['tight']
    for column, name in [(2, 'eta'), (4, 'A')]:
        analytic = np.array([r['derivative'][column] for r in forward['fast']])*WIDTH[column]
        records = []
        for h in SCALED_STEPS:
            checkpoint_guard()
            stencil, offsets, weights, d = _stencil(x, column, h, rstar, protocols)
            samples = [base if offset == 0 else
                       _sample(x+offset*d, tight_predictions(x+offset*d, task, rstar, protocols))
                       for offset in offsets]
            valid = all(s['row_keys'] == base['row_keys'] and all(s['available']) and
                        all(v is not None and np.isfinite(v) for v in s['errors']) for s in samples)
            record = dict(scaled_step=h, coordinate_step=float(WIDTH[column]*h), stencil=stencil,
                          offsets=offsets, weights=weights, denominator=2*h,
                          samples=samples, valid=bool(valid), passed=False)
            if valid:
                fd = sum(weight*np.array(s['errors']) for weight, s in zip(weights, samples))/(2*h)
                difference = fd-analytic
                scale = max(1., float(np.max(np.abs(fd))))
                error = float(np.max(np.abs(difference))/scale)
                record.update(finite_difference=fd.tolist(), analytic=analytic.tolist(),
                              signed_difference=difference.tolist(), normalization=scale,
                              scaled_error=error, passed=bool(np.all(np.isfinite(analytic)) and
                                                             np.isfinite(error) and error <= DERIVATIVE_LIMIT))
            records.append(record)
        smallest = records[-2:]
        stability = None
        if all(record['valid'] for record in smallest):
            a, b = [np.array(record['finite_difference']) for record in smallest]
            difference = b-a
            scale = max(1., float(np.max(np.abs(a))), float(np.max(np.abs(b))))
            error = float(np.max(np.abs(difference))/scale)
            stability = dict(steps=[record['scaled_step'] for record in smallest],
                             signed_difference=difference.tolist(), normalization=scale,
                             scaled_error=error, passed=bool(np.isfinite(error) and error <= DERIVATIVE_LIMIT))
        passed = all(record['passed'] for record in smallest) and stability is not None and stability['passed']
        result['columns'].append(dict(name=name, column=column, analytic=analytic.tolist(),
                                      steps=records, smallest_step_stability=stability, passed=bool(passed)))
    result['passed'] = bool(forward['passed'] and all(c['passed'] for c in result['columns']))
    return result
