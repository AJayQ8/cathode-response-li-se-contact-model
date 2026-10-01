"""Independent physical-coordinate and saved-fit audit for the stage49 pilot.

No scientific runner is imported, patched, or executed on import.

audit_case(witness, histories, readout_rows, reference_rows) expects:
* witness: id, x=[log10 L, log10 H0, eta, b, A], rstar, setting with
  registration='balance_aligned', timing='late', extraction='ordinary'.
* histories: P-LCO/N-LCO dictionaries returned by prospective27.project.
* readout_rows: P-LCO/N-LCO dictionaries with cathode, template, points,
  re_ohm_cm2, minus_im_ohm_cm2, direct16_re_ohm_cm2,
  direct16_minus_im_ohm_cm2, fitted_ratio, fitted_ratio_true_reference,
  true_ratio_t0, true_ratio_tend, and fit (the complete stage8 fit JSON).
  points and template use sweep35's schemas; optional saved diagnostic fields
  are checked when supplied. The only supported law/order is fixed_q/high_to_low.
* reference_rows: P-LCO/N-LCO dictionaries with cathode, template,
  fitted_reference, and fit (the complete stationary stage8 fit JSON).

audit_swaps(witness, diagonal_histories, crossed_histories) additionally accepts
the two exact decomposition29.project outputs for (P initial, N history) and
(N initial, P history). It audits physical trajectories only, with no EIS or
fitting. The returned stagewise R effects have units ohm cm2; *_over_rstar
effects divide those same absolute resistances by the common rstar, rather
than by the two different initial-cathode resistances. Sign resolution is an
observed numerical-disagreement diagnostic, not an error bound.

All six saved optimization results and their selection/objectives are audited.
Reconstructed initial guesses are reported, but saved fits do not contain the
initial iterates: their actual use is bound by the fitter source hash in the
calling manifest, not independently proved by this audit. No fits are rerun.

Scientific failures return passed=False with tests/error records. Operational
PoolError propagates so it cannot become a scientific failure. Caller owns
input/output hashes and preservation of this report.
"""
from functools import lru_cache
from pathlib import Path
import csv
import json

import numpy as np
from scipy.integrate import solve_ivp
from shared_compute import checkpoint_guard, PoolError


REVISION = Path(__file__).resolve().parent.parent
M, ZMIN, CURRENT, CHARGE, REST_SECONDS = 6.6, 1e-6, 1.2, .30, 1800.
ENDPOINT_LIMIT, SPECTRUM_LIMIT, QUADRATURE_LIMIT = 1e-7, 1e-7, 1e-8
FIT_LOW = np.array([-4., -4., .2, -4., -4., .2, -5., .2])
FIT_HIGH = np.array([3., 10., 1., 3., 10., 1., 3., 1.])
FIT_STARTS = [(6, 1, .55), (7, 30, .8), (8, 1000, .95),
              (6, 1000, .8), (7, 1, .95), (8, 30, .55)]
CATHODES = ('P-LCO', 'N-LCO')


def _read(path):
    return json.loads(path.read_text())


def _need(condition, message):
    if not condition:
        raise ValueError(message)


class Checks:
    def __init__(self):
        self.tests, self.maxima = [], {}

    def flag(self, name, condition, **details):
        self.tests.append(dict(name=name, passed=bool(condition), **details))
        return bool(condition)

    def close(self, name, error, tolerance, maximum=None):
        value = float(error)
        finite = bool(np.isfinite(value))
        if finite and maximum is not None:
            self.maxima[maximum] = max(self.maxima.get(maximum, 0.), value)
        return self.flag(name, finite and value <= tolerance,
                         error=value if finite else str(value), tolerance=tolerance)


@lru_cache(None)
def _source_fit(cathode, rate):
    return _read(REVISION/'impedance8/fits'/f'{cathode[0]}_{rate:g}C_Ch__source__linear__0.json')


@lru_cache(None)
def _wave(cathode):
    rows = list(csv.DictReader((REVISION/'mechanics3/waveforms.csv').open()))
    meta = next(r for r in _read(REVISION/'mechanics3/waveform_summary.json') if r['cathode'] == cathode)
    selected = [r for r in rows if r['cathode'] == cathode and r['quantity'] == 'pressure_kpa']
    tt = np.array([float(r['time_h']) for r in selected])
    pp = np.array([float(r['value']) for r in selected])
    _need(len(tt) == meta['pressure_points'] and np.all(np.diff(tt) > 0), 'Invalid source pressure samples')
    t0 = meta['charge_end_voltage_max']['time_h']
    p0 = float(np.interp(t0, tt, pp))
    return tt, pp, t0, p0, meta['common_end_h']


def _template(cathode, Rch):
    p = _source_fit(cathode, 2.)['best']['parameters']
    fast, slow = p['fast'], p['slow']
    return dict(Rref=Rch, fc=fast['fc_hz'], n=fast['n'], slow_r=slow['r_ohm_cm2'],
                slow_fc=slow['fc_hz'], slow_n=slow['n'], D=p['tail_d_ohm_cm2_at_1_hz'],
                tail_n=p['tail_n'], source='science_lab/papers/cathode_response/revision_20260920/'
                f'impedance8/fits/{cathode[0]}_2C_Ch__source__linear__0.json')


def _admittance(x, frequency):
    z = np.zeros_like(frequency, dtype=complex)
    for offset in (0, 3):
        R, fc, n = 10**x[offset], 10**x[offset+1], x[offset+2]
        Q = 1/(R*(2*np.pi*fc)**n)
        z += 1/(1/R+Q*(2j*np.pi*frequency)**n)
    Qtail = 1/(10**x[6]*(2*np.pi)**x[7])
    return z+1/(Qtail*(2j*np.pi*frequency)**x[7])


def _fixed_q(frequency, resistance, template):
    t = template
    Q = 1/(t['Rref']*(2*np.pi*t['fc'])**t['n'])
    Qslow = 1/(t['slow_r']*(2*np.pi*t['slow_fc'])**t['slow_n'])
    Qtail = 1/(t['D']*(2*np.pi)**t['tail_n'])
    return (1/(1/resistance+Q*(2j*np.pi*frequency)**t['n'])+
            1/(1/t['slow_r']+Qslow*(2j*np.pi*frequency)**t['slow_n'])+
            1/(Qtail*(2j*np.pi*frequency)**t['tail_n']))


def _audit_fit(ans, z, frequency, template, label, checks):
    checkpoint_guard()
    _need(len(ans['starts']) == 6 and [r['start_index'] for r in ans['starts']] == list(range(6)),
          label+': expected the six ordered source-fitter starts')
    _need(ans['task']['topology'] == 'source' and ans['task']['loss'] == 'linear' and
          ans['task']['fraction'] == 0., label+': wrong fitting objective/topology')
    _need(z.shape == frequency.shape and np.all(np.isfinite(z)) and np.all(np.abs(z) > 0),
          label+': invalid supplied spectrum')
    checks.close(label+'/source_fast_reference', abs(ans['source_bulk_li_r']-template['Rref']), 1e-12)
    checks.close(label+'/source_slow_reference', abs(ans['source_cathode_r']-template['slow_r']), 1e-12)
    objectives, reconstructed_starts = [], []
    for index, start in enumerate(ans['starts']):
        x = np.asarray(start['x'], dtype=float)
        _need(x.shape == (8,) and np.all(np.isfinite(x)), label+': invalid fitted coordinates')
        checks.flag(label+f'/bounds/{index}', np.all(x >= FIT_LOW) and np.all(x <= FIT_HIGH))
        e = (_admittance(x, frequency)-z)/np.abs(z)
        residual = np.r_[e.real, e.imag]
        objective = .5*float(residual@residual)
        objectives.append(objective)
        checks.close(label+f'/objective/{index}', abs(objective-start['objective']),
                     1e-9*max(1., objective), 'objective_reconstruction')
        bound_indices = np.flatnonzero(np.minimum(x-FIT_LOW, FIT_HIGH-x)/(FIT_HIGH-FIT_LOW) < 1e-5).tolist()
        checks.flag(label+f'/reported_bounds/{index}', bound_indices == start['bound_indices'])
        checks.flag(label+f'/evaluation_budget/{index}', 0 < start['nfev'] <= 2500)
        logfast, mid, n = FIT_STARTS[index]
        d = max(.02, abs(z[-1].imag)*np.sqrt(frequency[-1])/np.sin(np.pi/4))
        initial = np.array([np.log10(max(.01, template['Rref'])), logfast, n,
                            np.log10(max(.01, template['slow_r'])), np.log10(mid), n, np.log10(d), .5])
        reconstructed_starts.append(np.minimum(FIT_HIGH-1e-8, np.maximum(FIT_LOW+1e-8, initial)).tolist())
    best_index = ans['best_start']
    _need(isinstance(best_index, int) and 0 <= best_index < 6, label+': invalid best-start index')
    best = ans['best']
    checks.flag(label+'/selected_record', best == ans['starts'][best_index])
    checks.flag(label+'/selected_objective', best['objective'] == min(s['objective'] for s in ans['starts']) and
                objectives[best_index] <= min(objectives)+1e-10)
    checks.flag(label+'/selected_success', best['success'])
    x = np.asarray(best['x'])
    predicted = _admittance(x, frequency)
    saved = np.asarray(ans['predicted_re_ohm_cm2'])-1j*np.asarray(ans['predicted_minus_im_ohm_cm2'])
    _need(saved.shape == frequency.shape, label+': saved fitted curve has wrong shape')
    checks.close(label+'/saved_curve', np.max(np.abs(predicted-saved)/np.maximum(1., np.abs(saved))),
                 1e-12, 'fit_curve_reconstruction')
    e = (predicted-z)/np.abs(z)
    checks.close(label+'/rms', abs(float(np.sqrt(np.mean(np.abs(e)**2)))-ans['complex_relative_rms']),
                 1e-12, 'fit_metric_reconstruction')
    checks.close(label+'/max_residual', abs(float(np.max(np.abs(e)))-ans['complex_relative_max']),
                 1e-12, 'fit_metric_reconstruction')
    fast = 0 if x[1] >= x[4] else 3
    slow = 3-fast
    for branch, offset in [('fast', fast), ('slow', slow)]:
        for key, value in [('r_ohm_cm2', 10**x[offset]), ('fc_hz', 10**x[offset+1]), ('n', x[offset+2])]:
            checks.close(label+'/'+branch+'/'+key,
                         abs(value-best['parameters'][branch][key])/max(1., abs(value)), 1e-12)
    checks.close(label+'/branch_order', abs(abs(x[1]-x[4])-best['parameters']['branch_log10_separation']), 1e-12)
    R = float(10**x[fast])
    return R, dict(label=label, six_objectives=objectives, selected_start=best_index,
                   reconstructed_initial_coordinates=reconstructed_starts,
                   initial_guess_provenance='frozen source hash; initial iterates absent from saved fit',
                   selected_success=bool(best['success']),
                   all_start_failures=sum(not s['success'] for s in ans['starts']),
                   selected_fast_resistance=R, selected_complex_relative_rms=float(np.sqrt(np.mean(np.abs(e)**2))))


def _trajectory(witness, cathode, history, checks, history_cathode=None):
    initial_cathode = cathode
    history_cathode = cathode if history_cathode is None else history_cathode
    x, rstar = np.asarray(witness['x'], dtype=float), witness['rstar']
    L, H = 10**x[:2]
    eta, b, A = x[2:]
    U = _read(REVISION/'mechanics3/mechanical_checks.json')['li_equivalent_um_per_mah_cm2']
    Rch = _source_fit(cathode, 2.)['best']['parameters']['rs_plus_fast_r']
    charged = {(r['cathode'], r['rate_c']): r['balance_end_mah_cm2']
               for r in _read(REVISION/'mechanics3/fullcell_charge_history.json') if r['segment'] == 'charge'}
    q0 = charged[(history_cathode, .2)]-charged[(history_cathode, 2.)]
    tt, pp, t0, p0, end = _wave(history_cathode)
    if history_cathode != initial_cathode:
        cathode = initial_cathode+'/history_'+history_cathode
    qmax = (end-t0)*.12
    _need(q0 >= -1e-10 and q0+CHARGE <= qmax+1e-10, 'Prospective pressure source support exceeded')

    def pressure(q):
        _need(q >= -1e-10 and q <= qmax+1e-10, 'Pressure evaluation outside source support')
        return 2+eta*(float(np.interp(t0+q/.12, tt, pp))-p0)/1000

    qq = (tt-t0)*.12
    knots = np.unique(np.r_[q0, qq[(qq > q0) & (qq < q0+CHARGE)], q0+CHARGE])
    pressures = np.array([pressure(q) for q in knots])
    _need(np.min(pressures) > 0., 'Noncompressive pressure in prospective interval')
    checks.close(cathode+'/source_Rch', abs(history['R_ch']-Rch), 1e-12)
    checks.close(cathode+'/source_q0', abs(history['q0']-q0), 1e-12)
    checks.flag(cathode+'/source_charged_rate', history['source_charged_rate_c'] == 2.)
    checks.flag(cathode+'/loading_valid', history['loading']['valid'])
    for key, value in [('minimum_pressure_mpa', float(pressures.min())),
                       ('maximum_pressure_mpa', float(pressures.max()))]:
        checks.close(cathode+'/'+key, abs(value-history['loading'][key]), 1e-10, 'source_pressure_difference')
    rb, C = b*rstar, A*(1-b)*rstar
    a0 = C/(Rch-rb)
    _need(0 < a0 < 1, 'Invalid prospective initial contact')
    V0 = L*a0*a0/2
    initial = np.array([L*(1-a0), 0.])
    velocity = H*L*A**(M+1)
    floor = L*(1-A*ZMIN)

    def segment(y0, duration, current, charge_start):
        checkpoint_guard()
        def rhs(t, y):
            a = max(min(1-y[0]/L, 1.), A*ZMIN/4)
            p = pressure(charge_start+current*t)
            _need(p > 0, 'Noncompressive source pressure')
            recovery = velocity*(p/2)**M
            return [U*current/a-recovery*(1-a)/a**M,
                    recovery*(1-a)*a**(1-M)]
        def event(t, y):
            return y[0]-floor
        event.terminal, event.direction = True, 1
        sol = solve_ivp(rhs, (0., duration), y0, method='Radau', dense_output=True,
                        rtol=1e-12, atol=1e-14, first_step=min(duration, 1e-4), events=event)
        _need(sol.success and np.all(np.isfinite(sol.y)), 'Independent physical integration failed: '+sol.message)
        _need(not len(sol.t_events[0]), 'Independent prospective trajectory disconnects')
        return sol

    discharge = segment(initial, CHARGE/CURRENT, CURRENT, q0)
    rest = segment(discharge.y[:, -1], REST_SECONDS/3600, 0., q0+CHARGE)

    def observe(state, charge):
        a_raw = 1-np.asarray(state)[0]/L
        a = np.clip(a_raw, 0., 1.)
        K = np.asarray(state)[1]
        V = L*a*a/2
        _need(np.all(a > 0), 'Disconnected independent observation')
        ledger = np.abs(V-V0+U*charge-K)
        physical = ((a_raw > A*ZMIN) & (a_raw <= 1+1e-12) & (L*a <= 40+1e-10) &
                    (V+U*charge <= 40+U*1e-7) & (K >= -1e-7))
        numerical = ledger <= 1e-6*np.maximum(1., np.maximum(V, np.abs(K)))
        return dict(contact=a, resistance=rb+C/a, ratio=(rb+C/a)/Rch,
                    ledger=ledger, physical=physical, numerical=numerical, K=K, volume=V)

    rows = history['rows']
    _need([r['stage'] for r in rows] == ['end_discharge', 'after_30_min_rest'], 'Wrong projection endpoint order')
    endpoints = []
    for record, state in zip(rows, [discharge.y[:, -1], rest.y[:, -1]]):
        o = observe(state, CHARGE)
        checks.flag(cathode+'/'+record['stage']+'/availability', not record['stopped'] and
                    record['numerical_pass'] and record['physical_pass'])
        checks.flag(cathode+'/'+record['stage']+'/physical', np.all(o['physical']))
        checks.flag(cathode+'/'+record['stage']+'/ledger', np.all(o['numerical']))
        for key in ['ratio', 'independent_ratio']:
            _need(record[key] is not None, 'Saved prospective endpoint is unavailable')
            checks.close(cathode+'/'+record['stage']+'/'+key, abs(float(o['ratio'])-record[key]),
                         ENDPOINT_LIMIT, 'projection_endpoint_ratio')
        checks.close(cathode+'/'+record['stage']+'/charge', abs(record['stripped_charge_mAh_cm2']-CHARGE), 1e-6)
        checks.close(cathode+'/'+record['stage']+'/independent_charge',
                     abs(record['independent_stripped_charge_mAh_cm2']-CHARGE), 1e-6)
        checks.close(cathode+'/'+record['stage']+'/contact', abs(float(o['contact'])-record['a']), ENDPOINT_LIMIT)
        checks.close(cathode+'/'+record['stage']+'/recession', abs(float(state[0])-record['recession_um']), 1e-5)
        checks.close(cathode+'/'+record['stage']+'/remaining_volume',
                     abs(float(o['volume'])-record['remaining_volume_um']), 1e-5)
        checks.flag(cathode+'/'+record['stage']+'/projection_nonnegative', record['projection'] >= 0.)
        checks.close(cathode+'/'+record['stage']+'/projection', abs(record['projection']),
                     1e-7*max(1., 1/(A*A)))
        checks.close(cathode+'/'+record['stage']+'/replenishment', abs(float(o['K'])-record['replenished_volume_um']),
                     1e-5*max(1., abs(float(o['K']))), 'replenishment_difference_um')
        checks.close(cathode+'/'+record['stage']+'/initial_volume', abs(record['initial_volume_um']-V0), 1e-10)
        checks.close(cathode+'/'+record['stage']+'/ledger_error', float(o['ledger']),
                     1e-6*max(1., float(o['volume']), abs(float(o['K']))), 'volume_ledger_error_um')
        checks.close(cathode+'/'+record['stage']+'/resistance',
                     abs(float(o['resistance'])-record['resistance'])/Rch,
                     ENDPOINT_LIMIT, 'projection_endpoint_ratio')
        endpoints.append(dict(stage=record['stage'], ratio=float(o['ratio']),
                              resistance=float(o['resistance']), contact=float(o['contact']),
                              recession_um=float(state[0]), replenished_volume_um=float(state[1]),
                              remaining_volume_um=float(o['volume'])))
    # Inspect accepted integration nodes as well as the separately checked readout nodes.
    for name, sol, charges in [('discharge', discharge, CURRENT*discharge.t),
                               ('rest', rest, np.full(rest.t.shape, CHARGE))]:
        o = observe(sol.y, charges)
        checks.flag(cathode+'/'+name+'/trajectory_physical', np.all(o['physical']))
        checks.flag(cathode+'/'+name+'/trajectory_ledger', np.all(o['numerical']))
    return rest, observe, dict(cathode=initial_cathode, initial_cathode=initial_cathode,
                              history_cathode=history_cathode, R_ch=Rch, q0=q0, initial_contact=a0,
                              initial_volume_um=V0, final_pressure_mpa=pressure(q0+CHARGE),
                              endpoints=endpoints, nfev_discharge=discharge.nfev, nfev_rest=rest.nfev)


def _effects(values):
    """Stage29 symmetric two-factor decomposition, using absolute resistance."""
    pp, pn, np_, nn = [values[k] for k in ('PP', 'PN', 'NP', 'NN')]
    hp, hn, ip, inn = pp-pn, np_-nn, pp-np_, pn-nn
    return dict(fixed_initial_P_effect_R=hp, fixed_initial_N_effect_R=hn,
                symmetricmean_R=(hp+hn)/2,
                fixed_history_P_initial_effect_R=ip, fixed_history_N_initial_effect_R=inn,
                symmetric_initial_contribution_R=(ip+inn)/2,
                diagonalcontrast_R=pp-nn, interaction_R=hp-hn)


def audit_swaps(witness, diagonal_histories, crossed_histories):
    """Independently audit the four histories and stage29 decomposition.

    Reuses the supplied diagonal production histories. Returns two stage rows
    containing absolute R and common-rstar effects, each of three reconstruction
    routes, their pairwise differences, and fixed-start sign diagnostics.
    Scientific failures remain recorded; operational PoolError propagates.
    """
    checkpoint_guard()
    checks, outputs, stages, errors = Checks(), [], [], []
    result = dict(schema=1, witness_id=witness.get('id'), tests=checks.tests,
                  maxima=checks.maxima, histories=outputs, stages=stages, errors=errors,
                  passed=False, resistance_units='ohm cm2', over_rstar_units='dimensionless',
                  off_diagonal_spectral_fitting=False, numerical_sign_diagnostic_is_bound=False)
    try:
        _need(witness['setting'] == dict(registration='balance_aligned', timing='late', extraction='ordinary'),
              'Auditor supports only the predeclared stage49 interpretation')
        x = np.asarray(witness['x'], dtype=float)
        low = np.array([np.log10(.01), -10., 0., 0., .01])
        high = np.array([np.log10(40.), 5., 70., .99, 1.-1e-8])
        _need(x.shape == (5,) and np.all(np.isfinite(x)) and np.all(x >= low) and np.all(x <= high),
              'Witness outside original bounds')
        _need(len(diagonal_histories) == 2 and {h['cathode'] for h in diagonal_histories} == set(CATHODES),
              'Wrong diagonal cathode set')
        _need(len(crossed_histories) == 2 and
              {(h['initial_cathode'], h['history_cathode']) for h in crossed_histories} ==
              {('P-LCO', 'N-LCO'), ('N-LCO', 'P-LCO')}, 'Wrong crossed-history pairs')
        rstar = min(_source_fit(c, rate)['best']['parameters']['rs_plus_fast_r']
                    for c in CATHODES for rate in [.2, 2.])
        _need(np.isfinite(witness['rstar']) and witness['rstar'] > 0., 'Invalid rstar')
        checks.close('source_rstar', abs(witness['rstar']-rstar), 1e-12)
        baseline = (_source_fit('P-LCO', 2.)['best']['parameters']['rs_plus_fast_r']-
                    _source_fit('N-LCO', 2.)['best']['parameters']['rs_plus_fast_r'])
        result.update(baseline_startcontrast_R=baseline, baseline_startcontrast_over_rstar=baseline/rstar)
        cells = {h['cathode'][0]*2: (h['cathode'], h['cathode'], h) for h in diagonal_histories}
        cells.update({h['initial_cathode'][0]+h['history_cathode'][0]:
                      (h['initial_cathode'], h['history_cathode'], h) for h in crossed_histories})
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(dict(phase='input', error=type(exc).__name__+': '+str(exc)))
        return result
    audited = {}
    for key in ('PP', 'PN', 'NP', 'NN'):
        checkpoint_guard()
        initial, history_cathode, saved = cells[key]
        try:
            _need(saved['available'], 'Saved trajectory unavailable')
            _, _, out = _trajectory(witness, initial, saved, checks, history_cathode)
            if 'initial_a' in saved:
                checks.close(key+'/initial_contact', abs(saved['initial_a']-out['initial_contact']), 1e-12)
                checks.close(key+'/initial_z', abs(saved['initial_z']-out['initial_contact']/x[4]), 1e-12)
            out['cell'] = key
            outputs.append(out)
            audited[key] = out
        except PoolError:
            raise
        except (ValueError, RuntimeError, AssertionError, FloatingPointError, OverflowError,
                KeyError, TypeError, IndexError) as exc:
            errors.append(dict(cell=key, error=type(exc).__name__+': '+str(exc)[:400]))
    if len(audited) == 4:
        for stage in ('end_discharge', 'after_30_min_rest'):
            saved_rows = {k: next(r for r in cells[k][2]['rows'] if r['stage'] == stage) for k in cells}
            route_R = dict(
                primary={k: float(saved_rows[k]['resistance']) for k in cells},
                saved_physical={k: float(saved_rows[k]['independent_ratio']*cells[k][2]['R_ch']) for k in cells},
                independent_audit={k: next(r['resistance'] for r in audited[k]['endpoints'] if r['stage'] == stage)
                                   for k in cells})
            effects = {route: _effects(values) for route, values in route_R.items()}
            for route, effect in effects.items():
                checks.close(stage+'/'+route+'/decomposition_identity',
                             abs(effect['diagonalcontrast_R']-effect['symmetricmean_R']-
                                 effect['symmetric_initial_contribution_R']), 1e-12, 'decomposition_residual_R')
                checks.close(stage+'/'+route+'/interaction_identity',
                             abs(effect['interaction_R']-effect['fixed_history_P_initial_effect_R']+
                                 effect['fixed_history_N_initial_effect_R']), 1e-12)
            pairs = [('primary', 'saved_physical'), ('primary', 'independent_audit'),
                     ('saved_physical', 'independent_audit')]
            differences = {a+'__'+b: {k: abs(effects[a][k]-effects[b][k]) for k in effects[a]}
                           for a, b in pairs}
            maximum = {k: max(d[k] for d in differences.values()) for k in effects['primary']}
            checks.maxima['effect_route_disagreement_R'] = max(
                checks.maxima.get('effect_route_disagreement_R', 0.), *maximum.values())
            checks.maxima['effect_route_disagreement_over_rstar'] = checks.maxima['effect_route_disagreement_R']/rstar
            signs = {}
            for k in ('fixed_initial_P_effect_R', 'fixed_initial_N_effect_R', 'symmetricmean_R'):
                values = [e[k] for e in effects.values()]
                sign = 1 if all(v > 0 for v in values) else -1 if all(v < 0 for v in values) else 0
                signs[k] = dict(sign=sign, minimum_absolute_effect_R=min(abs(v) for v in values),
                                observed_max_disagreement_R=maximum[k],
                                resolved_against_observed_disagreement=bool(sign and min(abs(v) for v in values) > maximum[k]))
            record = dict(stage=stage, baseline_startcontrast_R=baseline,
                          baseline_startcontrast_over_rstar=baseline/rstar,
                          R_PP=route_R['independent_audit']['PP'], R_PN=route_R['independent_audit']['PN'],
                          R_NP=route_R['independent_audit']['NP'], R_NN=route_R['independent_audit']['NN'],
                          **effects['independent_audit'])
            record.update({k[:-2]+'_over_rstar': v/rstar for k, v in effects['independent_audit'].items()})
            record.update(route_resistances_R=route_R, route_effects_R=effects,
                          route_effects_over_rstar={route: {k[:-2]+'_over_rstar': v/rstar for k, v in effect.items()}
                                                   for route, effect in effects.items()},
                          pairwise_effect_disagreements_R=differences,
                          max_effect_disagreement_R=maximum,
                          max_effect_disagreement_over_rstar={k[:-2]+'_over_rstar': v/rstar for k, v in maximum.items()},
                          fixed_start_sign_diagnostics=signs)
            stages.append(record)
    result['passed'] = bool(not errors and len(audited) == 4 and all(t['passed'] for t in checks.tests))
    for row in stages:
        row['available'] = result['passed']
        row['sign_diagnostics_valid'] = result['passed']
    return result


def audit_case(witness, histories, readout_rows, reference_rows):
    """Audit one frozen witness; return tests/maxima without writing artifacts."""
    checkpoint_guard()
    checks, outputs, fit_audits, errors = Checks(), [], [], []
    result = dict(schema=1, witness_id=witness.get('id'), law='fixed_q', order='high_to_low',
                  tests=checks.tests, maxima=checks.maxima, histories=outputs, fit_audits=fit_audits,
                  errors=errors, passed=False, instrument_validated=False)
    try:
        _need(witness['setting'] == dict(registration='balance_aligned', timing='late', extraction='ordinary'),
              'Auditor supports only the predeclared stage49 interpretation')
        x = np.asarray(witness['x'], dtype=float)
        _need(x.shape == (5,) and np.all(np.isfinite(x)), 'Invalid witness coordinates')
        low = np.array([np.log10(.01), -10., 0., 0., .01])
        high = np.array([np.log10(40.), 5., 70., .99, 1.-1e-8])
        _need(np.all(x >= low) and np.all(x <= high), 'Witness outside original bounds')
        for rows, name in [(histories, 'projection'), (readout_rows, 'readout'), (reference_rows, 'reference')]:
            _need(len(rows) == 2 and {r['cathode'] for r in rows} == set(CATHODES), 'Wrong '+name+' cathode set')
        rstar = min(_source_fit(c, rate)['best']['parameters']['rs_plus_fast_r'] for c in CATHODES for rate in [.2, 2.])
        checks.close('source_rstar', abs(witness['rstar']-rstar), 1e-12)
        frequency = 10**np.array(_read(REVISION/'impedance8/spectra.json')['frequency_log10_hz'])
        _need(len(frequency) == 68 and np.all(np.diff(frequency) < 0), 'Wrong primary source frequency grid')
        schedule, clock = [], 0.
        for k, f in enumerate(frequency):
            start, stop = clock+.5, clock+.5+3/f
            schedule.append((clock, start, stop))
            clock = stop
        result['duration_seconds'] = float(clock)
        _need(clock <= REST_SECONDS, 'Readout exceeds independently integrated rest')
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(dict(phase='input', error=type(exc).__name__+': '+str(exc)))
        return result
    nodes, weights = np.polynomial.legendre.leggauss(16)
    for cathode in CATHODES:
        checkpoint_guard()
        try:
            history = next(h for h in histories if h['cathode'] == cathode)
            row = next(h for h in readout_rows if h['cathode'] == cathode)
            reference = next(h for h in reference_rows if h['cathode'] == cathode)
            _need(history['available'], 'Saved projection history unavailable')
            rest, observe, out = _trajectory(witness, cathode, history, checks)
            template = _template(cathode, out['R_ch'])
            checks.flag(cathode+'/readout_template', row['template'] == template)
            checks.flag(cathode+'/reference_template', reference['template'] == template)
            stationary = _fixed_q(frequency, template['Rref'], template)
            Rref, fit_record = _audit_fit(reference['fit'], stationary, frequency, template, cathode+'/reference', checks)
            fit_audits.append(fit_record)
            checks.close(cathode+'/fitted_reference', abs(Rref-reference['fitted_reference']), 1e-12)
            checks.close(cathode+'/reference_recovery', abs(Rref/template['Rref']-1.), 1e-6)
            checks.close(cathode+'/reference_spectrum_recovery', reference['fit']['complex_relative_max'], 1e-7)
            _need(len(row['points']) == 68, 'Wrong number of readout points')
            reconstructed, point_checks, max_rate = [], [], 0.
            for k, (point, (settle, start, stop)) in enumerate(zip(row['points'], schedule)):
                if k % 16 == 0:
                    checkpoint_guard()
                checks.flag(cathode+f'/point_order/{k}', point['frequency_index'] == k and
                            point['acquisition_rank'] == k and point['frequency_hz'] == frequency[k])
                checks.close(cathode+f'/schedule/{k}', max(abs(point['settle_start_s']-settle),
                             abs(point['measure_start_s']-start), abs(point['measure_end_s']-stop)),
                             1e-10, 'schedule_seconds')
                seconds = (start+stop)/2+nodes*(stop-start)/2
                o = observe(rest.sol(seconds/3600), CHARGE)
                ends = observe(rest.sol(np.array([start, stop])/3600), CHARGE)
                checks.flag(cathode+f'/readout_physical/{k}', np.all(o['physical']) and np.all(ends['physical']))
                checks.flag(cathode+f'/readout_ledger/{k}', np.all(o['numerical']) and np.all(ends['numerical']))
                reconstructed.append(np.dot(weights, _fixed_q(frequency[k], o['resistance'], template))/2)
                delta = float(np.max(np.abs(ends['resistance']-point['resistance_endpoints']))/template['Rref'])
                checks.close(cathode+f'/point_endpoint/{k}', delta, ENDPOINT_LIMIT, 'readout_endpoint_ratio')
                checks.close(cathode+f'/point_contact/{k}',
                             np.max(np.abs(ends['contact']-point['contact_endpoints'])), ENDPOINT_LIMIT)
                checks.close(cathode+f'/point_mean/{k}', abs(float(np.dot(weights, o['resistance'])/2)-
                             point['mean_instantaneous_resistance'])/template['Rref'], ENDPOINT_LIMIT)
                gamma = 10**x[1]*x[4]**(M+1)*(out['final_pressure_mpa']/2)**M
                C = x[4]*(1-x[3])*rstar
                rate = C/ends['contact']**2*gamma*(1-ends['contact'])*ends['contact']**(-M)/3600
                fractional = rate/ends['resistance']/frequency[k]
                max_rate = max(max_rate, float(np.max(fractional)))
                checks.close(cathode+f'/point_change_rate/{k}', np.max(np.abs(fractional-
                             point['fractional_R_change_per_cycle_at_endpoints'])), 1e-7)
                point_checks.append(dict(index=k, normalized_endpoint_difference=delta))
            z = np.asarray(row['re_ohm_cm2'])-1j*np.asarray(row['minus_im_ohm_cm2'])
            z16 = np.asarray(row['direct16_re_ohm_cm2'])-1j*np.asarray(row['direct16_minus_im_ohm_cm2'])
            actual = np.asarray(reconstructed)
            _need(z.shape == z16.shape == actual.shape == frequency.shape, 'Wrong saved spectrum shape')
            discrepancy = float(np.max(np.abs(z-actual)/np.abs(actual)))
            discrepancy16 = float(np.max(np.abs(z16-actual)/np.abs(actual)))
            quadrature = float(np.max(np.abs(z-z16)/np.abs(z16)))
            checks.close(cathode+'/physical_spectrum8', discrepancy, SPECTRUM_LIMIT, 'physical_coordinate_spectrum')
            checks.close(cathode+'/physical_spectrum16', discrepancy16, SPECTRUM_LIMIT, 'physical_coordinate_spectrum')
            checks.close(cathode+'/quadrature8_16', quadrature, QUADRATURE_LIMIT, 'quadrature_admittance')
            if 'quadrature_direct_error' in row:
                checks.close(cathode+'/reported_quadrature_error', abs(quadrature-row['quadrature_direct_error']), 1e-12)
            Rfit, fit_record = _audit_fit(row['fit'], z, frequency, template, cathode+'/dynamic', checks)
            fit_audits.append(fit_record)
            normalized = Rfit/Rref
            checks.close(cathode+'/normalized_fit', abs(normalized-row['fitted_ratio']), 1e-12)
            checks.close(cathode+'/true_reference_fit', abs(Rfit/template['Rref']-row['fitted_ratio_true_reference']), 1e-12)
            zero = out['endpoints'][0]['ratio']
            end = float(observe(rest.sol(clock/3600), CHARGE)['ratio'])
            checks.close(cathode+'/readout_t0', abs(zero-row['true_ratio_t0']), ENDPOINT_LIMIT, 'readout_endpoint_ratio')
            checks.close(cathode+'/readout_tend', abs(end-row['true_ratio_tend']), ENDPOINT_LIMIT, 'readout_endpoint_ratio')
            if 'max_fractional_R_change_per_cycle' in row:
                checks.close(cathode+'/reported_max_change_rate', abs(max_rate-row['max_fractional_R_change_per_cycle']), 1e-7)
            out.update(fitted_ratio=normalized, readout_endpoint_ratio=end,
                       physical_spectrum_relative_error=discrepancy, quadrature_error=quadrature,
                       max_fractional_R_change_per_cycle=max_rate, point_checks=point_checks)
            outputs.append(out)
        except PoolError:
            raise
        except (ValueError, RuntimeError, AssertionError, FloatingPointError, OverflowError,
                KeyError, TypeError, IndexError) as exc:
            errors.append(dict(cathode=cathode, error=type(exc).__name__+': '+str(exc)[:400]))
    if len(outputs) == 2:
        P, N = outputs
        result['independent_contrasts'] = dict(
            end_discharge=P['endpoints'][0]['ratio']-N['endpoints'][0]['ratio'],
            after_30_min_rest=P['endpoints'][1]['ratio']-N['endpoints'][1]['ratio'],
            primary_fitted_readout=P['fitted_ratio']-N['fitted_ratio'])
    result['passed'] = bool(not errors and len(outputs) == 2 and all(t['passed'] for t in checks.tests))
    return result
