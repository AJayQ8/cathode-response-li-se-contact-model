"""Explicit-m finite-height forecast kernel, versioned beside its evidence."""
from pathlib import Path
import importlib.util

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
_spec = importlib.util.spec_from_file_location(
    "frozen_finite22_inputs", PAPER/"finite22/run_finite_height.py")
source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(source)

U = float(source.U)
ZMIN = float(source.ZMIN)
LN10 = float(source.LN10)
LOW = np.array([np.log10(.01), -10., 0., 0., .01])
HIGH = np.array([np.log10(40.), 5., 70., .99, 1.-1e-8])
WIDTH = HIGH-LOW


def initial_state(x, rstar, Rch):
    """m-independent source impedance-to-contact map and its exact Jacobian seed."""
    return source.initial_state(np.asarray(x, dtype=float), rstar, Rch)


def domain_values(x, rstar, protocols):
    b, A = np.asarray(x)[3:]
    return np.array([p["R_ch"]/rstar-b-A+b*A-1e-10
                     for p in protocols if not p["excluded_ambiguous"]], dtype=float)


def advance_m(state, duration, x, cathode, q_start, current, m, independent=False):
    """Advance w=(a/A)^2 and its five fixed-m sensitivities (LSODA)."""
    x = np.asarray(x, dtype=float)
    L, H0 = 10.**x[:2]
    eta, A = x[2], x[4]
    D = U/(L*A*A)
    floor, upper = ZMIN*ZMIN, 1./(A*A)
    if state[0] <= floor:
        return state.copy(), True, 0., 0.
    if duration == 0.:
        return state.copy(), False, 0., 0.

    def rhs(t, y):
        z = max(float(np.sqrt(max(y[0], 0.))), ZMIN/4.)
        bq = source.pressure_increment(cathode, q_start+current*t)
        scale = (2.+eta*bq)/2.
        if scale <= 0.:
            raise ValueError("Noncompressive pressure during integration")
        H = H0*scale**m
        if A*z < 1.:
            g = (1.-A*z)*z**(1.-m)
            dg = -A*z**(1.-m)+(1.-A*z)*(1.-m)*z**(-m)
            direct_A = -2.*H*z**(2.-m)+4.*D*current/A
        else:
            g, dg = 0., 0.
            direct_A = 4.*D*current/A
        F = 2.*H0*scale**m*g-2.*D*current
        if independent:
            return [F]
        Fw = H*dg/z
        direct = np.array([2.*LN10*D*current,
                           2.*LN10*H*g,
                           H0*m*scale**(m-1.)*bq*g,
                           0., direct_A])
        out = np.empty(6)
        out[0] = F
        out[1:] = Fw*y[1:]+direct
        return out

    def event(t, y):
        return y[0]-floor
    event.terminal, event.direction = True, -1
    result = solve_ivp(rhs, (0., float(duration)), state[:1] if independent else state,
                       method="Radau" if independent else "LSODA",
                       rtol=1e-11 if independent else 1e-9,
                       atol=1e-13 if independent else 1e-11,
                       events=event)
    if not result.success or not np.all(np.isfinite(result.y[:, -1])):
        raise RuntimeError(result.message)
    out = state.copy()
    out[:len(result.y)] = result.y[:, -1]
    projection = max(0., float(out[0]-upper))
    if projection > 1e-7*max(1., upper):
        raise RuntimeError("Full-contact projection exceeds inherited tolerance")
    out[0] = min(upper, max(floor, out[0]))
    return out, bool(len(result.t_events[0])), float(result.t[-1]), projection


def predictions_m(x, task, rstar, protocols, m, derivatives=True):
    """All source rows at a fixed explicit m; no source model globals are changed."""
    x = np.asarray(x, dtype=float)
    L, H0 = 10.**x[:2]
    eta, b, A = x[2:]
    rb = b*rstar
    D = U/(L*A*A)
    out = []
    for p in protocols:
        if p["excluded_ambiguous"]:
            continue
        state0, z0, a0 = initial_state(x, rstar, p["R_ch"])
        loading = source.pressure.domain(p, eta)
        targets = p["targets"]
        if state0 is None or not loading["valid"]:
            reason = "Initial contact outside finite-height domain" if state0 is None else loading["reason"]
            for target in targets:
                out.append(dict(cathode=p["cathode"], rate_c=p["rate_c"], stage=target["stage"],
                                subset=p["subset"], observed_ratio=target["ratio"], predicted_ratio=None,
                                error=None, disconnected=False, observation_invalid=True,
                                invalid_reason=reason, initial_z=z0, initial_a=a0, loading=loading,
                                derivative=[0.]*5))
            continue
        start = state0.copy()
        s1, stop1, t1, c1 = advance_m(state0, p["t1"], x, p["cathode"], p["q0"], p["J"], m)
        sr, stopr, tr, cr = ((s1, True, 0., 0.) if stop1 else
                              advance_m(s1, p["rest"], x, p["cathode"], p["q0"]+p["J"]*t1, 0., m))
        s2, stop2, t2, c2 = ((sr, True, 0., 0.) if stopr else
                             advance_m(sr, p["t2"], x, p["cathode"], p["q0"]+p["J"]*t1, p["J"], m))
        first = (s1, stop1, p["J"]*t1) if task["timing"] == "early" else (sr, stopr, p["J"]*t1)
        for target, (state, stopped, charge) in zip(targets, [first, (s2, stop2, p["J"]*(t1+t2))]):
            w = float(state[0])
            z = float(np.sqrt(w))
            a = A*z
            ratio = None if stopped else float((rb+(rstar-rb)/z)/p["R_ch"])
            d_ratio = -((rstar-rb)/(2.*p["R_ch"]*z**3))*state[1:]
            d_ratio[3] += (rstar-rstar/z)/p["R_ch"]
            value = dict(cathode=p["cathode"], rate_c=p["rate_c"], stage=target["stage"],
                         subset=p["subset"], observed_ratio=target["ratio"], predicted_ratio=ratio,
                         error=None if stopped else ratio-target["ratio"], disconnected=bool(stopped),
                         observation_invalid=False, initial_z=z0, initial_a=a0,
                         initial_w=float(start[0]), w=w, z=z, effective_a=a,
                         loading=loading, height_spread_um=L,
                         maximum_remaining_height_um=L*a, mean_remaining_height_um=L*a/2.,
                         initial_mean_remaining_height_um=L*a0/2.,
                         creep_velocity_scale_um_h=H0*L*A**(m+1.), damage_coefficient=D,
                         stripped_charge_mAh_cm2=float(charge),
                         replenishment_scaled=float(w-start[0]+2.*D*charge),
                         required_initial_metal_stock_mAh_cm2=float(w/(2.*D)+charge),
                         nominal_available_metal_stock_mAh_cm2=40./source.U,
                         maximum_w_projection=max(c1, cr, c2),
                         derivative=(d_ratio.tolist() if derivatives and not stopped else [0.]*5))
            out.append(value)
    return out


def unavailable(row):
    return bool(row.get("observation_invalid", False) or row.get("disconnected", False) or
                row.get("predicted_ratio") is None or row.get("error") is None)


def physical_row(row):
    return (not row.get("observation_invalid", False) and 0. < row.get("initial_a", 0.) < 1.
            and 0. < row.get("effective_a", 0.) <= 1.
            and row.get("replenishment_scaled", -np.inf) >= -1e-7
            and row.get("maximum_remaining_height_um", np.inf) <= 40.+1e-10
            and row.get("required_initial_metal_stock_mAh_cm2", np.inf)
            <= row.get("nominal_available_metal_stock_mAh_cm2", -np.inf)+1e-7)


def physical_coordinate_audit(x, task, rstar, protocols, m, fast_rows):
    """Independent (g,K) Radau reconstruction from raw source-time waveforms."""
    import csv, json
    from functools import lru_cache
    mechanics = PAPER/"mechanics3"
    wave_rows = list(csv.DictReader((mechanics/"waveforms.csv").open()))
    summaries = json.loads((mechanics/"waveform_summary.json").read_text())
    raw = {}
    for meta in summaries:
        cathode = meta["cathode"]
        selected = [r for r in wave_rows if r["cathode"] == cathode and r["quantity"] == "pressure_kpa"]
        tt = np.array([float(r["time_h"]) for r in selected])
        pp = np.array([float(r["value"]) for r in selected])
        t0 = float(meta["charge_end_voltage_max"]["time_h"])
        end = float(meta["common_end_h"])
        p0 = float(np.interp(t0, tt, pp))
        inside = (tt > t0) & (tt < end)
        qk = np.r_[0., (tt[inside]-t0)*.12, (end-t0)*.12]
        dk = (np.interp(t0+qk/.12, tt, pp)-p0)/1000.
        raw[cathode] = (qk, dk, tt, pp, t0, p0, end)

    x = np.asarray(x, dtype=float)
    L, H0 = 10.**x[:2]
    eta, b, A = x[2:]
    audited = []
    max_ratio = max_pressure = max_ledger = max_charge = max_replenishment = max_projection = 0.
    by_key = {(r["cathode"], r["rate_c"], r["stage"], r["subset"]): r for r in fast_rows}
    for p in protocols:
        if p["excluded_ambiguous"]:
            continue
        state0, z0, a0 = initial_state(x, rstar, p["R_ch"])
        fast = [r for r in fast_rows if (r["cathode"], r["rate_c"]) == (p["cathode"], p["rate_c"])]
        if state0 is None:
            matches = all(r.get("observation_invalid", False) for r in fast)
            audited.extend(dict(cathode=r["cathode"], rate_c=r["rate_c"], stage=r["stage"],
                                invalid_initial=True, passed=matches) for r in fast)
            continue
        if len(fast) != len(p["targets"]):
            return dict(passed=False, reason="saved row count mismatch", rows=audited)
        qk, dk, tt, pp, t0, p0, endtime = raw[p["cathode"]]
        load_meta = source.pressure.domain(p, eta)
        if not load_meta["valid"]:
            return dict(passed=False, reason=load_meta["reason"], rows=audited)
        state = np.array([L*(1.-a0), 0.], dtype=float)
        volume0 = L*a0*a0/2.
        stopped = False
        total_charge = 0.
        stage_states = []
        projection_size = max((r["maximum_w_projection"] for r in fast), default=0.)
        segments = [(p["t1"], p["J"], p["q0"]),
                    (p["rest"], 0., p["q0"]+p["J"]*p["t1"]),
                    (p["t2"], p["J"], p["q0"]+p["J"]*p["t1"])]

        for index, (duration, current, qstart) in enumerate(segments):
            if not stopped and duration > 0.:
                # Compare source pressure interpolation at both ends and each raw knot.
                qend = qstart+current*duration
                qs = [qstart, qend]
                if current != 0.:
                    qs.extend(float(q) for q in qk if qstart < q < qend)
                for q in qs:
                    raw_delta = float(np.interp(q, qk, dk))
                    model_delta = float(source.pressure_increment(p["cathode"], q))
                    p_err = abs((2.+eta*raw_delta)-(2.+eta*model_delta))
                    max_pressure = max(max_pressure, p_err)
                def rhs(t, y):
                    a = max(min(1.-y[0]/L, 1.), A*ZMIN/4.)
                    q = qstart+current*t
                    delta = float(np.interp(q, qk, dk))
                    pressure = 2.+eta*delta
                    if pressure <= 0.:
                        raise ValueError("Noncompressive independently reconstructed pressure")
                    recovery = H0*L*A**(m+1.)*(pressure/2.)**m
                    return [source.U*current/a-recovery*(1.-a)/a**m,
                            recovery*(1.-a)*a**(1.-m)]
                def event(t, y):
                    return y[0]-L*(1.-A*ZMIN)
                event.terminal, event.direction = True, 1
                result = solve_ivp(rhs, (0., float(duration)), state, method="Radau",
                                   rtol=1e-11, atol=1e-13,
                                   first_step=min(float(duration), 1e-4), events=event)
                if not result.success or not np.all(np.isfinite(result.y[:, -1])):
                    return dict(passed=False, reason="independent Radau failure: "+result.message, rows=audited)
                state = result.y[:, -1]
                elapsed = float(result.t[-1])
                stopped = bool(len(result.t_events[0]))
                total_charge += current*elapsed
            stage_states.append((state.copy(), stopped, total_charge, projection_size))
            if stopped:
                # Record no additional evolution after physical contact loss.
                for _ in range(index+1, len(segments)):
                    stage_states.append((state.copy(), True, total_charge, projection_size))
                break
        if len(stage_states) < 3:
            stage_states.extend([stage_states[-1]]*(3-len(stage_states)))
        first = stage_states[0] if task["timing"] == "early" else stage_states[1]
        obs = [first, stage_states[2]]
        if len(fast) != 2:
            return dict(passed=False, reason="source data do not map to two stored observation stages", rows=audited)
        for r, (physical, physical_stop, charge, projection) in zip(fast, obs):
            a = max(0., min(1., 1.-physical[0]/L))
            volume = L*a*a/2.
            ratio = None if physical_stop else float((b*rstar+A*(1.-b)*rstar/a)/p["R_ch"])
            ratio_delta = None if ratio is None or unavailable(r) else abs(ratio-r["predicted_ratio"])
            ledger = abs(volume-volume0+source.U*charge-physical[1])
            replenishment = abs(physical[1]-r["replenishment_scaled"]*L*A*A/2.)
            charge_delta = abs(charge-r["stripped_charge_mAh_cm2"])
            max_ratio = max(max_ratio, ratio_delta or 0.)
            max_ledger = max(max_ledger, ledger)
            max_replenishment = max(max_replenishment, replenishment)
            max_charge = max(max_charge, charge_delta)
            max_projection = max(max_projection, projection)
            ledger_ok = ledger <= 1e-6*max(1., volume, abs(physical[1]))
            repl_ok = replenishment <= 1e-5*max(1., abs(physical[1]))
            projection_ok = projection <= 1e-7*max(1., 1./(A*A))
            numeric = ((ratio_delta is None or ratio_delta <= 1e-7) and ledger_ok and repl_ok
                       and charge_delta <= 1e-6 and projection_ok
                       and physical_stop == r["disconnected"])
            finite = (not physical_stop and 0. < a <= 1. and L*a <= 40.+1e-10
                      and volume/source.U+charge <= 40./source.U+1e-7 and physical[1] >= -1e-7)
            row = dict(cathode=r["cathode"], rate_c=r["rate_c"], stage=r["stage"],
                       independent_stopped=physical_stop, independent_ratio=ratio,
                       ratio_difference=ratio_delta, contact=a,
                       recession_um=float(physical[0]), replenished_volume_um=float(physical[1]),
                       remaining_volume_um=float(volume), initial_volume_um=float(volume0),
                       independent_stripped_charge_mAh_cm2=float(charge),
                       volume_ledger_error_um=float(ledger), replenishment_difference_um=float(replenishment),
                       pressure_error_mpa=max_pressure, physical_pass=bool(finite), numerical_pass=bool(numeric),
                       passed=bool(finite and numeric))
            audited.append(row)
    maxima = dict(ratio_difference=max_ratio, pressure_interpolation_mpa=max_pressure,
                  volume_ledger_um=max_ledger, stripped_charge_mAh_cm2=max_charge,
                  replenishment_um=max_replenishment, full_contact_projection=max_projection)
    return dict(rows=audited, maxima=maxima,
                passed=(len(audited) == 14 and all(r["passed"] for r in audited)
                        and max_pressure <= 1e-10 and max_ratio <= 1e-7))
