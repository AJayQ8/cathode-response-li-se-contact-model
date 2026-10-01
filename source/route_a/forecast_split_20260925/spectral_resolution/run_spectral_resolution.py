"""Deterministic zero-contrast spectral alternatives for the Route A forecast.

The primary source fitter is reused for all unconstrained recovery fits.  The
constrained profile uses the same model, residual, analytic Jacobian, loss,
bounds, starts and high-precision stage35 tolerances.  Its frequency variables
are reparameterized so the prescribed resistance remains in the faster branch
with at least the frozen 0.3-decade separation.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import time

import numpy as np
from shared_compute import checkpoint_guard

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REV = BASE.parent
ROOT = HERE.parents[5]
SWEEP = REV / "sweep35"
IM8 = REV / "impedance8"
TRANSFER49 = REV / "transfer49"
EXT = BASE / "post50_extension"
UPSTREAM_SPLIT_MANIFEST = BASE / "manifest.json"
UPSTREAM_TRANSFER_MANIFEST = TRANSFER49 / "continuation_v2_manifest.json"
UPSTREAM_EXTENSION_MANIFEST = EXT / "continuation_v2_manifest.json"
FREQUENCY_SOURCE = IM8 / "spectra.json"
SOURCE_CIRCUIT = IM8 / "run_circuit_probe.py"
STAGE35_RUNNER = SWEEP / "run_sweep.py"
STAGE35_REPAIR = SWEEP / "PRECISION_REPAIR.md"
PLAN = HERE / "PLAN.md"
AUDIT_SCRIPT = HERE / "audit_spectral_resolution.py"
MANIFEST = HERE / "manifest.json"
OUT = HERE / "cases"
RECOVERY_OUT = HERE / "recovery_fits"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


circuit = module("spectral_resolution_source_circuit", SOURCE_CIRCUIT)
circuit.OUT = RECOVERY_OUT
_least_squares = circuit.least_squares


def precise_least_squares(*args, **kwargs):
    kwargs.update(ftol=1e-13, xtol=1e-13, gtol=1e-13)
    return _least_squares(*args, **kwargs)


circuit.least_squares = precise_least_squares

FREQ_SEP_MIN = 0.3
FREQ_LOG_MIN = -4.0
FREQ_LOG_MAX = 10.0
SLOW_LOG_MAX = FREQ_LOG_MAX - FREQ_SEP_MIN
PROFILE_LO = np.array([0.0, 0.0, 0.2, -4.0, 0.2, -5.0, 0.2])
PROFILE_HI = np.array([SLOW_LOG_MAX - FREQ_LOG_MIN, 1.0, 1.0, 3.0, 1.0, 3.0, 1.0])
RECOVERY_TOLERANCE = 1e-7
RATIO_TOLERANCE = 1e-6
PLAN_PROTOCOL = {
    "frequency_source": "impedance8/spectra.json",
    "frequency_count": 68,
    "sweep_order": "high_to_low",
    "fast_element_hypothesis": "fixed-Q",
    "history_pressure_or_trajectory_changed": False,
    "kinetic_parameters_refit": False,
    "stationary_pre_resistance_treated_as_exact": True,
    "optimizer_ftol_xtol_gtol": 1e-13,
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def best_fit(fit_doc):
    if "fit" in fit_doc and "inputs" in fit_doc:
        return fit_doc["fit"]
    return fit_doc


def fit_path_for_original(relative_path):
    return SWEEP / relative_path


def fit_path_for_extension(relative_path):
    if relative_path.startswith("spectral_fits/"):
        return TRANSFER49 / relative_path
    if relative_path.startswith("fits/"):
        return EXT / relative_path
    raise ValueError(f"Unknown extension fit path: {relative_path}")


def curve_from_fit(fit):
    return np.asarray(fit["predicted_re_ohm_cm2"], dtype=float) - 1j * np.asarray(
        fit["predicted_minus_im_ohm_cm2"], dtype=float
    )


def spectrum_from_record(record):
    return np.asarray(record["re_ohm_cm2"], dtype=float) - 1j * np.asarray(
        record["minus_im_ohm_cm2"], dtype=float
    )


def extract_x(fit):
    p = fit["best"]["parameters"]
    fast, slow = p["fast"], p["slow"]
    return np.array([
        math.log10(fast["r_ohm_cm2"]), math.log10(fast["fc_hz"]), fast["n"],
        math.log10(slow["r_ohm_cm2"]), math.log10(slow["fc_hz"]), slow["n"],
        math.log10(p["tail_d_ohm_cm2_at_1_hz"]), p["tail_n"],
    ], dtype=float)


def fitted_fast(fit):
    return float(fit["best"]["parameters"]["fast"]["r_ohm_cm2"])


def input_metadata(case_id, cohort, source_case, split_case, p_record, n_record, setting):
    def concise(record):
        return {
            "dynamic_spectrum": rel(record["dynamic_source_file"]),
            "selected_fit": rel(record["source_fit_file"]),
            "stationary_reference_fit": rel(record["source_reference_file"]),
            "R_pre_fitted": float(record["r_pre"]),
            "R_input_bulk": float(record["r_bulk"]),
            "R_input_slow": float(record["r_slow"]),
            "primary_ratio": float(record["ratio"]),
            "loss": record["loss"],
            "source_fit_file": rel(record["source_fit_file"]),
            "source_template_file": rel(record["source_template_file"]),
            "source_case_cell": record["source_case_cell"],
        }
    return {
        "id": case_id,
        "cohort": cohort,
        "setting": setting,
        "source_case": rel(source_case),
        "forecast_split_case": rel(split_case) if split_case else None,
        "cells": {"P": concise(p_record), "N": concise(n_record)},
    }


def build_original_case(witness, split_case):
    wid = witness["id"]
    source_case = SWEEP / "cases" / f"{wid}_fixed_q_high_to_low.json"
    data = read(source_case)
    histories = {h["cathode"]: h for h in data["histories"]}
    records = {}
    for label, cathode, split_cell in [
        ("P", "P-LCO", "PP"),
        ("N", "N-LCO", "NN"),
    ]:
        h = histories[cathode]
        fit_file = fit_path_for_original(h["fit_path"])
        reference_file = fit_path_for_original(h["reference_fit_path"])
        fit = best_fit(read(fit_file))
        reference = best_fit(read(reference_file))
        primary_ratio = fitted_fast(fit) / fitted_fast(reference)
        split_ratio = split_case["cells"][split_cell]["fitted_ratio"]
        require(h["selected_fit_success"], f"Unsuccessful primary fit {wid}/{cathode}")
        require(abs(primary_ratio - h["fitted_ratio"]) <= 1e-12,
                f"stage35 primary ratio mismatch {wid}/{cathode}")
        require(abs(split_ratio - h["fitted_ratio"]) <= 1e-12,
                f"forecast split diagonal mismatch {wid}/{cathode}")
        z = spectrum_from_record(h)
        z_fit = curve_from_fit(fit)
        require(z.size == 68 and z_fit.size == 68, f"Unexpected spectrum length {wid}/{label}")
        task_loss = fit["task"]["loss"]
        records[label] = {
            "z_dynamic": z,
            "z_static": z_fit,
            "fit": fit,
            "reference": reference,
            "r_pre": fitted_fast(reference),
            "r_bulk": float(h["template"]["Rref"]),
            "r_slow": float(h["template"]["slow_r"]),
            "ratio": float(h["fitted_ratio"]),
            "loss": task_loss,
            "source_case_cell": split_cell,
            "source_fit_file": fit_file,
            "source_reference_file": reference_file,
            "source_template_file": IM8 / "fits" / f"{cathode[0]}_2C_Ch__source__{task_loss}__0.json",
            "dynamic_source_file": source_case,
        }
    records["metadata"] = input_metadata(
        wid, "original50", source_case, split_case["_path"], records["P"], records["N"], witness["setting"]
    )
    return records


def build_extension_case(case_id):
    source_case = EXT / "cases" / f"{case_id}.json"
    data = read(source_case)
    records = {}
    for label, cell_name in [("P", "PP"), ("N", "NN")]:
        cell = data["cells"][cell_name]
        fit_file = fit_path_for_extension(cell["fit_path"])
        fit_doc = read(fit_file)
        fit = best_fit(fit_doc)
        template = cell["template"]
        cathode = "P-LCO" if label == "P" else "N-LCO"
        reference_control = next(
            row for row in read(SWEEP / "controls.json")["records"]
            if row["extraction"] == "ordinary" and row["cathode"] == cathode
        )
        reference_file = fit_path_for_original(reference_control["fit_path"])
        reference = best_fit(read(reference_file))
        r_pre = float(cell["fitted_reference"])
        require(abs(r_pre - fitted_fast(reference)) <= 1e-10,
                f"Extension stationary reference mismatch {case_id}/{label}")
        require(abs(r_pre - reference_control["fitted_reference"]) <= 1e-10,
                f"Extension control normalization mismatch {case_id}/{label}")
        spectrum = cell["spectrum"]
        z = spectrum_from_record(spectrum)
        z_fit = curve_from_fit(fit)
        primary_ratio = fitted_fast(fit) / r_pre
        require(abs(primary_ratio - cell["fitted_ratio"]) <= 1e-12,
                f"Extension primary ratio mismatch {case_id}/{label}")
        require(z.size == 68 and z_fit.size == 68, f"Unexpected extension spectrum length {case_id}/{label}")
        require(fit["best"]["success"], f"Unsuccessful extension primary fit {case_id}/{label}")
        records[label] = {
            "z_dynamic": z,
            "z_static": z_fit,
            "fit": fit,
            "reference": reference,
            "r_pre": r_pre,
            "r_bulk": float(template["Rref"]),
            "r_slow": float(template["slow_r"]),
            "ratio": float(cell["fitted_ratio"]),
            "loss": fit["task"]["loss"],
            "source_case_cell": cell_name,
            "source_fit_file": fit_file,
            "source_reference_file": reference_file,
            "source_template_file": ROOT / template["source"],
            "dynamic_source_file": source_case,
        }
    records["metadata"] = input_metadata(
        case_id, "added3", source_case, None, records["P"], records["N"], data["witness"]["setting"]
    )
    return records


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def build_cases():
    split_manifest = read(UPSTREAM_SPLIT_MANIFEST)
    result = []
    split_cases = {}
    for witness in split_manifest["witnesses"]:
        wid = witness["id"]
        split_path = BASE / "cases" / f"{wid}.json"
        split_case = read(split_path)
        split_case["_path"] = split_path
        result.append(build_original_case(witness, split_case))
        split_cases[wid] = split_case
    require(len(result) == 50, "The frozen forecast-split manifest must contain all 50 original witnesses")
    extension_ids = ["s47_seed0_best_screen", "s47_seed0_optimizer", "s47_seed1_optimizer"]
    result.extend(build_extension_case(case_id) for case_id in extension_ids)
    require(len(result) == 53, "Expected 50 original and three added witnesses")
    require(len({x["metadata"]["id"] for x in result}) == 53, "Case identifiers must be unique")
    return result


def required_paths(cases):
    paths = {
        PLAN, Path(__file__).resolve(), AUDIT_SCRIPT, SOURCE_CIRCUIT, STAGE35_RUNNER, STAGE35_REPAIR,
        FREQUENCY_SOURCE, UPSTREAM_SPLIT_MANIFEST, UPSTREAM_TRANSFER_MANIFEST,
        UPSTREAM_EXTENSION_MANIFEST, BASE / "summary.json", BASE / "FINDINGS.md",
        BASE / "POST50_EVIDENCE_AUDIT.md", EXT / "FINDINGS.md", EXT / "PLAN.md",
        EXT / "continuation_v2_manifest.json", TRANSFER49 / "manifest.json",
        TRANSFER49 / "FINDINGS.md", SWEEP / "manifest.json", SWEEP / "controls.json",
    }
    for loss in ("linear", "soft_l1"):
        for cathode in ("P", "N"):
            paths.add(IM8 / "fits" / f"{cathode}_2C_Ch__source__{loss}__0.json")
    for case in cases:
        meta = case["metadata"]
        paths.add(ROOT / meta["source_case"])
        if meta["forecast_split_case"]:
            paths.add(ROOT / meta["forecast_split_case"])
        for label in ("P", "N"):
            cell = case[label]
            paths.add(cell["dynamic_source_file"])
            paths.add(cell["source_fit_file"])
            paths.add(cell["source_reference_file"])
            paths.add(cell["source_template_file"])
    return {p.resolve() for p in paths}


def verify_upstream_agreement(paths):
    upstreams = [
        read(UPSTREAM_SPLIT_MANIFEST),
        read(UPSTREAM_TRANSFER_MANIFEST),
        read(UPSTREAM_EXTENSION_MANIFEST),
        read(TRANSFER49 / "manifest.json"),
        read(SWEEP / "manifest.json"),
    ]
    for path in paths:
        key = rel(path)
        actual = sha(path)
        for manifest in upstreams:
            recorded = manifest.get("sha256", {}).get(key)
            if recorded is not None:
                require(recorded == actual, f"Upstream frozen hash mismatch: {key}")


def freeze():
    checkpoint_guard()
    split_summary = read(BASE / "summary.json")
    extension_summary = read(EXT / "summary.json")
    require(split_summary["all_numerical_pass"], "Original all50 forecast qualification failed")
    require(extension_summary["all_numerical_pass"], "Post50 three-vector qualification failed")
    cases = build_cases()
    paths = required_paths(cases)
    verify_upstream_agreement(paths)
    hashes = {rel(p): sha(p) for p in sorted(paths)}
    frequencies = 10 ** np.asarray(read(FREQUENCY_SOURCE)["frequency_log10_hz"], dtype=float)
    require(frequencies.size == 68 and np.all(np.diff(frequencies) < 0), "Expected frozen high-to-low 68-point axis")
    metadata = [c["metadata"] for c in cases]
    value = {
        "schema": "spectral-resolution-v1",
        "protocol": PLAN_PROTOCOL,
        "cases": metadata,
        "case_count": 53,
        "cohort_counts": {"original50": 50, "added3": 3},
        "frequency_hz_high_to_low": frequencies.tolist(),
        "profile_parameterization": {
            "coordinates": ["u=log10(fc_slow)+4", "t=normalized permitted frequency separation"],
            "log10_fc_slow": "-4+u",
            "log10_fc_fast": "-4+u+0.3+(13.7-u)*t",
            "bounds": {"u": [0.0, SLOW_LOG_MAX - FREQ_LOG_MIN], "t": [0.0, 1.0]},
            "consequence": "all original frequency boxes intersected with log10(fc_fast/fc_slow)>=0.3; fixed Rh belongs to fast branch",
        },
        "allocation_targets": {
            "balanced": {"P": "(rP+rN)/2", "N": "(rP+rN)/2"},
            "P_only": {"P": "rN", "N": "rN"},
            "N_only": {"P": "rP", "N": "rP"},
        },
        "sha256": hashes,
        "upstream_manifests": {
            rel(UPSTREAM_SPLIT_MANIFEST): sha(UPSTREAM_SPLIT_MANIFEST),
            rel(UPSTREAM_TRANSFER_MANIFEST): sha(UPSTREAM_TRANSFER_MANIFEST),
            rel(UPSTREAM_EXTENSION_MANIFEST): sha(UPSTREAM_EXTENSION_MANIFEST),
            rel(TRANSFER49 / "manifest.json"): sha(TRANSFER49 / "manifest.json"),
            rel(SWEEP / "manifest.json"): sha(SWEEP / "manifest.json"),
        },
        "stationary_control_templates": {
            rel(IM8 / "fits" / f"{c}_2C_Ch__source__{loss}__0.json"): sha(
                IM8 / "fits" / f"{c}_2C_Ch__source__{loss}__0.json"
            )
            for loss in ("linear", "soft_l1") for c in ("P", "N")
        },
        "implementation_sha256": sha(Path(__file__)),
        "plan_sha256": sha(PLAN),
    }
    if MANIFEST.exists():
        require(read(MANIFEST) == value, "Frozen source/input hashes differ; use a versioned continuation")
    else:
        save(MANIFEST, value)
    return value


def verify_frozen(manifest):
    require(sha(Path(__file__)) == manifest["implementation_sha256"], "Runner changed after freeze")
    require(sha(PLAN) == manifest["plan_sha256"], "Frozen scientific plan changed after freeze")
    for path, digest in manifest["sha256"].items():
        require(sha(ROOT / path) == digest, f"Frozen input changed: {path}")


def to_profile_y(x):
    logfast = float(x[1])
    logslow = float(x[4])
    u = logslow - FREQ_LOG_MIN
    width = SLOW_LOG_MAX - logslow
    separation = logfast - logslow
    if not (0 <= u <= SLOW_LOG_MAX - FREQ_LOG_MIN and width >= -1e-12 and separation >= FREQ_SEP_MIN - 1e-10):
        raise ValueError("Primary-fit start lies outside the ordered profile domain")
    t = 0.0 if width <= 1e-12 else (separation - FREQ_SEP_MIN) / width
    if not (-1e-10 <= t <= 1 + 1e-10):
        raise ValueError("Primary-fit start violates original frequency bounds")
    y = np.array([u, min(1.0, max(0.0, t)), x[2], x[3], x[5], x[6], x[7]], dtype=float)
    return y


def from_profile_y(y, log_r_fast):
    u, t, n_fast, log_r_slow, n_slow, log_d, tail_n = map(float, y)
    log_fc_slow = FREQ_LOG_MIN + u
    log_fc_fast = log_fc_slow + FREQ_SEP_MIN + (SLOW_LOG_MAX - log_fc_slow) * t
    return np.array([log_r_fast, log_fc_fast, n_fast, log_r_slow, log_fc_slow, n_slow, log_d, tail_n])


def profile_jacobian(y, log_r_fast, f, zobs, weights):
    x = from_profile_y(y, log_r_fast)
    jx = circuit.jacobian(x, f, zobs, weights, 0.0, False)
    u, t = float(y[0]), float(y[1])
    transform = np.zeros((8, 7), dtype=float)
    transform[1, 0] = 1.0 - t
    transform[1, 1] = SLOW_LOG_MAX - FREQ_LOG_MIN - u
    transform[2, 2] = 1.0
    transform[3, 3] = 1.0
    transform[4, 0] = 1.0
    transform[5, 4] = 1.0
    transform[6, 5] = 1.0
    transform[7, 6] = 1.0
    return jx @ transform


def profile_residual(y, f, zobs, weights, rs, free, log_r_fast):
    x = from_profile_y(y, log_r_fast)
    return circuit.residual(x, f, zobs, weights, rs, free)


def profile_starts(record, target_r, f):
    base_fit = record["fit"]
    target_log_r = math.log10(target_r)
    zobs = record["z_dynamic"]
    starts = []
    source_fast = max(0.01, record["r_bulk"])
    source_slow = max(0.01, record["r_slow"])
    low_im = abs(float(zobs[-1].imag))
    tail_seed = max(0.02, low_im * math.sqrt(float(f[-1])) / math.sin(math.pi / 4))
    for log_fc_fast, fc_mid, n in circuit.STARTS:
        x0 = np.array([
            target_log_r, log_fc_fast, n, math.log10(source_slow), math.log10(fc_mid), n,
            math.log10(tail_seed), 0.5,
        ], dtype=float)
        starts.append(("source_start_" + str(len(starts)), to_profile_y(x0)))
    selected_x = extract_x(base_fit)
    selected_x[0] = target_log_r
    selected_y = to_profile_y(selected_x)
    if not any(np.allclose(selected_y, y, rtol=0.0, atol=1e-12) for _, y in starts):
        starts.append(("primary_selected_fit_projected_to_fixed_Rh", selected_y))
    return starts


def profile_one(case_id, cell_label, allocation, record, target_ratio, f):
    target_r = float(target_ratio * record["r_pre"])
    task = {
        "case_id": case_id,
        "cell": cell_label,
        "allocation": allocation,
        "target_normalized_Rh": float(target_ratio),
        "target_absolute_Rh_ohm_cm2": target_r,
        "loss": record["loss"],
        "source_dynamic_spectrum_sha256": sha_json({
            "re": record["z_dynamic"].real.tolist(),
            "minus_im": (-record["z_dynamic"].imag).tolist(),
        }),
    }
    if not (circuit.LOW[0] <= math.log10(target_r) <= circuit.HIGH[0]):
        return {"task": task, "available": False, "reason": "prescribed Rh outside original circuit bounds", "starts": []}
    zobs = record["z_dynamic"]
    weights = np.abs(zobs)
    try:
        starts = profile_starts(record, target_r, f)
    except Exception as exc:
        return {"task": task, "available": False, "reason": "could not construct valid ordered profile starts: " + repr(exc), "starts": []}
    start_rows = []
    for start_name, y0 in starts:
        checkpoint_guard()
        try:
            ans = circuit.least_squares(
                profile_residual, y0,
                jac=lambda y, *unused_args, ff=f, zz=zobs, ww=weights, lr=math.log10(target_r):
                    profile_jacobian(y, lr, ff, zz, ww),
                args=(f, zobs, weights, 0.0, False, math.log10(target_r)),
                bounds=(PROFILE_LO, PROFILE_HI), loss=record["loss"],
                f_scale=circuit.FSCALE, max_nfev=2500,
                ftol=1e-13, xtol=1e-13, gtol=1e-13,
            )
            x = from_profile_y(ans.x, math.log10(target_r))
            pred = circuit.model(x, f, 0.0, False)
            params = circuit.parameters(x, 0.0, False)
            actual_sep = float(x[1] - x[4])
            raw_bound = np.flatnonzero(
                np.minimum(x - circuit.LOW, circuit.HIGH - x) /
                (circuit.HIGH - circuit.LOW) < 1e-5
            ).tolist()
            start_rows.append({
                "name": start_name,
                "y_start": y0.tolist(),
                "y": ans.x.tolist(),
                "x_source_coordinates": x.tolist(),
                "objective": float(ans.cost),
                "nfev": int(ans.nfev),
                "status": int(ans.status),
                "success": bool(ans.success),
                "termination": str(ans.message),
                "optimality": float(ans.optimality),
                "profile_bound_indices": np.flatnonzero(
                    np.minimum(ans.x - PROFILE_LO, PROFILE_HI - ans.x) /
                    (PROFILE_HI - PROFILE_LO) < 1e-5
                ).tolist(),
                "source_parameter_bound_indices": raw_bound,
                "parameters": params,
                "raw_fast_minus_slow_log10_fc": actual_sep,
                "fixed_fast_Rh_absolute_error": float(params["fast"]["r_ohm_cm2"] - target_r),
                "complex_relative_rms_to_dynamic": float(np.sqrt(np.mean(np.abs((pred - zobs) / weights) ** 2))),
                "complex_relative_max_to_dynamic": float(np.max(np.abs(pred - zobs) / weights)),
                "predicted_re_ohm_cm2": pred.real.tolist(),
                "predicted_minus_im_ohm_cm2": (-pred.imag).tolist(),
            })
        except Exception as exc:
            start_rows.append({"name": start_name, "y_start": y0.tolist(), "success": False,
                               "termination": "exception: " + repr(exc), "objective": None})
    finite = [r for r in start_rows if r.get("objective") is not None and np.isfinite(r["objective"])]
    if not finite:
        return {"task": task, "available": False, "reason": "all profile starts failed", "starts": start_rows}
    best = min(finite, key=lambda r: r["objective"])
    sep = best.get("raw_fast_minus_slow_log10_fc", -math.inf)
    branch_ok = bool(sep >= FREQ_SEP_MIN - 1e-10 and best["parameters"]["fast"]["r_ohm_cm2"] > 0)
    converged = bool(best.get("success", False))
    selected_valid = converged and branch_ok and abs(best["fixed_fast_Rh_absolute_error"]) <= 1e-12 * max(1., target_r)
    return {
        "task": task,
        "available": True,
        "selected_valid": selected_valid,
        "selected_start": best["name"],
        "selected_objective": best["objective"],
        "selected_success": converged,
        "fixed_Rh_in_faster_branch": branch_ok,
        "branch_separation_decades": sep,
        "branch_separation_pass": bool(sep >= FREQ_SEP_MIN - 1e-10),
        "selected_parameters": best["parameters"],
        "selected_x_source_coordinates": best["x_source_coordinates"],
        "selected_spectrum_re_ohm_cm2": best["predicted_re_ohm_cm2"],
        "selected_spectrum_minus_im_ohm_cm2": best["predicted_minus_im_ohm_cm2"],
        "selected_complex_relative_rms_to_dynamic": best["complex_relative_rms_to_dynamic"],
        "selected_complex_relative_max_to_dynamic": best["complex_relative_max_to_dynamic"],
        "starts": start_rows,
    }


def recover_one(case_id, cell_label, variant, record, zalt, target_ratio, f):
    spectrum_payload = {
        "re_ohm_cm2": zalt.real.tolist(),
        "minus_im_ohm_cm2": (-zalt.imag).tolist(),
        "source_bulk_li_r": float(record["r_bulk"]),
        "source_cathode_r": float(record["r_slow"]),
    }
    spectrum_digest = sha_json(spectrum_payload)
    spectrum_id = f"sr_{case_id}_{cell_label}_{variant}_{spectrum_digest[:10]}"
    task = {
        "spectrum_id": spectrum_id,
        "topology": "source",
        "loss": record["loss"],
        "fraction": 0.0,
        "input_spectrum_sha256": spectrum_digest,
    }
    file_path = RECOVERY_OUT / (circuit.task_id(task) + ".json")
    if file_path.exists():
        result = read(file_path)
        require(result["task"] == task, f"Recovery file does not match exact input: {file_path}")
    else:
        result = circuit.fit_one(task, spectrum_payload, f)
        result = read(file_path)
    best = result["best"]
    recovered_ratio = fitted_fast(result) / record["r_pre"]
    sep = float(best["parameters"]["branch_log10_separation"])
    curve = curve_from_fit(result)
    complex_error = float(np.max(np.abs(curve - zalt) / np.abs(zalt)))
    recovered_ok = bool(
        best["success"] and complex_error <= RECOVERY_TOLERANCE and
        abs(recovered_ratio - target_ratio) <= RATIO_TOLERANCE and
        sep >= FREQ_SEP_MIN - 1e-10
    )
    return {
        "variant": variant,
        "cell": cell_label,
        "expected_normalized_Rh": float(target_ratio),
        "recovered_normalized_Rh": float(recovered_ratio),
        "normalized_Rh_absolute_error": float(abs(recovered_ratio - target_ratio)),
        "complex_relative_max_to_constructed_spectrum": complex_error,
        "branch_separation_decades": sep,
        "branch_separation_pass": bool(sep >= FREQ_SEP_MIN - 1e-10),
        "best_success": bool(best["success"]),
        "best_start": result["best_start"],
        "recovery_pass": recovered_ok,
        "fit_path": rel(file_path),
        "fit_sha256": sha(file_path),
        "task": task,
        "all_start_objectives": [float(row["objective"]) for row in result["starts"]],
    }


def pooled_metric(alternative, reference, denominator):
    relative = np.abs(alternative - reference) / np.abs(denominator)
    return {"relative_complex_rms": float(np.sqrt(np.mean(relative ** 2))),
            "relative_complex_max": float(np.max(relative))}


def run_case(case, manifest_sha):
    tic = time.perf_counter()
    checkpoint_guard()
    case_id = case["metadata"]["id"]
    f = np.asarray(read(MANIFEST)["frequency_hz_high_to_low"], dtype=float)
    p, n = case["P"], case["N"]
    rp, rn = float(p["ratio"]), float(n["ratio"])
    d = rp - rn
    target_ratios = {
        "balanced": {"P": rp - d / 2, "N": rn + d / 2},
        "P_only": {"P": rp - d, "N": rn},
        "N_only": {"P": rp, "N": rn + d},
    }
    for label, values in target_ratios.items():
        require(abs(values["P"] - values["N"]) <= 2e-15, f"Nonzero prescribed contrast for {case_id}/{label}")
    profiled = {}
    for allocation, cells in [
        ("balanced", ("P", "N")),
        ("P_only", ("P",)),
        ("N_only", ("N",)),
    ]:
        for cell_label in cells:
            record = p if cell_label == "P" else n
            profiled[f"{allocation}_{cell_label}"] = profile_one(
                case_id, cell_label, allocation, record,
                target_ratios[allocation][cell_label], f,
            )
    spectra = {
        "P_original": p["z_static"],
        "N_original": n["z_static"],
    }
    for key, profile in profiled.items():
        if profile.get("selected_valid"):
            label = key.rsplit("_", 1)[1]
            allocation = key.rsplit("_", 1)[0]
            spectra[f"{label}_{allocation}"] = (
                np.asarray(profile["selected_spectrum_re_ohm_cm2"], dtype=float) -
                1j * np.asarray(profile["selected_spectrum_minus_im_ohm_cm2"], dtype=float)
            )
    recovery_specs = []
    for cell_label, record in [("P", p), ("N", n)]:
        recovery_specs.append((cell_label, "original", spectra[f"{cell_label}_original"], record["ratio"]))
    if "P_balanced" in spectra:
        recovery_specs.append(("P", "balanced", spectra["P_balanced"], target_ratios["balanced"]["P"]))
    if "P_P_only" in spectra:
        recovery_specs.append(("P", "P_only", spectra["P_P_only"], target_ratios["P_only"]["P"]))
    if "N_balanced" in spectra:
        recovery_specs.append(("N", "balanced", spectra["N_balanced"], target_ratios["balanced"]["N"]))
    if "N_N_only" in spectra:
        recovery_specs.append(("N", "N_only", spectra["N_N_only"], target_ratios["N_only"]["N"]))
    recovered = {}
    for cell_label, variant, zalt, expected in recovery_specs:
        record = p if cell_label == "P" else n
        recovered[f"{cell_label}_{variant}"] = recover_one(
            case_id, cell_label, variant, record, zalt, expected, f,
        )
    alternative_rows = {}
    for allocation in ("balanced", "P_only", "N_only"):
        if allocation == "balanced":
            variants = {"P": "balanced", "N": "balanced"}
            keys = {"P": "P_balanced", "N": "N_balanced"}
        elif allocation == "P_only":
            variants = {"P": "P_only", "N": "original"}
            keys = {"P": "P_P_only", "N": "N_original"}
        else:
            variants = {"P": "original", "N": "N_only"}
            keys = {"P": "P_original", "N": "N_N_only"}
        if any(k not in spectra for k in keys.values()):
            alternative_rows[allocation] = {
                "available": False,
                "reason": "required constrained profile unavailable",
                "target_ratios": target_ratios[allocation],
                "exact_zero_contrast": bool(abs(target_ratios[allocation]["P"] - target_ratios[allocation]["N"]) <= 2e-15),
            }
            continue
        zp, zn = spectra[keys["P"]], spectra[keys["N"]]
        zalt = np.concatenate([zp, zn])
        zdyn = np.concatenate([p["z_dynamic"], n["z_dynamic"]])
        zstat = np.concatenate([p["z_static"], n["z_static"]])
        denominator = np.concatenate([np.abs(p["z_dynamic"]), np.abs(n["z_dynamic"])])
        recovered_p = recovered[f"P_{variants['P']}"]
        recovered_n = recovered[f"N_{variants['N']}"]
        profile_pass = all(
            profiled[k].get("selected_valid", False)
            for k in (["balanced_P", "balanced_N"] if allocation == "balanced" else
                      ["P_only_P"] if allocation == "P_only" else ["N_only_N"])
        )
        recovered_pass = recovered_p["recovery_pass"] and recovered_n["recovery_pass"]
        alternative_rows[allocation] = {
            "available": True,
            "target_ratios": target_ratios[allocation],
            "exact_zero_contrast": bool(abs(target_ratios[allocation]["P"] - target_ratios[allocation]["N"]) <= 2e-15),
            "recovered_ratios": {"P": recovered_p["recovered_normalized_Rh"], "N": recovered_n["recovered_normalized_Rh"]},
            "recovered_contrast": float(recovered_p["recovered_normalized_Rh"] - recovered_n["recovered_normalized_Rh"]),
            "profile_pass": bool(profile_pass),
            "recovery_pass": bool(recovered_pass),
            "operational_zero_contrast_example": bool(profile_pass and recovered_pass),
            "spectral_perturbation": {
                "alternative_vs_dynamic": pooled_metric(zalt, zdyn, denominator),
                "alternative_vs_selected_static": pooled_metric(zalt, zstat, denominator),
                "selected_static_vs_dynamic": pooled_metric(zstat, zdyn, denominator),
                "per_cell": {
                    "P": {
                        "alternative_vs_dynamic": pooled_metric(zp, p["z_dynamic"], np.abs(p["z_dynamic"])),
                        "alternative_vs_selected_static": pooled_metric(zp, p["z_static"], np.abs(p["z_dynamic"])),
                        "selected_static_vs_dynamic": pooled_metric(p["z_static"], p["z_dynamic"], np.abs(p["z_dynamic"])),
                    },
                    "N": {
                        "alternative_vs_dynamic": pooled_metric(zn, n["z_dynamic"], np.abs(n["z_dynamic"])),
                        "alternative_vs_selected_static": pooled_metric(zn, n["z_static"], np.abs(n["z_dynamic"])),
                        "selected_static_vs_dynamic": pooled_metric(n["z_static"], n["z_dynamic"], np.abs(n["z_dynamic"])),
                    },
                },
            },
            "alternative_spectra": {
                "P_re_ohm_cm2": zp.real.tolist(), "P_minus_im_ohm_cm2": (-zp.imag).tolist(),
                "N_re_ohm_cm2": zn.real.tolist(), "N_minus_im_ohm_cm2": (-zn.imag).tolist(),
            },
            "recovery_ids": {"P": recovered_p["fit_path"], "N": recovered_n["fit_path"]},
        }
    valid_alt = [
        (name, row["spectral_perturbation"]["alternative_vs_dynamic"])
        for name, row in alternative_rows.items()
        if row.get("operational_zero_contrast_example")
    ]
    smallest = min(valid_alt, key=lambda v: v[1]["relative_complex_rms"]) if valid_alt else None
    result = {
        "case": case["metadata"],
        "manifest_sha256": manifest_sha,
        "primary_diagonal": {
            "rP": rp, "rN": rn, "contrast": d,
            "Rpre_P_fitted_ohm_cm2": float(p["r_pre"]),
            "Rpre_N_fitted_ohm_cm2": float(n["r_pre"]),
            "selected_fast_branch": {"P": p["fit"]["best"]["parameters"]["fast"], "N": n["fit"]["best"]["parameters"]["fast"]},
        },
        "target_ratios": target_ratios,
        "profiles": profiled,
        "recoveries": recovered,
        "alternatives": alternative_rows,
        "smallest_recovered_alternative": None if smallest is None else {
            "allocation": smallest[0], "spectral_perturbation": smallest[1],
            "interpretation": "smallest achieved among these three allocations for this witness; finite search only",
        },
        "case_pass": bool(all(v.get("operational_zero_contrast_example", False) for v in alternative_rows.values())),
        "wall_seconds": time.perf_counter() - tic,
    }
    out_path = OUT / f"{case_id}.json"
    save(out_path, result)
    return {"case_id": case_id, "cohort": case["metadata"]["cohort"],
            "case_pass": result["case_pass"], "seconds": result["wall_seconds"],
            "alternative_passes": {k: v.get("operational_zero_contrast_example", False) for k, v in alternative_rows.items()},
            "smallest": result["smallest_recovered_alternative"]}


def finite_difference_profile_jacobian(f):
    y = np.array([3.0, 0.35, 0.73, 1.05, 0.81, 0.4, 0.47])
    target_r = 19.0
    x = from_profile_y(y, math.log10(target_r))
    z = circuit.model(x, f, 0.0, False)
    weights = np.abs(z)
    analytic = profile_jacobian(y, math.log10(target_r), f, z, weights)
    errors = []
    for k in range(len(y)):
        h = 1e-6
        yp, ym = y.copy(), y.copy()
        yp[k] += h
        ym[k] -= h
        fd = (circuit.residual(from_profile_y(yp, math.log10(target_r)), f, z, weights, 0.0, False) -
              circuit.residual(from_profile_y(ym, math.log10(target_r)), f, z, weights, 0.0, False)) / (2 * h)
        scale = max(1.0, float(np.max(np.abs(fd))))
        errors.append(float(np.max(np.abs(fd - analytic[:, k])) / scale))
    return {"y_in_domain": y.tolist(), "relative_max_column_errors": errors,
            "max_relative_error": max(errors), "passed": bool(max(errors) <= 1e-7)}


def verify_domain_equivalence():
    forward_errors = []
    inverse_errors = []
    domain_points = 0
    for u in np.linspace(0.0, SLOW_LOG_MAX - FREQ_LOG_MIN, 17):
        for t in np.linspace(0.0, 1.0, 19):
            y = np.array([u, t, 0.6, 1.0, 0.7, 0.2, 0.5])
            x = from_profile_y(y, 1.2)
            slow, fast = float(x[4]), float(x[1])
            passed = bool(FREQ_LOG_MIN - 1e-12 <= slow <= SLOW_LOG_MAX + 1e-12 and
                          FREQ_LOG_MIN - 1e-12 <= fast <= FREQ_LOG_MAX + 1e-12 and
                          fast - slow >= FREQ_SEP_MIN - 1e-12)
            domain_points += int(passed)
            recovered = to_profile_y(x)
            forward_errors.append(float(np.max(np.abs(from_profile_y(recovered, 1.2) - x))))
    source_grid = 0
    for slow in np.linspace(FREQ_LOG_MIN, SLOW_LOG_MAX, 19):
        fast_min = slow + FREQ_SEP_MIN
        for fast in np.linspace(fast_min, FREQ_LOG_MAX, 17):
            x = np.array([1.2, fast, 0.6, 1.0, slow, 0.7, 0.2, 0.5])
            y = to_profile_y(x)
            reconstructed = from_profile_y(y, 1.2)
            inverse_errors.append(float(np.max(np.abs(reconstructed - x))))
            source_grid += 1
    max_roundtrip = max(forward_errors + inverse_errors)
    return {
        "transformed_domain_grid_points": domain_points,
        "source_domain_grid_points": source_grid,
        "max_roundtrip_log_frequency_error": max_roundtrip,
        "passed": bool(domain_points == 17 * 19 and source_grid == 19 * 17 and max_roundtrip <= 1e-12),
        "domain_statement": "The transform covers exactly -4<=log10(fc_slow)<log10(fc_fast)<=10 with separation>=0.3 decades.",
    }


def controls(manifest):
    control_path = HERE / "controls.json"
    if control_path.exists():
        existing = read(control_path)
        require(existing["manifest_sha256"] == sha(MANIFEST) and existing["passed"],
                "Existing qualification output has incompatible manifest or failed")
        return existing
    f = np.asarray(manifest["frequency_hz_high_to_low"], dtype=float)
    # Representative ordinary and robust stationary source templates, both cathodes.
    rows = []
    for extraction in ("ordinary", "robust"):
        loss = "soft_l1" if extraction == "robust" else "linear"
        for cell_label, cathode in (("P", "P-LCO"), ("N", "N-LCO")):
            path = IM8 / "fits" / f"{cathode[0]}_2C_Ch__source__{loss}__0.json"
            fit = best_fit(read(path))
            target_r = fitted_fast(fit)
            z = circuit.model(extract_x(fit), f, 0.0, False)
            record = {
                "fit": fit, "z_dynamic": z,
                "r_pre": target_r, "r_bulk": float(fit["source_bulk_li_r"]),
                "r_slow": float(fit["source_cathode_r"]), "loss": loss,
            }
            prof = profile_one(f"control_{extraction}_{cell_label}", cell_label, "unchanged_Rh",
                               record, 1.0, f)
            valid = bool(prof.get("selected_valid", False))
            profile_error = (prof.get("selected_complex_relative_max_to_dynamic", math.inf)
                             if prof.get("available") else math.inf)
            recovered = None
            if valid:
                zalt = (np.asarray(prof["selected_spectrum_re_ohm_cm2"]) -
                        1j * np.asarray(prof["selected_spectrum_minus_im_ohm_cm2"]))
                recovered = recover_one(f"control_{extraction}_{cell_label}", cell_label, "stationary",
                                        record, zalt, 1.0, f)
            rows.append({
                "extraction": extraction, "cell": cell_label,
                "template_source": rel(path), "target_Rh_ohm_cm2": target_r,
                "profile_valid": valid,
                "profile_max_complex_reconstruction": float(profile_error),
                "profile_branch_separation_decades": prof.get("branch_separation_decades"),
                "profile_starts": prof.get("starts", []),
                "recovery": recovered,
                "passed": bool(valid and profile_error <= RECOVERY_TOLERANCE and
                                recovered is not None and recovered["recovery_pass"]),
            })
            checkpoint_guard()
    jac = finite_difference_profile_jacobian(f)
    domain = verify_domain_equivalence()
    out = {
        "manifest_sha256": sha(MANIFEST),
        "representative_stationary_profiles": rows,
        "ordered_parameterization_jacobian": jac,
        "ordered_parameterization_domain": domain,
        "passed": bool(all(r["passed"] for r in rows) and jac["passed"] and domain["passed"]),
    }
    save(control_path, out)
    require(out["passed"], "Fixed-Rh profile qualification failed")
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["freeze", "pilot", "all"], required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    if args.mode == "freeze":
        m = freeze()
        print(json.dumps({"mode": "freeze", "cases": m["case_count"], "manifest_sha256": sha(MANIFEST)}), flush=True)
        return
    manifest = read(MANIFEST)
    verify_frozen(manifest)
    cases = build_cases()
    require([c["metadata"] for c in cases] == manifest["cases"], "Reconstructed dataset map differs from frozen manifest")
    control = controls(manifest)
    require(control["passed"], "Stationary profile controls failed")
    if args.mode == "pilot":
        wanted = ["w026", "w046", "s47_seed0_optimizer"]
        tasks = [c for c in cases if c["metadata"]["id"] in wanted]
        require(len(tasks) == len(wanted), "Pilot must include ordinary, robust and added-vector controls")
    else:
        tasks = cases
    reused = {}
    pending = []
    manifest_sha = sha(MANIFEST)
    for case in tasks:
        case_id = case["metadata"]["id"]
        path = OUT / f"{case_id}.json"
        if path.exists():
            saved = read(path)
            require(saved["manifest_sha256"] == manifest_sha and saved["case"]["id"] == case_id,
                    f"Existing case does not match frozen manifest: {path}")
            reused[rel(path)] = sha(path)
        else:
            pending.append(case)
    workers = int(os.environ.get("AJ_COMPUTE_WORKERS", "1"))
    completed = []
    OUT.mkdir(parents=True, exist_ok=True)
    RECOVERY_OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, case, manifest_sha) for case in pending]
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            print(json.dumps(row), flush=True)
            checkpoint_guard()
    out = {
        "mode": args.mode,
        "requested": len(tasks),
        "reused_sha256": reused,
        "new_results": completed,
        "workers": workers,
        "wall_seconds": time.perf_counter() - start,
        "manifest_sha256": manifest_sha,
        "controls_sha256": sha(HERE / "controls.json"),
    }
    save(HERE / f"{args.mode}_execution.json", out)
    print(json.dumps({"mode": args.mode, "completed": len(completed),
                      "seconds": out["wall_seconds"], "cases": len(tasks)}), flush=True)


if __name__ == "__main__":
    main()
