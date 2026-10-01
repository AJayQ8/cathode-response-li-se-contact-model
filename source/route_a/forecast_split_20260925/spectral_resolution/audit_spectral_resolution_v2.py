"""Independent angular-admittance reconstruction of spectral alternatives."""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
MANIFEST = HERE / "manifest.json"
PLAN = HERE / "PLAN.md"
RUNNER = HERE / "run_spectral_resolution.py"
CONTINUATION = HERE / "audit_v2_manifest.json"
CASE_DIR = HERE / "cases"
RECOVERY_DIR = HERE / "recovery_fits"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def close(a, b, atol=2e-12, rtol=2e-10):
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))


def required(condition, message, failures):
    if not condition:
        failures.append(message)


def unwrap(doc):
    if "fit" in doc and "inputs" in doc:
        return doc["fit"]
    return doc


def x_from_fit(fit):
    p = fit["best"]["parameters"]
    fast, slow = p["fast"], p["slow"]
    return np.array([
        math.log10(fast["r_ohm_cm2"]), math.log10(fast["fc_hz"]), fast["n"],
        math.log10(slow["r_ohm_cm2"]), math.log10(slow["fc_hz"]), slow["n"],
        math.log10(p["tail_d_ohm_cm2_at_1_hz"]), p["tail_n"],
    ], dtype=float)


def fast_r(fit):
    return float(fit["best"]["parameters"]["fast"]["r_ohm_cm2"])


def z_from_x(x, f):
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    result = np.zeros(f.shape, dtype=complex)
    for offset in (0, 3):
        resistance = 10.0 ** x[offset]
        fc = 10.0 ** x[offset + 1]
        exponent = x[offset + 2]
        q = 1.0 / (resistance * (2.0 * np.pi * fc) ** exponent)
        result += 1.0 / (1.0 / resistance + q * (2j * np.pi * f) ** exponent)
    tail_d = 10.0 ** x[6]
    tail_n = x[7]
    tail_q = 1.0 / (tail_d * (2.0 * np.pi) ** tail_n)
    result += 1.0 / (tail_q * (2j * np.pi * f) ** tail_n)
    return result


def z_from_fit(fit, f):
    return z_from_x(x_from_fit(fit), f)


def objective(zpred, zobs, loss):
    weights = np.abs(zobs)
    error = (zpred - zobs) / weights
    residual = np.r_[error.real, error.imag]
    if loss == "linear":
        return float(0.5 * np.sum(residual ** 2))
    if loss == "soft_l1":
        scaled2 = (residual / 0.02) ** 2
        return float(0.5 * 0.02 ** 2 * np.sum(2.0 * (np.sqrt(1.0 + scaled2) - 1.0)))
    raise ValueError(f"Unsupported frozen loss {loss}")


def complex_from_record(record):
    return np.asarray(record["re_ohm_cm2"], dtype=float) - 1j * np.asarray(
        record["minus_im_ohm_cm2"], dtype=float
    )


def primary_inputs(meta, label):
    item = meta["cells"][label]
    source_case = ROOT / meta["source_case"]
    if meta["cohort"] == "original50":
        data = read(source_case)
        cathode = "P-LCO" if label == "P" else "N-LCO"
        record = next(row for row in data["histories"] if row["cathode"] == cathode)
        zdyn = complex_from_record(record)
        rbulk = float(record["template"]["Rref"])
        rslow = float(record["template"]["slow_r"])
    else:
        data = read(source_case)
        record = data["cells"][item["source_case_cell"]]
        zdyn = complex_from_record(record["spectrum"])
        rbulk = float(record["template"]["Rref"])
        rslow = float(record["template"]["slow_r"])
    fit = unwrap(read(ROOT / item["selected_fit"]))
    reference = unwrap(read(ROOT / item["stationary_reference_fit"]))
    zstatic = np.asarray(fit["predicted_re_ohm_cm2"], dtype=float) - 1j * np.asarray(
        fit["predicted_minus_im_ohm_cm2"], dtype=float
    )
    zstatic_independent = z_from_fit(fit, np.asarray(read(MANIFEST)["frequency_hz_high_to_low"], dtype=float))
    return {
        "item": item, "dynamic": zdyn, "static": zstatic, "fit": fit,
        "static_independent": zstatic_independent,
        "reference": reference, "r_bulk": rbulk, "r_slow": rslow,
        "r_pre": fast_r(reference), "ratio": fast_r(fit) / fast_r(reference),
    }


def metric(a, b, denominator):
    relative = np.abs(a - b) / np.abs(denominator)
    return {"relative_complex_rms": float(np.sqrt(np.mean(relative ** 2))),
            "relative_complex_max": float(np.max(relative))}


def compare_metric(actual, expected, tag, failures, maximums):
    for key in ("relative_complex_rms", "relative_complex_max"):
        delta = abs(float(actual[key]) - float(expected[key]))
        maximums["metric_" + key] = max(maximums.get("metric_" + key, 0.0), delta)
        required(delta <= 5e-13, f"{tag}/{key}: {actual[key]} vs {expected[key]}", failures)


def audit_profile(case_result, profile_key, profile, record, f, failures, maximums):
    if not profile.get("available"):
        return {"available": False, "valid": False, "best_success": False}
    finite = []
    zobs = record["dynamic"]
    target = float(profile["task"]["target_absolute_Rh_ohm_cm2"])
    target_log = math.log10(target)
    for row in profile["starts"]:
        if row.get("objective") is None or "x_source_coordinates" not in row:
            continue
        x = np.asarray(row["x_source_coordinates"], dtype=float)
        pred = z_from_x(x, f)
        obj = objective(pred, zobs, profile["task"]["loss"])
        delta = abs(obj - float(row["objective"]))
        maximums["profile_objective_difference"] = max(maximums.get("profile_objective_difference", 0.0), delta)
        required(delta <= max(2e-12, 5e-9 * abs(obj)),
                 f"{case_result['case']['id']}/{profile_key}/{row['name']} objective mismatch", failures)
        required(abs(x[0] - target_log) <= 1e-13,
                 f"{case_result['case']['id']}/{profile_key}/{row['name']} fixed Rh mismatch", failures)
        sep = float(x[1] - x[4])
        required(sep >= 0.3 - 1e-10,
                 f"{case_result['case']['id']}/{profile_key}/{row['name']} lost faster-branch ordering", failures)
        required(np.all(x >= np.array([-4., -4., .2, -4., -4., .2, -5., .2]) - 1e-9) and
                 np.all(x <= np.array([3., 10., 1., 3., 10., 1., 3., 1.]) + 1e-9),
                 f"{case_result['case']['id']}/{profile_key}/{row['name']} outside original source bounds", failures)
        finite.append((row["name"], float(row["objective"]), row))
    if not finite:
        required(False, f"{case_result['case']['id']}/{profile_key} has no finite profile start", failures)
        return {"available": True, "valid": False, "best_success": False}
    selected = min(finite, key=lambda item: item[1])
    required(selected[0] == profile["selected_start"],
             f"{case_result['case']['id']}/{profile_key} selected start is not minimum objective", failures)
    required(abs(selected[1] - float(profile["selected_objective"])) <= 1e-14,
             f"{case_result['case']['id']}/{profile_key} selected objective mismatch", failures)
    selected_x = np.asarray(profile["selected_x_source_coordinates"], dtype=float)
    selected_z = z_from_x(selected_x, f)
    stored_z = np.asarray(profile["selected_spectrum_re_ohm_cm2"]) - 1j * np.asarray(
        profile["selected_spectrum_minus_im_ohm_cm2"]
    )
    spec_error = float(np.max(np.abs(selected_z - stored_z) / np.abs(selected_z)))
    maximums["profile_spectrum_reconstruction"] = max(maximums.get("profile_spectrum_reconstruction", 0.0), spec_error)
    required(spec_error <= 5e-13,
             f"{case_result['case']['id']}/{profile_key} profile curve reconstruction mismatch", failures)
    branch = bool(selected_x[1] > selected_x[4] and selected_x[1] - selected_x[4] >= 0.3 - 1e-10)
    selected_success = bool(selected[2].get("success", False))
    valid = branch and selected_success and profile.get("selected_valid", False)
    required(bool(profile["fixed_Rh_in_faster_branch"]) == branch,
             f"{case_result['case']['id']}/{profile_key} branch flag mismatch", failures)
    return {"available": True, "valid": valid, "best_success": selected_success,
            "branch_separation_decades": float(selected_x[1] - selected_x[4]),
            "objective": float(selected[1])}


def spectrum_for_variant(allocation, cell, case_result, primary):
    if allocation == "original":
        return primary["static"]
    row = case_result["alternatives"][allocation]
    if not row.get("available"):
        return None
    return complex_from_record({
        "re_ohm_cm2": row["alternative_spectra"][f"{cell}_re_ohm_cm2"],
        "minus_im_ohm_cm2": row["alternative_spectra"][f"{cell}_minus_im_ohm_cm2"],
    })


def recovery_allocation_variant(cell, variant):
    if variant == "original":
        return None
    if cell == "P" and variant == "balanced":
        return "balanced"
    if cell == "P" and variant == "P_only":
        return "P_only"
    if cell == "N" and variant == "balanced":
        return "balanced"
    if cell == "N" and variant == "N_only":
        return "N_only"
    raise ValueError((cell, variant))


def audit_recovery(case_result, key, recovery, cell, variant, primary, f, failures, maximums):
    allocation = recovery_allocation_variant(cell, variant)
    if allocation is None:
        zalt = primary["static"]
        target_ratio = primary["ratio"]
    else:
        row = case_result["alternatives"].get(allocation, {})
        if not row.get("available"):
            required(False, f"{case_result['case']['id']}/{key} recovery points to unavailable alternative", failures)
            return {"valid": False}
        zalt = complex_from_record({
            "re_ohm_cm2": row["alternative_spectra"][f"{cell}_re_ohm_cm2"],
            "minus_im_ohm_cm2": row["alternative_spectra"][f"{cell}_minus_im_ohm_cm2"],
        })
        target_ratio = float(row["target_ratios"][cell])
    payload = {
        "re_ohm_cm2": zalt.real.tolist(),
        "minus_im_ohm_cm2": (-zalt.imag).tolist(),
        "source_bulk_li_r": primary["r_bulk"],
        "source_cathode_r": primary["r_slow"],
    }
    digest = sha_json(payload)
    fit_path = ROOT / recovery["fit_path"]
    required(sha(fit_path) == recovery["fit_sha256"],
             f"{case_result['case']['id']}/{key} recovery fit hash mismatch", failures)
    fit = read(fit_path)
    required(fit["task"] == recovery["task"],
             f"{case_result['case']['id']}/{key} recovery task mismatch", failures)
    required(fit["task"].get("input_spectrum_sha256") == digest,
             f"{case_result['case']['id']}/{key} recovery input fingerprint mismatch", failures)
    required(fit["task"]["loss"] == primary["fit"]["task"]["loss"],
             f"{case_result['case']['id']}/{key} recovery loss mismatch", failures)
    starts = []
    for start in fit["starts"]:
        x = np.asarray(start["x"], dtype=float)
        pred = z_from_x(x, f)
        cost = objective(pred, zalt, fit["task"]["loss"])
        delta = abs(cost - float(start["objective"]))
        maximums["recovery_objective_difference"] = max(maximums.get("recovery_objective_difference", 0.0), delta)
        required(delta <= max(2e-12, 5e-9 * abs(cost)),
                 f"{case_result['case']['id']}/{key}/start{start['start_index']} objective mismatch", failures)
        starts.append((int(start["start_index"]), cost, start))
    # Use the source fitter's recorded objectives to verify its selection.
    # Independent recomputation validates every cost above, but roundoff can
    # reorder near-ties in the two equivalent circuit implementations.
    best = min(starts, key=lambda item: float(item[2]["objective"]))
    required(best[0] == int(fit["best_start"]), f"{case_result['case']['id']}/{key} selected recovery start mismatch", failures)
    required(int(fit["best_start"]) == int(recovery["best_start"]),
             f"{case_result['case']['id']}/{key} recorded recovery start mismatch", failures)
    best_x = np.asarray(fit["best"]["x"], dtype=float)
    best_z = z_from_x(best_x, f)
    reconstructed = float(np.max(np.abs(best_z - zalt) / np.abs(zalt)))
    maximums["recovery_spectrum_reconstruction"] = max(maximums.get("recovery_spectrum_reconstruction", 0.0), reconstructed)
    # The fitter sorts its two symmetric branches by frequency in the reported
    # parameters; raw x[0] need not be the fast resistance.
    recovered_ratio = fast_r(fit) / primary["r_pre"]
    sep = float(fit["best"]["parameters"]["branch_log10_separation"])
    recovered_pass = bool(fit["best"]["success"] and reconstructed <= 1e-7 and
                          abs(recovered_ratio - target_ratio) <= 1e-6 and sep >= 0.3 - 1e-10)
    required(bool(fit["best"]["success"]) == bool(recovery["best_success"]),
             f"{case_result['case']['id']}/{key} recorded optimizer success mismatch", failures)
    required(abs(recovered_ratio - recovery["recovered_normalized_Rh"]) <= 1e-13,
             f"{case_result['case']['id']}/{key} saved recovery ratio mismatch", failures)
    required(reconstructed <= 1e-7, f"{case_result['case']['id']}/{key} recovery spectrum exceeds 1e-7", failures)
    required(sep >= 0.3 - 1e-10, f"{case_result['case']['id']}/{key} recovery lost fast branch", failures)
    required(recovered_pass == recovery["recovery_pass"],
             f"{case_result['case']['id']}/{key} recovery status mismatch", failures)
    return {"valid": recovered_pass, "recovered_ratio": float(recovered_ratio),
            "target_ratio": float(target_ratio), "complex_relative_max": reconstructed,
            "branch_separation_decades": sep}


def case_metrics(case_result, primary, failures, maximums):
    for allocation, row in case_result["alternatives"].items():
        if not row.get("available"):
            continue
        zp = complex_from_record({"re_ohm_cm2": row["alternative_spectra"]["P_re_ohm_cm2"],
                                  "minus_im_ohm_cm2": row["alternative_spectra"]["P_minus_im_ohm_cm2"]})
        zn = complex_from_record({"re_ohm_cm2": row["alternative_spectra"]["N_re_ohm_cm2"],
                                  "minus_im_ohm_cm2": row["alternative_spectra"]["N_minus_im_ohm_cm2"]})
        alt = np.r_[zp, zn]
        dyn = np.r_[primary["P"]["dynamic"], primary["N"]["dynamic"]]
        static = np.r_[primary["P"]["static"], primary["N"]["static"]]
        den = np.abs(dyn)
        actual = {
            "alternative_vs_dynamic": metric(alt, dyn, den),
            "alternative_vs_selected_static": metric(alt, static, den),
            "selected_static_vs_dynamic": metric(static, dyn, den),
        }
        for key, value in actual.items():
            compare_metric(value, row["spectral_perturbation"][key],
                           f"{case_result['case']['id']}/{allocation}/{key}", failures, maximums)
        for cell, zalt in (("P", zp), ("N", zn)):
            refs = primary[cell]
            cell_actual = {
                "alternative_vs_dynamic": metric(zalt, refs["dynamic"], np.abs(refs["dynamic"])),
                "alternative_vs_selected_static": metric(zalt, refs["static"], np.abs(refs["dynamic"])),
                "selected_static_vs_dynamic": metric(refs["static"], refs["dynamic"], np.abs(refs["dynamic"])),
            }
            for key, value in cell_actual.items():
                compare_metric(value, row["spectral_perturbation"]["per_cell"][cell][key],
                               f"{case_result['case']['id']}/{allocation}/{cell}/{key}", failures, maximums)


def quantiles(values):
    if not values:
        return None
    return {"min": float(np.min(values)), "median": float(np.median(values)), "max": float(np.max(values))}


def make_summary(case_results, maximums, failures, controls):
    summary = {
        "counts": {"original50": 0, "added3": 0, "all53": 0},
        "allocations": {},
        "finite_search_minimum_by_case": {},
        "controls": {
            "passed": bool(controls["passed"]),
            "stationary_profiles": len(controls["representative_stationary_profiles"]),
            "jacobian_max_error": controls["ordered_parameterization_jacobian"]["max_relative_error"],
            "domain_equivalence_passed": controls["ordered_parameterization_domain"]["passed"],
        },
        "audit_maximum_differences": maximums,
        "audit_failures": failures,
    }
    cohorts = ["original50", "added3", "all53"]
    for c in case_results:
        cohort = c["case"]["cohort"]
        summary["counts"][cohort] += 1
        summary["counts"]["all53"] += 1
    for allocation in ("balanced", "P_only", "N_only"):
        summary["allocations"][allocation] = {}
        for cohort in cohorts:
            subset = [c for c in case_results if cohort == "all53" or c["case"]["cohort"] == cohort]
            rows = [c["alternatives"].get(allocation, {}) for c in subset]
            available = [r for r in rows if r.get("available")]
            recovered = [r for r in available if r.get("operational_zero_contrast_example")]
            metric_names = ["alternative_vs_dynamic", "alternative_vs_selected_static", "selected_static_vs_dynamic"]
            distributions = {}
            for name in metric_names:
                distributions[name] = {
                    stat: {
                        "relative_complex_rms": quantiles([r["spectral_perturbation"][name]["relative_complex_rms"] for r in available]),
                        "relative_complex_max": quantiles([r["spectral_perturbation"][name]["relative_complex_max"] for r in available]),
                    }
                    for stat in ("sampled_min_median_max",)
                }
            summary["allocations"][allocation][cohort] = {
                "cases": len(subset),
                "alternatives_available": len(available),
                "profile_and_recovery_passes": len(recovered),
                "failures": [c["case"]["id"] for c in subset if not c["alternatives"].get(allocation, {}).get("operational_zero_contrast_example", False)],
                "perturbations": distributions,
                "recovered_contrast_max_absolute": max([abs(r["recovered_contrast"]) for r in recovered], default=None),
            }
    minima = {}
    for cohort in cohorts:
        subset = [c for c in case_results if cohort == "all53" or c["case"]["cohort"] == cohort]
        rows = [c["smallest_recovered_alternative"] for c in subset if c["smallest_recovered_alternative"]]
        minima[cohort] = {
            "cases_with_recovered_alternative": len(rows),
            "relative_complex_rms_min_median_max": quantiles([r["spectral_perturbation"]["relative_complex_rms"] for r in rows]),
            "relative_complex_max_min_median_max": quantiles([r["spectral_perturbation"]["relative_complex_max"] for r in rows]),
            "allocations_selected_as_finite_search_minimum": {
                a: sum(1 for r in rows if r["allocation"] == a) for a in ("balanced", "P_only", "N_only")
            },
        }
    summary["finite_search_minimum_by_case"] = minima
    return summary


def render_findings(summary, case_results):
    lines = [
        "# Deterministic spectral-resolution diagnostic — Route A",
        "",
        "25 September 2026. Frozen all53 post-discharge forecasts; no kinetic refit, noise model, measured repeatability, new waveform or physical validation.",
        "",
        "The profiles use the original fixed-Q source circuit, magnitude weighting, loss, six-start extraction and 1e-13 stage35 tolerances. Added profile starts are the selected primary fit with only its fast Rh replaced by the fixed target. The constrained frequency transform preserves the fixed Rh on the faster branch and enforces the 0.3-decade separation. Original fitted stationary pre-discharge references are treated as exact.",
        "",
        "The original v1 audit is preserved in `audit.json` and reported 174 discrepancies. Its primary-fit input fingerprints used algebraically equivalent independently reconstructed spectra rather than the exact serialized curves passed to the recovery fitter, angular-admittance roundoff reordered near-tied starts, and raw optimizer coordinates were mistaken for the reported frequency-sorted fast branch. This versioned audit continuation corrects those verification semantics; it does not change any forecast, spectrum, fitting target, threshold, or fit output.",
        "",
        f"Independent angular-admittance reconstruction passed: {len(summary['audit_failures'])} audit discrepancies; {summary['controls']['stationary_profiles']} stationary profile controls passed; reduced-Jacobian finite-difference maximum relative error {summary['controls']['jacobian_max_error']:.3e}; transformed-domain equivalence {summary['controls']['domain_equivalence_passed'] }.",
        "",
        "Each value below is a finite-search perturbation of a constructed exact circuit spectrum relative to the saved sequential dynamic spectrum. Relative complex RMS and largest pointwise relative complex change use the original dynamic |Z| denominator. The static-fit residual is reported separately; these deterministic perturbations are not noise levels or instrument tolerances.",
        "",
        "| Allocation | Cohort | alternatives / cases | recovered zero-contrast examples | alt vs dynamic RMS (min / median / max) | alt vs dynamic point max (min / median / max) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    def fmt(q):
        return "unavailable" if q is None else f"{q['min']:.6g} / {q['median']:.6g} / {q['max']:.6g}"
    for alloc in ("balanced", "P_only", "N_only"):
        for cohort in ("original50", "added3", "all53"):
            row = summary["allocations"][alloc][cohort]
            d = row["perturbations"]["alternative_vs_dynamic"]["sampled_min_median_max"]
            rms = d["relative_complex_rms"]
            mx = d["relative_complex_max"]
            lines.append(f"| {alloc} | {cohort} | {row['alternatives_available']} / {row['cases']} | {row['profile_and_recovery_passes']} | {fmt(rms)} | {fmt(mx)} |")
    lines.extend([
        "",
        "The all53 column is descriptive: the three added stage47 fits include near-duplicate search outputs and do not add three independent replications. The primary total perturbation is alternative-versus-dynamic; alternative-versus-selected-static and selected-static-versus-dynamic components are saved per cell and per allocation to expose inherited fit residual and cancellation.",
        "",
        "Only alternatives whose original unconstrained extraction reconstructs within 1e-7 maximum relative complex error, recovers each prescribed normalized fast resistance within 1e-6, and retains the faster branch with at least 0.3-decade separation are called operational zero-contrast examples. All starts, failed profiles, failed recoveries and unavailable alternatives remain in the case records.",
        "",
        "This provides finite-search upper-bound examples for the spectral change sufficient to erase the model forecast under the same circuit family. Without a specified, measured complex-spectrum error allowance, it does not show that an experiment can or cannot resolve the contrast. It does not yield an empirical precision, repeatability, variance estimate, replicate count, or cell count. The fast apex remains above the measured band; systematic, initial-state, memory and fixture-transfer uncertainties remain separate.",
        "",
        "Frozen forecast inputs are in `manifest.json`; exact postrun output and launcher receipt hashes are in `audit_v2_manifest.json`. Every profile start and recovery fit is retained in the case JSON files and `recovery_fits/`. This diagnostic does not close the full reviewer request set.",
        "",
    ])
    return "\n".join(lines)


def main():
    tic = time.perf_counter()
    failures = []
    maximums = {}
    manifest = read(MANIFEST)
    continuation = read(CONTINUATION)
    required(continuation["base_manifest_sha256"] == sha(MANIFEST),
             "Audit continuation points to another frozen run", failures)
    required(continuation["corrected_audit_script_sha256"] == sha(Path(__file__)),
             "Corrected audit script differs from continuation receipt", failures)
    required(continuation["original_audit_script_sha256"] == sha(HERE / "audit_spectral_resolution.py"),
             "Original audit script differs from preserved receipt", failures)
    required(continuation["original_failed_audit_sha256"] == sha(HERE / "audit.json"),
             "Preserved v1 audit output changed after continuation freeze", failures)
    for rel_path, digest in continuation["preserved_v1_outputs_sha256"].items():
        required(sha(HERE / rel_path) == digest,
                 f"Preserved v1 output changed after continuation freeze: {rel_path}", failures)
    for rel_path, digest in continuation["postrun_outputs_sha256"].items():
        required(sha(HERE / rel_path) == digest,
                 f"Postrun output changed after audit continuation freeze: {rel_path}", failures)
    for path, digest in continuation["launcher_receipts_sha256"].items():
        required(sha(Path(path)) == digest,
                 f"Shared-launcher receipt changed after continuation freeze: {path}", failures)
    required(sha(RUNNER) == manifest["implementation_sha256"], "Runner source hash differs from manifest", failures)
    required(sha(PLAN) == manifest["plan_sha256"], "Plan hash differs from manifest", failures)
    for path, digest in manifest["sha256"].items():
        if sha(ROOT / path) != digest:
            failures.append(f"Frozen input hash mismatch: {path}")
    controls = read(HERE / "controls.json")
    required(controls["passed"], "Stationary controls failed", failures)
    cases = []
    for item in manifest["cases"]:
        case_path = CASE_DIR / f"{item['id']}.json"
        if not case_path.exists():
            failures.append(f"Missing case output: {item['id']}")
            continue
        case = read(case_path)
        required(case["manifest_sha256"] == sha(MANIFEST), f"Case manifest mismatch: {item['id']}", failures)
        required(case["case"] == item, f"Case identity mismatch: {item['id']}", failures)
        cases.append(case)
    required(len(cases) == 53, f"Expected all53 saved cases, found {len(cases)}", failures)
    f = np.asarray(manifest["frequency_hz_high_to_low"], dtype=float)
    for case in cases:
        primary = {cell: primary_inputs(case["case"], cell) for cell in ("P", "N")}
        for cell in ("P", "N"):
            record = primary[cell]
            item = case["case"]["cells"][cell]
            required(abs(record["ratio"] - float(item["primary_ratio"])) <= 1e-12,
                     f"{case['case']['id']}/{cell} primary ratio provenance mismatch", failures)
            required(abs(record["ratio"] - float(case["primary_diagonal"]["r" + cell])) <= 1e-12,
                     f"{case['case']['id']}/{cell} case ratio mismatch", failures)
            saved_curve = np.asarray(record["fit"]["predicted_re_ohm_cm2"]) - 1j * np.asarray(
                record["fit"]["predicted_minus_im_ohm_cm2"]
            )
            e = float(np.max(np.abs(saved_curve - record["static_independent"]) /
                              np.abs(record["static_independent"])))
            maximums["primary_static_spectrum_reconstruction"] = max(
                maximums.get("primary_static_spectrum_reconstruction", 0.0), e)
            required(e <= 5e-13, f"{case['case']['id']}/{cell} primary curve reconstruction mismatch", failures)
        rp, rn = primary["P"]["ratio"], primary["N"]["ratio"]
        d = rp - rn
        expected_targets = {
            "balanced": {"P": rp - d / 2, "N": rn + d / 2},
            "P_only": {"P": rp - d, "N": rn},
            "N_only": {"P": rp, "N": rn + d},
        }
        required(close(np.array(case["target_ratios"]["balanced"]["P"]), np.array(expected_targets["balanced"]["P"]), 1e-14, 0),
                 f"{case['case']['id']} balanced target mismatch", failures)
        for allocation, target in expected_targets.items():
            required(abs(target["P"] - target["N"]) <= 2e-15,
                     f"{case['case']['id']}/{allocation} prescribed contrast not zero", failures)
            row = case["alternatives"][allocation]
            required(abs(row["target_ratios"]["P"] - target["P"]) <= 2e-15 and
                     abs(row["target_ratios"]["N"] - target["N"]) <= 2e-15,
                     f"{case['case']['id']}/{allocation} target record mismatch", failures)
        profile_audit = {}
        for key, profile in case["profiles"].items():
            cell = key.rsplit("_", 1)[1]
            profile_audit[key] = audit_profile(case, key, profile, primary[cell], f, failures, maximums)
        recovery_audit = {}
        for key, recovery in case["recoveries"].items():
            cell, variant = key.split("_", 1)
            recovery_audit[key] = audit_recovery(case, key, recovery, cell, variant,
                                                primary[cell], f, failures, maximums)
        case_metrics(case, primary, failures, maximums)
        # Independently verify that each zero-contrast example selects its
        # fixed-R profile only on changed cells and retains the original exact
        # selected circuit for the unchanged cell.
        for allocation, row in case["alternatives"].items():
            if not row.get("available"):
                continue
            if allocation == "balanced":
                changed = ["P", "N"]
            elif allocation == "P_only":
                changed = ["P"]
            else:
                changed = ["N"]
            for cell in changed:
                key = f"{allocation}_{cell}"
                required(case["profiles"][key].get("selected_valid", False),
                         f"{case['case']['id']}/{allocation}/{cell} selected profile invalid", failures)
    summary = make_summary(cases, maximums, failures, controls)
    summary["wall_seconds"] = time.perf_counter() - tic
    summary["manifest_sha256"] = sha(MANIFEST)
    summary["audit_continuation_sha256"] = sha(CONTINUATION)
    summary["audit_script_sha256"] = sha(Path(__file__))
    summary["audit_passed"] = not failures
    summary["case_output_sha256"] = {str(path.relative_to(HERE)): sha(path)
                                     for path in sorted(CASE_DIR.glob("*.json"))}
    summary["recovery_fit_sha256"] = {str(path.relative_to(HERE)): sha(path)
                                      for path in sorted(RECOVERY_DIR.glob("*.json"))}
    (HERE / "summary_v2.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    audit = {
        "audit_passed": not failures,
        "failures": failures,
        "maximum_independent_differences": maximums,
        "cases_audited": len(cases),
        "controls_passed": controls["passed"],
        "manifest_sha256": sha(MANIFEST),
        "audit_continuation_sha256": sha(CONTINUATION),
        "runner_sha256": sha(RUNNER),
        "audit_script_sha256": sha(Path(__file__)),
        "case_output_sha256": summary["case_output_sha256"],
        "recovery_fit_sha256": summary["recovery_fit_sha256"],
        "wall_seconds": summary["wall_seconds"],
    }
    (HERE / "audit_v2.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    (HERE / "FINDINGS_v2.md").write_text(render_findings(summary, cases))
    print(json.dumps({"audit_passed": audit["audit_passed"], "cases": len(cases),
                      "failures": len(failures), "seconds": summary["wall_seconds"]}), flush=True)
    if failures:
        raise RuntimeError(f"Independent audit found {len(failures)} discrepancy(s); see audit_v2.json")


if __name__ == "__main__":
    main()
