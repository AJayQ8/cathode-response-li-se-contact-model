"""One bounded, min-only challenge to the saved P/N acquisition forecast."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from shared_compute import PoolError, checkpoint_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PAPER = HERE.parent
STAGE28 = PAPER / "counterexample28"
STAGE35 = PAPER / "sweep35"
OUT_DIRS = ["progress", "optimizer_returns", "searches", "cases", "fits"]
for name in OUT_DIRS:
    (HERE / name).mkdir(parents=True, exist_ok=True)


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = load_module("challenge_stage28", STAGE28 / "run_contrast_profile.py")
base.HERE = HERE
sweep = load_module("challenge_stage35", STAGE35 / "run_sweep.py")
sweep.circuit.OUT = HERE / "fits"
model = base.model
assessment = base.assessment
projection = base.projection
read, save, sha = model.read, model.save, model.sha
WIDTH = model.HIGH - model.LOW
LIMIT = 0.03 - 2e-6
SCREEN = 0.03
ACQUISITION_BIAS_LIMIT = 0.001
TASK_SEEDS = [("w003", "ordinary"), ("w007", "robust"), ("w009", "published")]
MAX_EVALUATIONS = 120
MAX_OPTIMIZATION_SECONDS = 900.0
MAX_ITERATIONS = 30


def setting_task(w):
    return {k: w["setting"][k] for k in ("registration", "timing", "extraction")}


def task_id(task):
    return f"{task['extraction']}_{task['registration']}_{task['timing']}_min_{task['seed_id']}"


def safe_json(path, value):
    save(path, value)


def freeze():
    m28 = read(STAGE28 / "manifest.json")
    m35 = read(STAGE35 / "manifest.json")
    assert abs(float(model.M)-6.6) < 1e-12
    sources = {}
    for manifest in (m28, m35):
        for rel, digest in manifest["sha256"].items():
            assert sha(ROOT / rel) == digest, rel
            sources[rel] = digest
    extras = [
        HERE / "PROTOCOL.md", Path(__file__).resolve(),
        STAGE28 / "manifest.json", STAGE28 / "controls.json", STAGE28 / "summary.json",
        STAGE28 / "execution.json", STAGE28 / "PLAN.md", STAGE28 / "FINDINGS.md",
        STAGE35 / "manifest.json", STAGE35 / "controls.json", STAGE35 / "summary.json",
        STAGE35 / "run_sweep.py", STAGE35 / "verify_and_summarize.py",
        STAGE35 / "PLAN.md", STAGE35 / "FINDINGS.md",
    ]
    witnesses = []
    for wid, extraction in TASK_SEEDS:
        w = next(w for w in m35["witnesses"] if w["id"] == wid)
        assert w["setting"]["extraction"] == extraction
        assert w["setting"]["registration"] == "local"
        assert w["setting"]["timing"] == ("early" if wid in ("w003", "w007") else "late")
        case_path = STAGE35 / "cases" / f"{wid}_fixed_q_high_to_low.json"
        case = read(case_path)
        assert case["task"] == dict(witness_id=wid, law="fixed_q", order="high_to_low")
        assert case["numerical_pass"] and case["instantaneous_budget_pass"]
        assert case["positive_contrast"]
        extras.append(case_path)
        for h in case["histories"]:
            fit_path = STAGE35 / h["fit_path"]
            extras.append(fit_path)
        witnesses.append(dict(witness=w, original_case=str(case_path.relative_to(ROOT)),
                               original_case_sha256=sha(case_path), original_case_values={
                                   k: case[k] for k in ("fitted_contrast", "true_contrast_t0", "bias_from_t0",
                                                        "instantaneous_budget_pass", "numerical_pass")}))
    # Bind the six stationary 2 C extraction-specific references used by stage35.
    controls = read(STAGE35 / "controls.json")
    assert controls["passed"] and controls["manifest_sha256"] == sha(STAGE35 / "manifest.json")
    for row in controls["records"]:
        extras.append(STAGE35 / row["fit_path"])
    for path in extras:
        rel = str(path.relative_to(ROOT))
        digest = sha(path)
        if rel in sources:
            assert sources[rel] == digest, rel
        sources[rel] = digest
    frozen = dict(
        reference_commit="c5fe80a4fad4b70bf3d4e72f7d1419e296703795",
        protocol_sha256=sha(HERE / "PROTOCOL.md"),
        stage28_manifest_sha256=sha(STAGE28 / "manifest.json"),
        stage35_manifest_sha256=sha(STAGE35 / "manifest.json"),
        source_sha256=sources,
        witnesses=witnesses,
        model=dict(M=float(model.M), low=model.LOW.tolist(), high=model.HIGH.tolist(),
                   all_14_observations=True, screen=SCREEN, inward_search_limit=LIMIT),
        search=dict(objective="stage28 instantaneous discharge-end P-minus-N normalized resistance",
                    direction="minimum only", maxiter=MAX_ITERATIONS, ftol=1e-9,
                    complete_evaluation_cap=MAX_EVALUATIONS,
                    wall_seconds_per_task=MAX_OPTIMIZATION_SECONDS),
        acquisition=dict(law="fixed_q", order="high_to_low", source_frequency_count=68,
                         settle_seconds=0.5, cycles_per_frequency=3,
                         duration_seconds=106.9561255904,
                         fit_tolerances=dict(ftol=1e-13, xtol=1e-13, gtol=1e-13),
                         quadrature_nodes=[8, 16], quadrature_relative_limit=1e-8,
                         physical_spectrum_relative_limit=1e-7,
                         acquisition_bias_limit=ACQUISITION_BIAS_LIMIT),
        no_exponent_variants=True, no_extra_seeds=True, no_extracted_objective_optimization=True)
    path = HERE / "manifest.json"
    if path.exists():
        assert read(path) == frozen, "Frozen challenge manifest changed"
    else:
        safe_json(path, frozen)
    return frozen


def all14_assessment(task, x, label):
    rstar, protocols = model.data(task["extraction"], task["registration"])
    p = model.Problem(task)
    item = assessment.assess(p, np.asarray(x, dtype=float), label)
    points = item["points"]
    usable = [r for r in points if not model.base.unavailable(r)]
    max_error = max((abs(float(r["error"])) for r in usable), default=float("inf"))
    item["max_all14_abs_error"] = max_error
    item["all_known_admitted"] = bool(
        len(points) == 14 and len(usable) == 14 and item["numerical_pass"]
        and item["physical_observation_pass"] and max_error <= SCREEN
        and all(item["subsets"][s]["screen_pass"] for s in ("train", "check")))
    item["rstar"] = rstar
    return item


def controls(frozen):
    path = HERE / "controls.json"
    if path.exists():
        old = read(path)
        assert old["manifest_sha256"] == sha(HERE / "manifest.json") and old["passed"]
        return old
    tests = []
    for entry in frozen["witnesses"]:
        w = entry["witness"]
        task = dict(setting_task(w), direction="min", seed=0)
        x = np.asarray(w["x"], dtype=float)
        y = (x - model.LOW) / WIDTH
        value, derivative, good = base.contrast(x, task)
        independent, _, independent_good = base.contrast(x, task, independent=True)
        assert good and independent_good
        errors, stencils = [], []

        def direct(v):
            val, _, valid = base.contrast(model.LOW + WIDTH * v, task, independent=True)
            assert valid
            return val

        for col in range(5):
            h = 1e-5
            step = np.eye(5)[col] * h
            if y[col] - h >= 0 and y[col] + h <= 1:
                fd = (direct(y + step) - direct(y - step)) / (2 * h)
                stencil = "central"
            elif y[col] + 2 * h <= 1:
                fd = (-3 * independent + 4 * direct(y + step) - direct(y + 2 * step)) / (2 * h)
                stencil = "forward_second_order"
            else:
                fd = (3 * independent - 4 * direct(y - step) + direct(y - 2 * step)) / (2 * h)
                stencil = "backward_second_order"
            errors.append(float(abs(fd - derivative[col] * WIDTH[col]) / max(1., abs(fd))))
            stencils.append(stencil)
        prior = read(ROOT / entry["original_case"])
        candidate = all14_assessment(task, x, "saved_start_control")
        test = dict(seed_id=w["id"], setting=w["setting"], seed_x=x.tolist(),
                    objective=independent, fast_independent_value_difference=abs(value-independent),
                    saved_seed_instantaneous=prior["true_contrast_t0"],
                    saved_seed_fitted=prior["fitted_contrast"],
                    saved_seed_value_difference=abs(independent-prior["true_contrast_t0"]),
                    derivative_scaled_errors=errors, derivative_stencils=stencils,
                    all14_max_abs_error=candidate["max_all14_abs_error"],
                    all14_admitted=candidate["all_known_admitted"])
        test["passed"] = bool(max(errors) <= .001 and abs(value-independent) <= 1e-4
                              and abs(independent-prior["true_contrast_t0"]) <= 1e-7
                              and candidate["all_known_admitted"])
        tests.append(test)
        checkpoint_guard()
    report = dict(tests=tests, passed=all(t["passed"] for t in tests),
                  derivative_scaled_target=.001, direct_value_target=1e-4,
                  saved_seed_value_target=1e-7, all14_screen=SCREEN,
                  manifest_sha256=sha(HERE / "manifest.json"))
    safe_json(path, report)
    return report


class SearchLimit(Exception):
    pass


class CappedProblem(base.Problem):
    def __init__(self, task):
        super().__init__(task)
        self.limit_reason = None
        self.last_iterate = None
        self.iterations = 0

    def calculate(self, y):
        if self.last_y is not None and np.array_equal(y, self.last_y):
            return super().calculate(y)
        if self.evaluations >= MAX_EVALUATIONS:
            self.limit_reason = "complete_evaluation_cap"
            raise SearchLimit(self.limit_reason)
        if time.perf_counter() - self.started >= MAX_OPTIMIZATION_SECONDS:
            self.limit_reason = "optimization_wall_limit"
            raise SearchLimit(self.limit_reason)
        return super().calculate(y)


def sweep_fit(spectrum_id, z, f, template, loss):
    task = dict(spectrum_id=spectrum_id, topology="source", loss=loss, fraction=0.)
    path = HERE / "fits" / (sweep.circuit.task_id(task) + ".json")
    spectrum = dict(re_ohm_cm2=z.real.tolist(), minus_im_ohm_cm2=(-z.imag).tolist(),
                    source_bulk_li_r=template["Rref"], source_cathode_r=template["slow_r"])
    if path.exists():
        result = read(path)
        assert result["task"] == task
    else:
        sweep.circuit.fit_one(task, spectrum, f)
        result = read(path)
    return result, str(path.relative_to(HERE))


def physical_spectrum_check(w, h, template, points, z, f, order):
    # Independent physical-recession route used by the stage35 saved-output audit.
    L = 10**w["x"][0]
    gamma = h["gammas_per_hour"][1]
    a0 = h["rows"][0]["independent"]["contact"][0]
    g0 = L * (1-a0)
    m = float(model.M)

    def rhs(t, g):
        a = 1 - g[0] / L
        return [-L * gamma * (1-a) * a**(-m)]

    def jac(t, g):
        a = 1 - g[0] / L
        return np.array([[gamma * (-a**(-m) - m*(1-a)*a**(-m-1))]])

    duration = sweep.schedule(f, order)[1]
    sol = solve_ivp(rhs, (0., duration/3600), [g0], method="Radau", jac=jac,
                    dense_output=True, rtol=1e-12, atol=1e-13)
    assert sol.success
    nodes, weights = np.polynomial.legendre.leggauss(16)
    b, A = w["x"][3:]
    rb = b * w["rstar"]
    C = A * (1-b) * w["rstar"]
    max_spec = 0.
    max_endpoint = 0.
    for k, row in enumerate(points):
        u, v = row["measure_start_s"], row["measure_end_s"]
        seconds = (u+v)/2 + nodes*(v-u)/2
        contact = 1 - sol.sol(seconds/3600)[0] / L
        resistance = rb + C/contact
        actual = sweep.forward(f[k], resistance, template, "fixed_q", direct=True)
        actual = np.dot(weights, actual)/2
        max_spec = max(max_spec, float(abs(z[k]-actual)/abs(actual)))
        ends = 1 - sol.sol(np.array([u, v])/3600)[0] / L
        r_ends = rb + C/ends
        max_endpoint = max(max_endpoint, float(np.max(np.abs(r_ends-row["resistance_endpoints"]) / template["Rref"])))
    return dict(physical_spectrum_relative_error=max_spec,
                physical_endpoint_relative_error=max_endpoint,
                passed=bool(max_spec <= 1e-7 and max_endpoint <= 1e-8))


def project_readout(task, seed_w, x, label, assessment_record):
    w = dict(id=f"challenge_{task_id(task)}_{label}", x=np.asarray(x).tolist(),
             rstar=seed_w["rstar"], setting=deepcopy(seed_w["setting"]))
    source_witness = dict(w)
    f = sweep.frequency()
    schedule, duration = sweep.schedule(f, "high_to_low")
    ctrl = read(STAGE35 / "controls.json")
    extraction = task["extraction"]
    cells = []
    for cathode in ("P-LCO", "N-LCO"):
        hist = projection.project(source_witness, cathode)
        if not hist.get("available"):
            cells.append(dict(cathode=cathode, available=False, physical_projection=hist))
            continue
        xarr = np.asarray(x, dtype=float)
        A = float(xarr[4])
        H = 10**float(xarr[1])
        q_end = float(hist["q0"]) + projection.Q
        pressure = 2 + float(xarr[2]) * model.pressure_increment(cathode, q_end)
        gamma = H * A**(model.M+1) * (pressure/2)**model.M
        L = 10**float(xarr[0])
        independent_contact = 1.0 - float(hist["rows"][0]["recession_um"]) / L
        contact_route_difference = abs(independent_contact-float(hist["rows"][0]["a"]))
        h = dict(cathode=cathode, R_ch=hist["R_ch"], gammas_per_hour=[0., gamma],
                 rows=[dict(independent=dict(contact=[independent_contact],
                                            ratio=hist["rows"][0]["independent_ratio"]))])
        template = sweep.make_template(extraction, cathode, h["R_ch"])
        reference = next(r for r in ctrl["records"] if r["extraction"] == extraction and r["cathode"] == cathode)
        assert template == reference["template"]
        z, points = sweep.generate(w, h, template, "fixed_q", schedule, f, 8)
        z16, _ = sweep.generate(w, h, template, "fixed_q", schedule, f, 16, True)
        quadrature = float(np.max(np.abs(z-z16) / np.abs(z16)))
        spectrum_check = physical_spectrum_check(w, h, template, points, z, f, "high_to_low")
        fit, fit_path = sweep_fit(f"{task_id(task)}_{label}_{cathode[0]}", z, f, template, sweep.loss_for(extraction))
        resistance = float(fit["best"]["parameters"]["fast"]["r_ohm_cm2"])
        fitted_ratio = resistance / reference["fitted_reference"]
        instant_zero = float(hist["rows"][0]["independent_ratio"])
        a_end = sweep.rest.implicit_contact(h["rows"][0]["independent"]["contact"][0], gamma, duration)
        b, A = xarr[3:]
        true_end = float((b*w["rstar"] + A*(1-b)*w["rstar"]/a_end) / h["R_ch"])
        inside = bool(true_end-1e-6 <= fitted_ratio <= instant_zero+1e-6)
        cells.append(dict(cathode=cathode, available=True, physical_projection=hist,
                          template=template, reference_fit_path=reference["fit_path"],
                          reference_fit_sha256=sha(STAGE35 / reference["fit_path"]), fit_path=fit_path,
                          generated_re_ohm_cm2=z.real.tolist(), generated_minus_im_ohm_cm2=(-z.imag).tolist(),
                          generated_direct16_re_ohm_cm2=z16.real.tolist(),
                          generated_direct16_minus_im_ohm_cm2=(-z16.imag).tolist(), points=points,
                          quadrature_direct_error=quadrature, physical_spectrum_check=spectrum_check,
                          fitted_ratio=float(fitted_ratio), true_ratio_t0=instant_zero,
                          true_ratio_tend=true_end, fit_inside_instantaneous_extrema=inside,
                          physical_vs_model_contact_difference=contact_route_difference,
                          selected_fit_success=bool(fit["best"]["success"]),
                          all_start_failures=sum(not bool(r["success"]) for r in fit["starts"]),
                          selected_bounds=fit["best"]["bound_indices"],
                          branch_log10_separation=fit["best"]["parameters"]["branch_log10_separation"],
                          complex_relative_rms=fit["complex_relative_rms"],
                          max_fractional_R_change_per_cycle=max(max(p["fractional_R_change_per_cycle_at_endpoints"]) for p in points)))
    if not all(c["available"] for c in cells):
        return dict(label=label, x=np.asarray(x).tolist(), all14=assessment_record,
                    histories=cells, forecast_available=False, forecast_qualified=False,
                    manifest_sha256=sha(HERE / "manifest.json"))
    P, N = cells
    contrast = P["fitted_ratio"] - N["fitted_ratio"]
    true_contrast = P["true_ratio_t0"] - N["true_ratio_t0"]
    interval = [P["true_ratio_tend"] - N["true_ratio_t0"],
                P["true_ratio_t0"] - N["true_ratio_tend"]]
    bias = contrast - true_contrast
    numerical = all(c["selected_fit_success"] and c["quadrature_direct_error"] <= 1e-8
                    and c["physical_spectrum_check"]["passed"] for c in cells)
    out = dict(label=label, x=np.asarray(x).tolist(), all14=assessment_record,
               task=dict(witness_id=w["id"], law="fixed_q", order="high_to_low"),
               setting=w["setting"], duration_seconds=duration, histories=cells,
               fitted_contrast=float(contrast), true_contrast_t0=float(true_contrast),
               bias_from_t0=float(bias), positive_contrast=bool(contrast > 0),
               instantaneous_bias_budget_pass=bool(abs(bias) <= ACQUISITION_BIAS_LIMIT),
               true_monotone_window_interval=interval,
               pair_inside_true_window=bool(interval[0]-2e-6 <= contrast <= interval[1]+2e-6),
               all_cells_inside_instantaneous_extrema=all(c["fit_inside_instantaneous_extrema"] for c in cells),
               numerical_pass=bool(numerical), forecast_available=True,
               forecast_qualified=bool(assessment_record["all_known_admitted"] and numerical),
               manifest_sha256=sha(HERE / "manifest.json"))
    return out


def candidate_outputs(task, seed_w, optimizer, problem):
    x_start = np.asarray(seed_w["x"], dtype=float)
    selected = [("start", x_start)]
    if optimizer is not None:
        selected.append(("optimizer_return", model.LOW + WIDTH*np.asarray(optimizer.x)))
    elif problem.last_iterate is not None:
        selected.append(("last_iterate_on_limit", model.LOW + WIDTH*problem.last_iterate))
    if problem.best is not None:
        selected.append(("best_feasible_visited", np.asarray(problem.best["x"], dtype=float)))
    items = []
    seen = {}
    prior_path = STAGE35 / "cases" / f"{seed_w['id']}_fixed_q_high_to_low.json"
    prior = read(prior_path)
    for label, x in selected:
        checkpoint_guard()
        key = tuple(np.asarray(x, dtype=float).tolist())
        if key in seen:
            reused = deepcopy(seen[key]); reused["label"] = label; items.append(reused); continue
        if np.array_equal(np.asarray(x), x_start):
            rec = all14_assessment(task, x, label)
            result = dict(label=label, x=np.asarray(x).tolist(), all14=rec,
                          reused_saved_seed_readout=True, source=str(prior_path.relative_to(ROOT)),
                          source_sha256=sha(prior_path), forecast_available=True,
                          forecast_qualified=bool(rec["all_known_admitted"] and prior["numerical_pass"]),
                          fitted_contrast=prior["fitted_contrast"], true_contrast_t0=prior["true_contrast_t0"],
                          bias_from_t0=prior["bias_from_t0"], positive_contrast=prior["positive_contrast"],
                          numerical_pass=prior["numerical_pass"], instantaneous_bias_budget_pass=prior["instantaneous_budget_pass"],
                          saved_case=prior)
        else:
            rec = all14_assessment(task, x, label)
            if rec["all_known_admitted"]:
                try:
                    result = project_readout(task, seed_w, x, label, rec)
                except PoolError:
                    raise
                except Exception as exc:
                    result = dict(label=label, x=np.asarray(x).tolist(), all14=rec,
                                  forecast_available=False, forecast_qualified=False,
                                  readout_error=type(exc).__name__+": "+str(exc),
                                  manifest_sha256=sha(HERE / "manifest.json"))
            else:
                result = dict(label=label, x=np.asarray(x).tolist(), all14=rec,
                              forecast_available=False, forecast_qualified=False,
                              reason="Candidate did not pass the unchanged all-14 compatibility/physical/numerical gates",
                              manifest_sha256=sha(HERE / "manifest.json"))
        seen[key] = deepcopy(result)
        items.append(result)
    return items


def search_task(task, seed_w):
    checkpoint_guard()
    started = time.perf_counter()
    problem = CappedProblem(task)
    x0 = np.asarray(seed_w["x"], dtype=float)
    y0 = (x0 - model.LOW) / WIDTH
    assert np.all(y0 >= 0) and np.all(y0 <= 1)

    def callback(y):
        problem.iterations += 1
        problem.last_iterate = np.asarray(y, dtype=float).copy()
        if problem.iterations % 10 == 0:
            checkpoint_guard()
        if time.perf_counter() - started >= MAX_OPTIMIZATION_SECONDS:
            problem.limit_reason = "optimization_wall_limit"
            raise SearchLimit(problem.limit_reason)

    optimizer = None
    limit_error = None
    try:
        optimizer = minimize(problem.objective, y0, jac=problem.gradient, method="SLSQP",
                             bounds=[(0., 1.)]*5,
                             constraints=[dict(type="ineq", fun=problem.constraint, jac=problem.constraint_jac)],
                             callback=callback,
                             options=dict(maxiter=MAX_ITERATIONS, ftol=1e-9))
    except SearchLimit as exc:
        limit_error = str(exc)
        problem.limit_reason = problem.limit_reason or limit_error
    except PoolError:
        raise
    except Exception as exc:
        limit_error = type(exc).__name__ + ": " + str(exc)
        problem.limit_reason = "optimizer_exception"
    if optimizer is not None:
        optimizer_record = dict(returned=True, success=bool(optimizer.success), status=int(optimizer.status),
                                message=str(optimizer.message), nit=int(optimizer.nit), nfev=int(optimizer.nfev),
                                njev=int(optimizer.njev), fun=float(optimizer.fun),
                                x=(model.LOW + WIDTH*np.asarray(optimizer.x)).tolist())
    else:
        optimizer_record = dict(returned=False, success=False, status=None,
                                stop_reason=problem.limit_reason, exception=limit_error,
                                last_completed_optimizer_iterate=None if problem.last_iterate is None else
                                (model.LOW + WIDTH*problem.last_iterate).tolist())
    ret = dict(id=task_id(task), task=task, seed=seed_w, optimizer=optimizer_record,
               best_feasible=problem.best, improvement_trace=problem.trace,
               evaluation_attempts=problem.evaluations, evaluation_failures=problem.failures,
               max_evaluations=MAX_EVALUATIONS, max_iterations=MAX_ITERATIONS,
               optimization_wall_limit_seconds=MAX_OPTIMIZATION_SECONDS,
               optimization_seconds=time.perf_counter()-started,
               actual_limit_reason=problem.limit_reason, manifest_sha256=sha(HERE / "manifest.json"))
    safe_json(HERE / "optimizer_returns" / f"{ret['id']}.json", ret)
    candidates = candidate_outputs(task, seed_w, optimizer, problem)
    result = dict(id=ret["id"], task=task, optimizer=optimizer_record,
                  evaluation_attempts=problem.evaluations, evaluation_failures=problem.failures,
                  optimization_seconds=ret["optimization_seconds"], total_seconds=time.perf_counter()-started,
                  actual_limit_reason=problem.limit_reason, best_feasible=problem.best,
                  candidates=candidates, manifest_sha256=sha(HERE / "manifest.json"))
    safe_json(HERE / "searches" / f"{ret['id']}.json", result)
    checkpoint_guard()
    return dict(id=ret["id"], evaluations=problem.evaluations, iterations=optimizer_record.get("nit", problem.iterations),
                limit=problem.limit_reason, seconds=result["total_seconds"],
                qualified_candidates=sum(c.get("forecast_qualified", False) for c in candidates))


def main():
    checkpoint_guard()
    frozen = freeze()
    control = controls(frozen)
    if not control["passed"]:
        safe_json(HERE / "execution.json", dict(status="CONTROL_FAILURE", controls=control,
                  manifest_sha256=sha(HERE / "manifest.json")))
        raise RuntimeError("Local-family implementation/derivative controls failed")
    tasks = []
    for entry in frozen["witnesses"]:
        w = entry["witness"]
        task = dict(setting_task(w), direction="min", seed=0, seed_id=w["id"])
        tasks.append((task, w))
    workers = int(os.environ.get("AJ_COMPUTE_WORKERS", "1"))
    assert workers == 3
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(search_task, task, w) for task, w in tasks]
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                row = dict(status="WORKER_FAILURE", error=type(exc).__name__+": "+str(exc))
            rows.append(row)
            print(json.dumps(row), flush=True)
            checkpoint_guard()
    status = "COMPLETED" if len(rows) == 3 and all(r.get("status") != "WORKER_FAILURE" for r in rows) else "INCOMPLETE"
    execution = dict(status=status, tasks=rows, completed=sum(r.get("status") != "WORKER_FAILURE" for r in rows),
                     workers=workers, wall_seconds=time.perf_counter()-started,
                     manifest_sha256=sha(HERE / "manifest.json"))
    safe_json(HERE / "execution.json", execution)
    summarize(frozen, execution)
    print(json.dumps(dict(status=status, seconds=execution["wall_seconds"])), flush=True)


def summarize(frozen, execution):
    rows = []
    for p in sorted((HERE / "searches").glob("*.json")):
        search = read(p)
        for candidate in search["candidates"]:
            available = bool(candidate.get("forecast_available", False))
            qualified = bool(candidate.get("forecast_qualified", False))
            bias_pass = candidate.get("instantaneous_bias_budget_pass")
            contrast = candidate.get("fitted_contrast")
            rows.append(dict(seed=search["task"]["seed_id"], extraction=search["task"]["extraction"],
                             registration=search["task"]["registration"], timing=search["task"]["timing"],
                             label=candidate["label"], all14_admitted=candidate["all14"]["all_known_admitted"],
                             forecast_available=available, forecast_qualified=qualified,
                             instantaneous_bias_budget_pass=bias_pass,
                             bias_budget_qualified=bool(qualified and available and bias_pass is True),
                             fitted_contrast=None if contrast is None else float(contrast),
                             contrast_pp=None if contrast is None else 100*float(contrast),
                             true_contrast_t0=candidate.get("true_contrast_t0"),
                             bias_from_t0=candidate.get("bias_from_t0"),
                             numerical_pass=candidate.get("numerical_pass"),
                             readout_error=candidate.get("readout_error"), reason=candidate.get("reason"),
                             reused_saved_seed_readout=bool(candidate.get("reused_saved_seed_readout", False))))
    accepted = [r for r in rows if r["bias_budget_qualified"] and r["fitted_contrast"] is not None]
    qualified = [r for r in rows if r["forecast_qualified"] and r["fitted_contrast"] is not None]
    nonpositive = [r for r in qualified if r["fitted_contrast"] <= 0]
    nonpositive_biasqualified = [r for r in accepted if r["fitted_contrast"] <= 0]
    old = read(STAGE35 / "summary.json")
    old_primary = next(g for g in old["groups"] if g["law"] == "fixed_q" and g["order"] == "high_to_low")
    out = dict(status=execution["status"], candidate_forecasts=rows,
               original_three_seed_contrasts_pp={entry["witness"]["id"]: 100*entry["original_case_values"]["fitted_contrast"]
                                                  for entry in frozen["witnesses"]},
               original_50_sample_range_pp=[100*old_primary["fitted_contrast_min_median_max"][0],
                                            100*old_primary["fitted_contrast_min_median_max"][2]],
               lowest_numerically_qualified_candidate_pp=min((r["contrast_pp"] for r in qualified), default=None),
               lowest_bias_budget_qualified_candidate_pp=min((r["contrast_pp"] for r in accepted), default=None),
               new_numerically_qualified_sample_range_pp=(None if not qualified else
                   [min(r["contrast_pp"] for r in qualified), max(r["contrast_pp"] for r in qualified)]),
               new_bias_budget_qualified_sample_range_pp=(None if not accepted else
                   [min(r["contrast_pp"] for r in accepted), max(r["contrast_pp"] for r in accepted)]),
               nonpositive_numerically_qualified_candidates=nonpositive,
               nonpositive_with_bias_budget_pass=nonpositive_biasqualified,
               execution=execution, controls=read(HERE / "controls.json"),
               manifest_sha256=sha(HERE / "manifest.json"),
               global_minimum_certified=False,
               negative_forecast_found=bool(nonpositive_biasqualified))
    save(HERE / "summary.json", out)


if __name__ == "__main__":
    main()
