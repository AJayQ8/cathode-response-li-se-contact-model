"""Frozen three-witness extension of the all50 fitted-readout attribution."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
SPLIT = HERE.parent
REV = SPLIT.parent
ROOT = Path(subprocess.check_output(
    ["git", "rev-parse", "--path-format=absolute", "--show-toplevel"],
    cwd=HERE, text=True).strip())
COMMON = Path(subprocess.check_output(
    ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
    cwd=HERE, text=True).strip())
sys.path.insert(0, str(COMMON / "aj-compute" / "current"))
from shared_compute import checkpoint_guard


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite durable output")
    return value


def save(path, value, guarded=True):
    if guarded:
        checkpoint_guard()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(plain(value), indent=2, allow_nan=False) + "\n"
    temp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temp.open("w") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


# Unchanged science and fitting modules; only the output directory is redirected.
sweep = module("post50_retained_sweep35", REV / "sweep35" / "run_sweep.py")
sweep.circuit.OUT = HERE / "fits"
decomposition_module = module("post50_retained_decomposition29",
                       REV / "decomposition29" / "run_decomposition.py")
prediction = module("post50_retained_transfer49",
                    REV / "transfer49" / "run_prediction.py")
auditor = module("post50_independent_transfer49_auditor",
                 REV / "transfer49" / "audit_prediction.py")

SETTING = dict(registration="balance_aligned", timing="late", extraction="ordinary")
CATHODES = ("P-LCO", "N-LCO")
CHARGE = .30
CURRENT = 1.2
SPECTRUM_LIMIT = 1e-8
INDEPENDENT_SPECTRUM_LIMIT = 1e-7
EXPECTED_CANDIDATES = [
    dict(key="s47_seed0_best_screen", seed=0, label="best_screen",
         case="s47_seed0",
         x=[1.4078251851728787, -2.169138710810378, 3.2027163060177837,
            0.6943401026272241, 0.9999999899999752]),
    dict(key="s47_seed0_optimizer", seed=0, label="optimizer", case=None,
         x=[1.4078252066319195, -2.169139110999475, 3.202737390380797,
            0.6943400675846597, 0.9999999899999522]),
    dict(key="s47_seed1_optimizer", seed=1, label="optimizer",
         case="s47_seed1",
         x=[1.407825215182568, -2.169139112758451, 3.202739816554419,
            0.694340063232971, 0.9999999899905719]),
]


def add_digest(hashes, path):
    path = Path(path).resolve()
    relative = str(path.relative_to(ROOT))
    digest = sha(path)
    old = hashes.get(relative)
    require(old is None or old == digest, "conflicting frozen source hash: " + relative)
    hashes[relative] = digest
    return relative, digest


def verify_manifest_sources(manifest_path, hashes):
    manifest = read(manifest_path)
    require(isinstance(manifest.get("sha256"), dict), "missing source hash map")
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        require(path.is_file() and sha(path) == expected,
                "upstream frozen source mismatch: " + relative)
        add_digest(hashes, path)
    add_digest(hashes, manifest_path)
    return manifest


def verify_receipt(stage, relative, expected_manifest=None):
    path = Path(stage) / relative
    receipt_rel = "receipts/" + relative.replace("/", "__")
    receipt_path = Path(stage) / receipt_rel
    require(path.is_file() and receipt_path.is_file(), "missing receipted artifact: " + relative)
    receipt = read(receipt_path)
    require(receipt.get("path") == relative and receipt.get("sha256") == sha(path),
            "artifact receipt mismatch: " + relative)
    if expected_manifest is not None:
        require(receipt.get("manifest_sha256") == expected_manifest,
                "artifact manifest mismatch: " + relative)
    add_digest({}, path)
    return read(path)


def candidate_sources(hashes):
    stage = REV / "transfer47"
    stage_manifest = read(stage / "manifest.json")
    manifest_sha = sha(stage / "manifest.json")
    execution = verify_receipt(stage, "execution_pilot.json", manifest_sha)
    controls = verify_receipt(stage, "controls.json", manifest_sha)
    require(execution["completed"] == 2 and len(execution["tasks"]) == 2 and controls["passed"],
            "stage47 task execution/controls are incomplete")
    witnesses = []
    for item in EXPECTED_CANDIDATES:
        task = next(t for t in stage_manifest["tasks"] if t["seed"] == item["seed"])
        require({k: task[k] for k in SETTING} == SETTING,
                "stage47 setting changed for " + item["key"])
        task_id = task["id"]
        result_rel = "results/" + task_id + ".json"
        raw_rel = "optimizer_returns/" + task_id + ".json"
        result = verify_receipt(stage, result_rel, manifest_sha)
        raw = verify_receipt(stage, raw_rel, manifest_sha)
        require(result["task"] == raw["task"] == task and
                result["optimizer_return_sha256"] == sha(stage / raw_rel) and
                result["seed"] == raw["seed"], "stage47 terminal provenance mismatch")
        candidate = next((c for c in result["candidates"] if c["label"] == item["label"]), None)
        require(candidate is not None and candidate["x"] == item["x"],
                "selected stage47 vector/label changed: " + item["key"])
        prediction.check_candidate(candidate, SETTING)
        require(candidate["qualified_witness"] and candidate["numerical_pass"] and
                candidate["physical_observation_pass"] and candidate["all_known_screen_pass"] and
                candidate["all_known_max_error"] <= .03,
                "selected vector failed the frozen scientific screen")
        require(candidate["forward"]["x"] == candidate["x"] and
                candidate["forward"]["rstar"] > 0,
                "stage47 candidate forward record mismatch")
        source = str((stage / result_rel).relative_to(ROOT))
        witness = dict(id=item["key"], original_id="stage47/" + task_id + "/" + item["label"],
                       x=candidate["x"], setting=SETTING,
                       rstar=candidate["forward"]["rstar"], eta=candidate["eta"],
                       source=source, source_sha256=sha(stage / result_rel),
                       label=candidate["label"],
                       all_known_max_error=candidate["all_known_max_error"],
                       observed_numerical_disagreement=candidate["observed_numerical_disagreement"])
        witnesses.append(dict(selection=item, witness=witness,
                              qualification=dict(qualified=candidate["qualified_witness"],
                                                 numerical_pass=candidate["numerical_pass"],
                                                 physical_observation_pass=candidate["physical_observation_pass"],
                                                 all14_max_error=candidate["all_known_max_error"],
                                                 screen=.03)))
        add_digest(hashes, stage / result_rel)
        add_digest(hashes, stage / raw_rel)
        add_digest(hashes, stage / "receipts" / result_rel.replace("/", "__"))
        add_digest(hashes, stage / "receipts" / raw_rel.replace("/", "__"))
    return witnesses


def stage49_case(case_id, expected_witness, hashes):
    stage = REV / "transfer49"
    case_path = stage / "cases" / (case_id + ".json")
    receipt = verify_receipt(stage, "cases/" + case_id + ".json", sha(stage / "manifest.json"))
    require(receipt["witness"]["x"] == expected_witness["x"] and
            receipt["witness"]["setting"] == SETTING and
            receipt["witness"]["label"] == expected_witness["label"] and
            receipt["witness"]["rstar"] == expected_witness["rstar"],
            "stage49 saved case does not match exact selected vector: " + case_id)
    require(receipt["audit"]["passed"] and receipt["available"] and receipt["readout_available"] and
            receipt["history_effect_available"],
            "stage49 case was not previously qualified: " + case_id)
    require(len(receipt["histories"]) == 2 and len(receipt["crossed_histories"]) == 2 and
            all(h["available"] and all(r["numerical_pass"] and r["physical_pass"] for r in h["rows"])
                for h in receipt["histories"] + receipt["crossed_histories"]),
            "stage49 history source incomplete: " + case_id)
    require(len(receipt["readout_rows"]) == 2 and
            all(r["selected_fit_success"] and r["quadrature_pass"] for r in receipt["readout_rows"]),
            "stage49 diagonal readout missing/failed: " + case_id)
    stage_manifest_sha = sha(stage / "manifest.json")
    add_digest(hashes, case_path)
    add_digest(hashes, stage / "receipts" / ("cases__" + case_id + ".json"))
    for row in receipt["readout_rows"]:
        fit_rel = row["fit_path"]
        wrapper = verify_receipt(stage, fit_rel, stage_manifest_sha)
        require(wrapper["fit"] == row["fit"] and
                wrapper["fit"]["task"]["loss"] == "linear" and
                wrapper["fit"]["task"]["topology"] == "source",
                "stage49 fit wrapper differs from saved diagonal: " + fit_rel)
        sid, fit_id = wrapper["fit"]["task"]["spectrum_id"], wrapper["fit"]["id"]
        raw_path = stage / "spectral_work" / sid / (fit_id + ".json")
        require(raw_path.is_file() and sha(raw_path) == wrapper["raw_sha256"],
                "stage49 raw fit receipt mismatch: " + sid)
        add_digest(hashes, raw_path)
        add_digest(hashes, stage / "receipts" / fit_rel.replace("/", "__"))
        add_digest(hashes, stage / fit_rel)
    return receipt


def freeze():
    require(SETTING == dict(registration="balance_aligned", timing="late", extraction="ordinary"),
            "Only the predeclared ordinary late interpretation is allowed")
    hashes = {}
    upstream = [
        REV / "transfer47" / "manifest.json",
        REV / "transfer49" / "manifest.json",
        REV / "sweep35" / "manifest.json",
        SPLIT / "manifest.json",
    ]
    manifests = {}
    for path in upstream:
        manifests[str(path.relative_to(REV))] = verify_manifest_sources(path, hashes)
    old_summary = read(SPLIT / "summary.json")
    require(old_summary["witnesses"] == 50 and old_summary["all_numerical_pass"],
            "original all50 cohort is not terminal/pass")
    require(len(list(csv.DictReader((SPLIT / "per_witness.csv").open()))) == 50,
            "original all50 witness table changed")
    for name in ["FINDINGS.md", "per_witness.csv", "summary.json", "audit.json",
                 "artifact_manifest.json"]:
        add_digest(hashes, SPLIT / name)

    selected = candidate_sources(hashes)
    old_cases = {}
    for item in selected:
        case_id = item["selection"]["case"]
        if case_id is not None:
            old_cases[case_id] = stage49_case(case_id, item["witness"], hashes)
    require(set(old_cases) == {"s47_seed0", "s47_seed1"},
            "Expected only the two existing selected stage49 forecasts")

    # Add the frozen code/protocol/source data explicitly, including the
    # independent trajectory and circuit audit implementations.
    explicit = [
        SPLIT / "POST50_EVIDENCE_AUDIT.md",
        SPLIT / "PLAN.md",
        SPLIT / "run_forecast_split.py",
        SPLIT / "audit_saved.py",
        HERE / "PLAN.md",
        HERE / "run_post50_extension.py",
        HERE / "job.json",
        REV / "sweep35" / "run_sweep.py",
        REV / "sweep35" / "controls.json",
        REV / "sweep35" / "summary.json",
        REV / "impedance8" / "run_circuit_probe.py",
        REV / "impedance8" / "spectra.json",
        REV / "impedance8" / "fits" / "P_2C_Ch__source__linear__0.json",
        REV / "impedance8" / "fits" / "N_2C_Ch__source__linear__0.json",
        REV / "readout34" / "run_readout.py",
        REV / "decomposition29" / "run_decomposition.py",
        REV / "decomposition29" / "manifest.json",
        REV / "decomposition29" / "controls.json",
        REV / "prospective27" / "run_projection.py",
        REV / "robust23" / "resume_startup_repair.py",
        REV / "transfer49" / "PLAN.md",
        REV / "transfer49" / "run_prediction.py",
        REV / "transfer49" / "audit_prediction.py",
        REV / "mechanics3" / "waveforms.csv",
        REV / "mechanics3" / "waveform_summary.json",
        REV / "mechanics3" / "fullcell_charge_history.json",
        REV / "mechanics3" / "mechanical_checks.json",
    ]
    for path in explicit:
        require(path.is_file(), "missing explicit frozen source: " + str(path))
        add_digest(hashes, path)

    # Exact factor labels and no outcome-based selection are part of the freeze.
    vector_records = []
    for item in selected:
        vector_records.append(dict(key=item["selection"]["key"],
                                   seed=item["selection"]["seed"],
                                   label=item["selection"]["label"],
                                   x=item["witness"]["x"],
                                   rstar=item["witness"]["rstar"],
                                   qualification=item["qualification"],
                                   prior_stage49_case=item["selection"]["case"],
                                   selection_uses_forecast_outcome=False))
    out = dict(schema=1, sha256=hashes, setting=SETTING,
               candidates=vector_records, all_three_retained=True,
               original_cohort=50, combined_cohort_label="descriptive 53 entries",
               screen=0.03, current_mA_cm2=CURRENT, discharge_mAh_cm2=CHARGE,
               discharge_hours=CHARGE/CURRENT, readout_order="high_to_low",
               law="fixed_q", frequencies=68, settling_seconds=.5,
               cycles_per_frequency=3, quadrature_orders=[8, 16],
               quadrature_relative_limit=SPECTRUM_LIMIT, fit_loss="linear",
               fit_topology="source", six_starts=True, fit_tolerances=1e-13,
               reused_stage49_cases=sorted(old_cases),
               new_crossed_spectra=4, seed0_optimizer_new_histories=4,
               seed0_optimizer_new_spectra=4, total_new_spectra=8,
               kinetic_refits=0, historical_output_mutation=False,
               global_bound=False, experimental_validation=False,
               independent_checks=dict(trajectory="transfer49/audit_prediction.py",
                                       spectrum="16-point direct reconstruction on independent trajectory",
                                       fit="reconstructed six-start source-circuit objectives"),
               runtime=dict(python=platform.python_version(), numpy=np.__version__,
                            scipy=scipy.__version__),
               selection_source="POST50_EVIDENCE_AUDIT.md and exact stage47 labels")
    target = HERE / "manifest.json"
    if target.exists():
        require(read(target) == out, "frozen post50 source/selection manifest changed")
    else:
        require(not any((HERE / name).exists() for name in
                        ["cases", "fits", "summary.json", "combined53.json", "preflight.json"]),
                "post50 outputs exist without matching manifest")
        save(target, out, guarded=False)
    return out


def verify_frozen():
    manifest = read(HERE / "continuation_v2_manifest.json")
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        require(path.is_file() and sha(path) == expected, "frozen extension source changed: " + relative)
    for absolute, expected in manifest["external_sources"].items():
        path = Path(absolute)
        require(path.is_file() and sha(path) == expected, "frozen launcher evidence changed: " + absolute)
    require(manifest["parent_manifest_sha256"] == sha(HERE / "manifest.json"),
            "parent v1 manifest changed")
    require(len(manifest["candidates"]) == 3 and manifest["new_crossed_spectra"] == 4 and
            manifest["total_new_spectra"] == 8 and manifest["kinetic_refits"] == 0,
            "frozen numerical contract changed")
    return manifest


def freeze_continuation():
    parent_path = HERE / "manifest.json"
    parent = read(parent_path)
    parent_sha = sha(parent_path)
    hashes = dict(parent["sha256"])
    for relative, expected in hashes.items():
        path = ROOT / relative
        require(path.is_file() and sha(path) == expected, "parent v1 dependency changed: " + relative)
    add_digest(hashes, parent_path)
    note = HERE / "REPAIR_01.md"
    runner_v1 = HERE / "run_post50_extension.py"
    runner_v2 = HERE / "run_post50_extension_v2.py"
    job_v2 = HERE / "job_v2.json"
    for path in (note, runner_v1, runner_v2, job_v2):
        require(path.is_file(), "missing repair continuation source: " + str(path))
        add_digest(hashes, path)

    # Preserve and bind the completed v1 case and the two fits produced before
    # the operational exception.
    prior_case_rel = "cases/s47_seed0_best_screen.json"
    prior_case_path = HERE / prior_case_rel
    require(prior_case_path.is_file(), "completed first v1 case is missing")
    prior_case = read(prior_case_path)
    require(prior_case["manifest_sha256"] == parent_sha and
            prior_case["witness"]["id"] == "s47_seed0_best_screen" and
            prior_case["numerical_pass"],
            "completed v1 case does not match parent manifest or checks")
    preserved = {prior_case_rel: sha(prior_case_path)}
    add_digest(hashes, prior_case_path)
    for cell_key, cell in prior_case["cells"].items():
        if not cell["reused"]:
            fit_path = HERE / cell["fit_path"]
            require(fit_path.is_file() and sha(fit_path) == cell["fit_sha256"],
                    "completed v1 fit is missing or changed: " + cell_key)
            preserved[cell["fit_path"]] = sha(fit_path)
            add_digest(hashes, fit_path)

    host_dir = COMMON / "aj-compute" / "hosts" / "bd131eabf0960011"
    run_id = "218af0fc71dd4702af62c0e1cadb2956"
    receipt_path = host_dir / "receipts" / (run_id + ".json")
    log_dir = host_dir / "logs" / run_id
    receipt = read(receipt_path)
    require(receipt["status"] == "failed" and receipt["exit_status"] == 1 and
            receipt["id"] == run_id, "v1 failure receipt changed or not terminal")
    stdout_path = log_dir / "running.stdout.log"
    stderr_path = log_dir / "running.stderr.log"
    stdout = stdout_path.read_text()
    stderr = stderr_path.read_text()
    require("s47_seed0_best_screen" in stdout and
            "AttributeError: 'function' object has no attribute 'project'" in stderr,
            "v1 failure evidence does not match the repaired implementation defect")
    external_sources = {str(path.resolve()): sha(path) for path in
                        (receipt_path, stdout_path, stderr_path)}
    continuation = dict(schema=1, repair_id="operational-repair-01",
                        parent_manifest_sha256=parent_sha, sha256=hashes,
                        external_sources=external_sources,
                        failed_run_id=run_id,
                        preserved_prior_outputs=preserved,
                        candidates=parent["candidates"],
                        setting=parent["setting"],
                        reused_stage49_cases=parent["reused_stage49_cases"],
                        new_crossed_spectra=parent["new_crossed_spectra"],
                        seed0_optimizer_new_histories=parent["seed0_optimizer_new_histories"],
                        seed0_optimizer_new_spectra=parent["seed0_optimizer_new_spectra"],
                        total_new_spectra=parent["total_new_spectra"],
                        kinetic_refits=0, historical_output_mutation=False,
                        checks_unchanged=True,
                        continuation_inputs=sorted(preserved),
                        runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                     scipy=scipy.__version__))
    target = HERE / "continuation_v2_manifest.json"
    if target.exists():
        require(read(target) == continuation,
                "continuation source/repair/output evidence changed; retain this version")
    else:
        save(target, continuation, guarded=False)
    return continuation


def preflight():
    m = verify_frozen()
    old_cases = {}
    stage49 = REV / "transfer49"
    for case_id in m["reused_stage49_cases"]:
        case = verify_receipt(stage49, "cases/" + case_id + ".json",
                              sha(stage49 / "manifest.json"))
        require(case["audit"]["passed"] and case["available"] and case["readout_available"],
                "reused stage49 case failed current receipt check")
        old_cases[case_id] = case
    receipt = dict(status="GO", integration_advances_executed_by_preflight=0,
                    peak_ram_estimate_bytes=3 * 1024**3,
                    worst_case_remaining_persistent_output_bytes=int(.05 * 1024**3),
                    largest_transient_write_bytes=int(.01 * 1024**3),
                    filesystem_volatility_allowance_bytes=int(.05 * 1024**3),
                    untouched_reserve_bytes=10 * 1024**3,
                    planned_active_hours=.1, candidates=3,
                    verified_reused_stage49_cases=sorted(old_cases),
                    verified_source_hashes=len(m["sha256"]),
                    kinetic_refits=0, expected_new_spectra=8)
    save(HERE / "preflight_v2.json", receipt, guarded=False)
    print(json.dumps(receipt, sort_keys=True))


def stage47_candidates():
    m = read(HERE / "manifest.json")
    stage = REV / "transfer47"
    output = []
    for spec in EXPECTED_CANDIDATES:
        task = next(t for t in read(stage / "manifest.json")["tasks"] if t["seed"] == spec["seed"])
        result = read(stage / "results" / (task["id"] + ".json"))
        c = next(c for c in result["candidates"] if c["label"] == spec["label"])
        w = dict(id=spec["key"], original_id="stage47/" + task["id"] + "/" + spec["label"],
                 x=c["x"], setting=SETTING, rstar=c["forward"]["rstar"], eta=c["eta"],
                 source="science_lab/papers/cathode_response/revision_20260920/transfer47/results/" + task["id"] + ".json",
                 source_sha256=sha(stage / "results" / (task["id"] + ".json")),
                 label=spec["label"], all_known_max_error=c["all_known_max_error"],
                 observed_numerical_disagreement=c["observed_numerical_disagreement"])
        require(w["x"] == spec["x"], "candidate differs from frozen selector")
        output.append(w)
    require([x["key"] for x in m["candidates"]] == [x["id"] for x in output],
            "candidate order changed")
    return output


def make_templates():
    controls = read(REV / "sweep35" / "controls.json")
    require(controls["passed"], "stage35 source controls failed")
    refs = {}
    for cathode in CATHODES:
        ref = next(x for x in controls["records"]
                   if x["extraction"] == "ordinary" and x["cathode"] == cathode)
        require(ref["passed"], "ordinary stationary reference failed: " + cathode)
        refs[cathode] = ref
    return refs


def map_histories(case):
    cells = {}
    for row in case["histories"]:
        cells[row["cathode"][0] * 2] = row
    for row in case["crossed_histories"]:
        cells[row["initial_cathode"][0] + row["history_cathode"][0]] = row
    require(set(cells) == {"PP", "PN", "NP", "NN"}, "wrong stage49 history cells")
    return cells


def new_history(witness, initial, history):
    result = decomposition_module.project(witness, initial, history)
    require(result["initial_cathode"] == initial and result["history_cathode"] == history,
            "decomposition29 returned mismatched factor labels")
    if initial == history:
        # Stage49's independent auditor uses the legacy diagonal key; retain
        # both explicit stage29 labels and add this checked alias.
        result["cathode"] = initial
    return result


def make_cell_spectrum(witness, key, cell_key, history, reference, readout=None):
    initial = "P-LCO" if cell_key[0] == "P" else "N-LCO"
    history_cathode = "P-LCO" if cell_key[1] == "P" else "N-LCO"
    require(history["available"], "unavailable history: " + cell_key)
    require(history.get("initial_cathode", initial) == initial and
            history.get("history_cathode", history_cathode) == history_cathode,
            "history cell mapping is reversed: " + cell_key)
    history_row = next(x for x in history["rows"] if x["stage"] == "end_discharge")
    require(history_row["numerical_pass"] and history_row["physical_pass"],
            "end-discharge history did not pass: " + cell_key)
    ratio_t0 = history_row["independent_ratio"]
    require(math.isfinite(ratio_t0) and ratio_t0 > 0 and
            abs(ratio_t0-history_row["ratio"]) <= 1e-7,
            "primary/independent end-discharge ratio mismatch: " + cell_key)
    Rch = history["R_ch"]
    template = sweep.make_template("ordinary", initial, Rch)
    require(template == reference["template"], "initial-state spectrum template changed")
    fitted_reference = reference["fitted_reference"]
    if readout is not None:
        require(readout["cathode"] == initial and readout["template"] == template and
                readout["selected_fit_success"] and readout["quadrature_pass"],
                "reused diagonal spectrum/template failed: " + cell_key)
        require(abs(readout["true_ratio_t0"]-ratio_t0) <= 1e-12,
                "reused true_ratio_t0 differs from independent physical endpoint: " + cell_key)
        row = dict(initial_cathode=initial, history_cathode=history_cathode,
                   R_ch=Rch, q0=history["q0"], template=template,
                   fitted_reference=fitted_reference,
                   fitted_ratio=readout["fitted_ratio"],
                   true_ratio_t0=readout["true_ratio_t0"],
                   fitted_resistance=(readout["fitted_ratio"] * fitted_reference),
                   spectrum=dict(re_ohm_cm2=readout["re_ohm_cm2"],
                                 minus_im_ohm_cm2=readout["minus_im_ohm_cm2"],
                                 direct16_re_ohm_cm2=readout["direct16_re_ohm_cm2"],
                                 direct16_minus_im_ohm_cm2=readout["direct16_minus_im_ohm_cm2"]),
                   fit=readout["fit"], fit_path=readout["fit_path"],
                   fit_sha256=None, reused=True)
        return row

    f = sweep.frequency()
    schedule, duration = sweep.schedule(f, "high_to_low")
    require(len(f) == 68 and abs(duration - 106.9561255903888) <= 1e-10,
            "primary acquisition duration/frequencies changed")
    x = np.asarray(witness["x"], dtype=float)
    H, A = 10**x[1], x[4]
    pfast, pdirect = sweep.rest.pressures(witness, history_cathode, history["q0"] + CHARGE)
    gamma = H * A**(sweep.rest.model.M + 1) * (pdirect / 2)**sweep.rest.model.M
    contact = history_row["a"]
    ratio = history_row["independent_ratio"]
    adapter = dict(cathode=history_cathode, R_ch=Rch, q0=history["q0"],
                   gammas_per_hour=[gamma, gamma],
                   rows=[dict(independent=dict(contact=[contact], ratio=ratio))])
    z, points = sweep.generate(witness, adapter, template, "fixed_q", schedule, f, 8)
    z16, _ = sweep.generate(witness, adapter, template, "fixed_q", schedule, f, 16, True)
    quad = float(np.max(np.abs(z-z16) / np.abs(z16)))
    task = dict(spectrum_id="post50_" + key + "_" + cell_key,
                topology="source", loss="linear", fraction=0.)
    fit_path = HERE / "fits" / (sweep.circuit.task_id(task) + ".json")
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    spectrum = dict(re_ohm_cm2=z.real.tolist(), minus_im_ohm_cm2=(-z.imag).tolist(),
                    source_bulk_li_r=template["Rref"], source_cathode_r=template["slow_r"])
    if not fit_path.exists():
        checkpoint_guard()
        sweep.circuit.fit_one(task, spectrum, f)
    fit = read(fit_path)
    require(fit["task"] == task and len(fit["starts"]) == 6,
            "new spectrum fit task/start count mismatch")
    Rfit = fit["best"]["parameters"]["fast"]["r_ohm_cm2"]
    output = dict(initial_cathode=initial, history_cathode=history_cathode,
                  R_ch=Rch, q0=history["q0"], template=template,
                  fitted_reference=fitted_reference, fitted_resistance=Rfit,
                  fitted_ratio=Rfit/fitted_reference, true_ratio_t0=ratio,
                  spectrum=dict(re_ohm_cm2=z.real.tolist(),
                                minus_im_ohm_cm2=(-z.imag).tolist(),
                                direct16_re_ohm_cm2=z16.real.tolist(),
                                direct16_minus_im_ohm_cm2=(-z16.imag).tolist()),
                  points=points, duration_seconds=duration,
                  quadrature_direct_error=quad, quadrature_pass=bool(quad <= SPECTRUM_LIMIT),
                  fit=fit, fit_path=str(fit_path.relative_to(HERE)),
                  fit_sha256=sha(fit_path), reused=False)
    return output


def independent_spectrum_checks(witness, cells, results, frequency, checks):
    nodes, weights = np.polynomial.legendre.leggauss(16)
    schedule, duration = sweep.schedule(frequency, "high_to_low")
    maximum = 0.0
    records = []
    for key in ("PP", "PN", "NP", "NN"):
        cell = cells[key]
        history = cell["history"]
        initial = cell["initial"]
        history_cathode = cell["history_cathode"]
        row = results[key]
        rest, observe, trajectory = auditor._trajectory(
            witness, initial, history, checks, history_cathode)
        template = row["template"]
        reconstructed = []
        for k, item in enumerate(schedule):
            if k % 16 == 0:
                checkpoint_guard()
            seconds = (item["measure_start_s"] + item["measure_end_s"])/2
            seconds = seconds + nodes*(item["measure_end_s"]-item["measure_start_s"])/2
            observed = observe(rest.sol(seconds/3600), CHARGE)
            reconstructed.append(np.dot(weights,
                auditor._fixed_q(frequency[k], observed["resistance"], template))/2)
        reconstructed = np.asarray(reconstructed)
        z8 = np.asarray(row["spectrum"]["re_ohm_cm2"]) - 1j*np.asarray(row["spectrum"]["minus_im_ohm_cm2"])
        z16 = np.asarray(row["spectrum"]["direct16_re_ohm_cm2"]) - 1j*np.asarray(row["spectrum"]["direct16_minus_im_ohm_cm2"])
        e8 = float(np.max(np.abs(z8-reconstructed)/np.abs(reconstructed)))
        e16 = float(np.max(np.abs(z16-reconstructed)/np.abs(reconstructed)))
        maximum = max(maximum, e8, e16)
        checks.close(key + "/independent_spectrum8", e8, INDEPENDENT_SPECTRUM_LIMIT,
                     "independent_spectrum_error")
        checks.close(key + "/independent_spectrum16", e16, INDEPENDENT_SPECTRUM_LIMIT,
                     "independent_spectrum_error")
        records.append(dict(cell=key, independent_duration_seconds=duration,
                            max_relative_error_8=e8, max_relative_error_16=e16,
                            independent_trajectory=trajectory))
    return dict(records=records, maximum_relative_error=maximum,
                passed=all(t["passed"] for t in checks.tests))


def decompose(values):
    pp, pn, np_, nn = [values[k] for k in ("PP", "PN", "NP", "NN")]
    raw = dict(total=pp-nn, history_at_N_initial=np_-nn,
               initial_at_N_history=pn-nn, interaction=pp-pn-np_+nn,
               history_at_P_initial=pp-pn, initial_at_P_history=pp-np_,
               history_symmetric=((np_-nn)+(pp-pn))/2,
               initial_symmetric=((pn-nn)+(pp-np_))/2)
    return dict(components_pp={k:100*v for k, v in raw.items()},
                baseline_closure_pp=100*(raw["total"]-raw["history_at_N_initial"]-
                                         raw["initial_at_N_history"]-raw["interaction"]),
                symmetric_closure_pp=100*(raw["total"]-raw["history_symmetric"]-
                                           raw["initial_symmetric"]),
                symmetric_history_fraction=(None if raw["total"] == 0 else
                                            raw["history_symmetric"]/raw["total"]))


def process_witness(witness, selection, refs, old_cases, manifest_sha):
    checkpoint_guard()
    started = time.perf_counter()
    old_id = selection["case"]
    if old_id is not None:
        source_case = old_cases[old_id]
        histories = map_histories(source_case)
    else:
        histories = {}
        for initial, history in [("P-LCO", "P-LCO"), ("P-LCO", "N-LCO"),
                                 ("N-LCO", "P-LCO"), ("N-LCO", "N-LCO")]:
            key = initial[0] + history[0]
            histories[key] = new_history(witness, initial, history)
            checkpoint_guard()

    cells = {}
    result_cells = {}
    previous_rows = {}
    if old_id is not None:
        source_case = old_cases[old_id]
        previous_rows = {r["cathode"]: r for r in source_case["readout_rows"]}
    for key in ("PP", "PN", "NP", "NN"):
        initial = "P-LCO" if key[0] == "P" else "N-LCO"
        history_cathode = "P-LCO" if key[1] == "P" else "N-LCO"
        h = histories[key]
        cells[key] = dict(initial=initial, history_cathode=history_cathode, history=h)
        reused_row = previous_rows.get(initial) if (old_id is not None and key[0] == key[1]) else None
        row = make_cell_spectrum(witness, witness["id"], key, h, refs[initial], reused_row)
        result_cells[key] = row
        checkpoint_guard()

    f = sweep.frequency()
    checks = auditor.Checks()
    fit_audits = []
    for key in ("PP", "PN", "NP", "NN"):
        row = result_cells[key]
        z = np.asarray(row["spectrum"]["re_ohm_cm2"]) - 1j*np.asarray(row["spectrum"]["minus_im_ohm_cm2"])
        fit_record = row["fit"]
        fitted, report = auditor._audit_fit(
            fit_record, z, f, row["template"], witness["id"] + "/" + key, checks)
        fitted_reference = row["fitted_reference"]
        checks.close(witness["id"] + "/" + key + "/ratio",
                     abs(fitted/fitted_reference-row["fitted_ratio"]), 1e-12)
        if not row["reused"]:
            qerr = row["quadrature_direct_error"]
            checks.close(witness["id"] + "/" + key + "/quadrature", qerr,
                         SPECTRUM_LIMIT)
        fit_audits.append(report)

    physical_audit = auditor.audit_swaps(
        witness,
        [histories["PP"], histories["NN"]],
        [histories["PN"], histories["NP"]])
    # Keep the full independent stage49 audit, including any adverse tests.
    if physical_audit["passed"]:
        checks.flag(witness["id"] + "/physical_audit_swaps", True)
    else:
        checks.flag(witness["id"] + "/physical_audit_swaps", False,
                    errors=physical_audit.get("errors", []))

    spectrum_audit = independent_spectrum_checks(witness, cells, result_cells, f, checks)
    fitted = {k: result_cells[k]["fitted_ratio"] for k in result_cells}
    true = {k: result_cells[k]["true_ratio_t0"] for k in result_cells}
    fitted_split = decompose(fitted)
    true_split = decompose(true)
    closure = max(abs(fitted_split["baseline_closure_pp"]),
                  abs(fitted_split["symmetric_closure_pp"]),
                  abs(true_split["baseline_closure_pp"]),
                  abs(true_split["symmetric_closure_pp"]))
    checks.close(witness["id"] + "/decomposition_closure", closure, 1e-10)
    numerical = all(t["passed"] for t in checks.tests)
    output = dict(witness=witness, selection=selection,
                  history_sources=dict(prior_stage49_case=old_id,
                                       generated_by="decomposition29.project" if old_id is None else "receipted transfer49 case"),
                  histories={k: histories[k] for k in ("PP", "PN", "NP", "NN")},
                  cells=result_cells,
                  decompositions=dict(fitted_ratio=fitted_split, true_ratio_t0=true_split),
                  independent_physical_audit=physical_audit,
                  independent_spectrum_audit=spectrum_audit,
                  independent_fit_audits=fit_audits, checks=checks.tests,
                  numerical_pass=bool(numerical),
                  wall_seconds=time.perf_counter()-started,
                  manifest_sha256=manifest_sha)
    save(HERE / "cases" / (witness["id"] + ".json"), output)
    return output


def summary_statistics(rows, field):
    metrics = ("total", "history_at_N_initial", "initial_at_N_history", "interaction",
               "history_at_P_initial", "initial_at_P_history", "history_symmetric",
               "initial_symmetric")
    summary = {}
    for metric in metrics:
        values = np.asarray([r["decompositions"][field]["components_pp"][metric] for r in rows],
                            dtype=float)
        summary[metric] = dict(count=int(len(values)), minimum=float(np.min(values)),
                               median=float(np.median(values)), maximum=float(np.max(values)),
                               positive=int(np.sum(values > 0)),
                               negative=int(np.sum(values < 0)),
                               zero=int(np.sum(values == 0)))
    fractions = np.asarray([r["decompositions"][field]["symmetric_history_fraction"] for r in rows],
                           dtype=float)
    summary["symmetric_history_fraction"] = dict(
        count=int(len(fractions)), minimum=float(np.min(fractions)),
        median=float(np.median(fractions)), maximum=float(np.max(fractions)))
    return summary


def csv_row(record):
    row = dict(witness=record["witness"]["id"],
               registration=record["witness"]["setting"]["registration"],
               timing=record["witness"]["setting"]["timing"],
               extraction=record["witness"]["setting"]["extraction"])
    for key in ("PP", "PN", "NP", "NN"):
        cell = record["cells"][key]
        row[key + "_fitted_ratio"] = cell["fitted_ratio"]
        row[key + "_true_ratio"] = cell["true_ratio_t0"]
    for field in ("fitted_ratio", "true_ratio_t0"):
        for name, value in record["decompositions"][field]["components_pp"].items():
            row[field + "_" + name + "_pp"] = value
    return row


def write_summaries(manifest, records):
    original_rows = list(csv.DictReader((SPLIT / "per_witness.csv").open()))
    require(len(original_rows) == 50, "original all50 table changed after freeze")
    summary = dict(candidates=len(records), all_three_retained=True,
                   all_numerical_pass=all(r["numerical_pass"] for r in records),
                   failed_ids=[r["witness"]["id"] for r in records if not r["numerical_pass"]],
                   separately_reported=[r["witness"]["id"] for r in records],
                   fitted_ratio=summary_statistics(records, "fitted_ratio"),
                   true_ratio_t0=summary_statistics(records, "true_ratio_t0"),
                   source_original50=dict(manifest_sha256=sha(SPLIT / "manifest.json"),
                                          summary_sha256=sha(SPLIT / "summary.json"),
                                          per_witness_sha256=sha(SPLIT / "per_witness.csv")),
                   manifest_sha256=sha(HERE / "continuation_v2_manifest.json"))
    save(HERE / "summary.json", summary)
    ext_rows = [csv_row(r) for r in records]
    names = list(original_rows[0])
    with (HERE / "per_witness.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(ext_rows)

    combined = original_rows + ext_rows
    combined_summary = dict(label="descriptive pooled entries only; not independent samples",
                             original50_count=len(original_rows), post50_count=len(ext_rows),
                             total_entries=len(combined),
                             post50_keys=[r["witness"] for r in ext_rows],
                             near_duplicate_parameter_provenance=[
                                 ["s47_seed0_best_screen", "s47_seed0_optimizer"],
                                 ["s47_seed0_optimizer", "s47_seed1_optimizer"]],
                             entries_are_independent=False,
                             original50_manifest_sha256=sha(SPLIT / "manifest.json"),
                             post50_manifest_sha256=sha(HERE / "continuation_v2_manifest.json"),
                             fitted_ratio={}, true_ratio_t0={})
    for field in ("fitted_ratio", "true_ratio_t0"):
        combined_summary[field] = {}
        for metric in ("total", "history_at_N_initial", "initial_at_N_history", "interaction",
                       "history_at_P_initial", "initial_at_P_history", "history_symmetric",
                       "initial_symmetric"):
            col = field + "_" + metric + "_pp"
            values = np.asarray([float(r[col]) for r in combined], dtype=float)
            combined_summary[field][metric] = dict(count=len(values),
                minimum=float(np.min(values)), median=float(np.median(values)),
                maximum=float(np.max(values)), positive=int(np.sum(values > 0)),
                negative=int(np.sum(values < 0)), zero=int(np.sum(values == 0)))
        frac = np.asarray([
            float(r[field + "_history_symmetric_pp"]) / float(r[field + "_total_pp"])
            for r in combined], dtype=float)
        combined_summary[field]["symmetric_history_fraction"] = dict(
            count=len(frac), minimum=float(np.min(frac)), median=float(np.median(frac)),
            maximum=float(np.max(frac)))
    save(HERE / "combined53.json", combined_summary)


def run_all():
    manifest = verify_frozen()
    receipt = read(HERE / "preflight_v2.json")
    require(receipt["status"] == "GO" and receipt["integration_advances_executed_by_preflight"] == 0 and
            receipt["verified_source_hashes"] == len(manifest["sha256"]),
            "fresh zero-step preflight missing or mismatched")
    old_cases = {}
    stage49 = REV / "transfer49"
    for case_id in manifest["reused_stage49_cases"]:
        old_cases[case_id] = verify_receipt(stage49, "cases/" + case_id + ".json",
                                             sha(stage49 / "manifest.json"))
    witnesses = stage47_candidates()
    refs = make_templates()
    results = []
    for witness, spec in zip(witnesses, EXPECTED_CANDIDATES):
        case_path = HERE / "cases" / (witness["id"] + ".json")
        if case_path.exists():
            result = read(case_path)
            digest = sha(case_path)
            parent_case = spec["key"] == "s47_seed0_best_screen" and                 result["manifest_sha256"] == manifest["parent_manifest_sha256"] and                 manifest["preserved_prior_outputs"].get("cases/s47_seed0_best_screen.json") == digest
            continuation_case = result["manifest_sha256"] == sha(HERE / "continuation_v2_manifest.json")
            require((parent_case or continuation_case) and
                    result["witness"]["x"] == witness["x"] and result["numerical_pass"],
                    "existing exact extension case differs; retain and inspect")
            if parent_case:
                for cell_key, cell in result["cells"].items():
                    if not cell["reused"]:
                        require(manifest["preserved_prior_outputs"].get(cell["fit_path"]) == sha(HERE / cell["fit_path"]),
                                "preserved v1 fit differs from continuation chain: " + cell_key)
        else:
            result = process_witness(witness, spec, refs, old_cases,
                                     sha(HERE / "continuation_v2_manifest.json"))
        results.append(result)
        print(json.dumps(dict(id=witness["id"], numerical_pass=result["numerical_pass"],
                              seconds=result["wall_seconds"])), flush=True)
    write_summaries(manifest, results)
    execution = dict(completed=len(results), passed=all(x["numerical_pass"] for x in results),
                     witnesses=[x["witness"]["id"] for x in results],
                     workers=int(os.environ.get("AJ_COMPUTE_WORKERS", "1")),
                     total_wall_seconds=float(sum(x["wall_seconds"] for x in results)),
                     expected_new_spectra=8, kinetic_refits=0,
                     manifest_sha256=sha(HERE / "continuation_v2_manifest.json"))
    save(HERE / "execution.json", execution)
    require(len(results) == 3, "one of the frozen selected vectors was not reported")
    return execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["freeze-v2", "preflight", "all"], required=True)
    args = parser.parse_args()
    if args.phase == "freeze-v2":
        result = freeze_continuation()
        print(json.dumps(dict(status="FROZEN-V2", candidates=len(result["candidates"]),
                              hashes=len(result["sha256"]), preserved=list(result["preserved_prior_outputs"]),
                              sha256=sha(HERE / "continuation_v2_manifest.json"))))
    elif args.phase == "preflight":
        preflight()
    else:
        execution = run_all()
        print(json.dumps(execution, sort_keys=True))


if __name__ == "__main__":
    main()
