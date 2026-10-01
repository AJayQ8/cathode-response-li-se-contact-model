#!/usr/bin/env python3
"""Verify repository hashes and arithmetic from saved outputs; never run the model."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION_PREFIX = "science_lab/papers/cathode_response/revision_20260920/"
TOL = 2e-12


def fail(message: str) -> None:
    raise SystemExit("FAIL: " + message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(relative: str):
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# Hash all published files except the checksum list itself. Ignore repository metadata,
# interpreter caches/virtual environments, and optional regenerated-figure outputs.
IGNORED_DIRECTORY_NAMES = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

def is_published_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
        return False
    if relative.parts[:2] == ("figures", "regenerated"):
        return False
    return True

checksums_path = ROOT / "SHA256SUMS.txt"
if not checksums_path.is_file():
    fail("SHA256SUMS.txt is missing")
expected = {}
for line in checksums_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        checksum, relative = line.split("  ", 1)
    except ValueError:
        fail("malformed SHA256SUMS.txt line: " + line)
    expected[relative] = checksum
actual_files = {
    str(path.relative_to(ROOT))
    for path in ROOT.rglob("*")
    if path.is_file() and path != checksums_path and is_published_file(path)
}
if set(expected) != actual_files:
    fail("SHA256SUMS.txt file inventory differs (missing=%r extra=%r)" %
         (sorted(actual_files - set(expected)), sorted(set(expected) - actual_files)))
for relative, checksum in expected.items():
    if digest((ROOT / relative).read_bytes()) != checksum:
        fail("file hash mismatch: " + relative)

export_manifest = json.loads((ROOT / "provenance/EXPORT_MANIFEST.json").read_text(encoding="utf-8"))
archive_sha = export_manifest["scientific_source_history"]["frozen_evidence_archive_sha256"]

# Validate evidence ZIP CRCs, member hashes, and the exact saved table paths.
archive_members = {}
verified_member_count = 0
for bundle_name, details in export_manifest["evidence_bundles"].items():
    path = ROOT / details["path"]
    if digest(path.read_bytes()) != details["sha256"]:
        fail("evidence bundle hash mismatch: " + details["path"])
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            fail("evidence ZIP CRC failure: %s member %s" % (details["path"], bad))
        manifest = json.loads(archive.read("EVIDENCE_MANIFEST.json"))
        if manifest["source_evidence_archive_sha256"] != archive_sha:
            fail("evidence bundle points at a different frozen archive: " + bundle_name)
        if manifest["member_count"] != len(manifest["members"]):
            fail("evidence bundle member count mismatch: " + bundle_name)
        for member in manifest["members"]:
            verified_member_count += 1
            content = archive.read(member["path"])
            if digest(content) != member["sha256"]:
                fail("evidence member hash mismatch: %s:%s" % (bundle_name, member["path"]))
            archive_members.setdefault(member["path"], set()).add(bundle_name)

# Confirm every provenance path named by the released calibration tables is present.
fit_rows = read_csv("data/calibration_938_rows.csv")
vector_rows = read_csv("data/compatible_vectors.csv")
required_source_paths = set()
for row in fit_rows:
    if row.get("fit_source"):
        required_source_paths.add(row["fit_source"])
for row in vector_rows:
    for field in ("parameter_source_path", "case_or_projection_path"):
        value = row.get(field, "").strip()
        if value:
            required_source_paths.add(value if value.startswith("science_lab/") else REVISION_PREFIX + value)
missing_paths = sorted(required_source_paths - set(archive_members))
if missing_paths:
    fail("table source path is missing from evidence bundles: " + missing_paths[0])

# Parse, but never import or execute, the archived scientific source snapshots.
source_rows = read_csv("source/SOURCE_MANIFEST.csv")
required_scientific_code = {
    "source/route_a/transfer47/qualified_scalar.py",
    "source/route_a/forecast9/run_recovery_probe.py",
    "source/route_a/kinetics7/run_kinetic_probe.py",
    "source/route_a/impedance8/run_circuit_probe.py",
    "source/route_a/readout34/run_readout.py",
    "source/route_a/weak33/run_weak_response.py",
    "source/route_a/refine43/precision_v2/run_precision.py",
    "source/route_a/robust23/resume_startup_repair.py",
    "source/route_a/robust23/run_robustness.py",
    "source/route_a/finite22/run_finite_height.py",
    "source/route_a/spatial32/run_spatial_probe.py",
    "source/route_a/prospective27/run_projection.py",
    "source/route_a/coupling5/electrical_boundary.py",
    "source/route_a/geometry20/run_geometry_probe.py",
    "source/route_a/current39/run_current.py",
    "source/route_a/interaction36/run_interaction.py",
    "source/route_a/transport4/run_transport.py",
}
exports = {row["export_path"] for row in source_rows}
missing_code = sorted(required_scientific_code - exports)
if missing_code:
    fail("scientific source dependency snapshot missing: " + missing_code[0])
for row in source_rows:
    path = ROOT / row["export_path"]
    if digest(path.read_bytes()) != row["sha256"]:
        fail("source snapshot hash mismatch: " + row["export_path"])
    if path.suffix == ".py":
        try:
            ast.parse(path.read_bytes(), filename=row["export_path"])
        except SyntaxError as exc:
            fail("Python source does not parse: %s: %s" % (row["export_path"], exc))

# Check saved observation fits and the immutable 0.03 all-14 screen.
if len(fit_rows) != 938:
    fail("calibration_938_rows.csv does not have 938 comparisons")
max_calibration_error = 0.0
for row in fit_rows:
    observed = float(row["observed_ratio"])
    predicted = float(row["predicted_ratio"])
    error = float(row["error"])
    if not close(error, predicted - observed):
        fail("calibration error is inconsistent with target and prediction")
    max_calibration_error = max(max_calibration_error, abs(error))
if max_calibration_error > 0.03 + TOL:
    fail("one or more of the 938 calibration comparisons exceeds 0.03")

if len(vector_rows) != 67:
    fail("compatible_vectors.csv does not contain 67 records")
if len({row["dedup_cluster_id"] for row in vector_rows}) != 50:
    fail("saved compatible-vector table does not contain 50 deduplication clusters")
cohort_counts = {}
for row in vector_rows:
    cohort_counts[row["cohort"]] = cohort_counts.get(row["cohort"], 0) + 1
if cohort_counts != {"original50": 50, "post50": 3, "endpoint_m": 14}:
    fail("compatible-vector archive group counts changed: %r" % cohort_counts)

# Recompute the 14-residual maximum for each bounded-search output, including the rejected return.
challenge = read_csv("data/challenge_forecasts.csv")
residuals = read_csv("data/challenge_all14_residuals.csv")
if len(challenge) != 4 or len(residuals) != 56:
    fail("challenge output is not four records with 14 residuals each")
residual_groups = {}
for row in residuals:
    observed = float(row["observed_ratio"])
    predicted = float(row["predicted_ratio"])
    residual = float(row["residual"])
    if not close(residual, predicted - observed):
        fail("challenge residual is inconsistent with target and prediction")
    residual_groups.setdefault(row["record_id"], []).append(abs(residual))
if any(len(values) != 14 for values in residual_groups.values()):
    fail("a challenge output does not have 14 residuals")
if len([row for row in challenge if row["all14_screen_pass"] == "True"]) != 3:
    fail("challenge screen pass count changed")
for row in challenge:
    values = residual_groups.get(row["record_id"])
    if values is None:
        fail("challenge residuals missing for " + row["record_id"])
    maximum = max(values)
    if not close(maximum, float(row["max_all14_abs_ratio_error"])):
        fail("challenge maximum residual mismatch for " + row["record_id"])
    passed = maximum <= 0.03 + TOL
    if passed != (row["all14_screen_pass"] == "True"):
        fail("challenge screen flag disagrees with saved residuals")
    if row["forecast_available"] == "True" and not (passed and row["physical_checks_pass"] == "True" and row["numerical_checks_pass"] == "True"):
        fail("forecast available without all required qualification flags")
failed_published = [row for row in challenge if row["extraction"] == "published"]
if len(failed_published) != 1 or failed_published[0]["forecast_available"] != "False":
    fail("failed published-extraction return is not preserved as unavailable")

# Rebuild forecast-group extrema using only the supplied compatible vectors and accepted challenge outputs.
ranges = read_csv("data/forecast_ranges.csv")
if len(ranges) != 9:
    fail("forecast_ranges.csv should contain nine exponent/extraction groups")
for row in ranges:
    m = row["m"]
    extraction = row["extraction"]
    base = [float(item["total_fitted_P_minus_N_forecast_pp"]) for item in vector_rows
            if item["m"] == m and item["extraction"] == extraction]
    added = [float(item["forecast_pp"]) for item in challenge
             if m == "6.6" and item["extraction"] == extraction and item["forecast_available"] == "True"]
    expected_count = int(row["original_n"]) + int(row["challenge_n"])
    all_values = base + added
    if len(base) != int(row["original_n"]) or len(added) != int(row["challenge_n"]):
        fail("forecast record count does not match saved source groups for %s/%s" % (m, extraction))
    if len(all_values) != int(row["n"]) or len(all_values) != expected_count:
        fail("forecast total count mismatch for %s/%s" % (m, extraction))
    if not close(min(all_values), float(row["min_pp"])) or not close(max(all_values), float(row["max_pp"])):
        fail("forecast range mismatch for %s/%s" % (m, extraction))

m66 = [row for row in ranges if row["m"] == "6.6"]
endpoint = [row for row in ranges if row["m"] in ("5.9", "7.3")]
if sum(int(row["n"]) for row in m66) != 56 or sum(int(row["n"]) for row in endpoint) != 14:
    fail("baseline/endpoint forecast record counts changed")
if not close(min(float(row["min_pp"]) for row in m66), 0.9959776026459366):
    fail("baseline lower forecast endpoint changed")
if not close(max(float(row["max_pp"]) for row in m66), 5.615423296302824):
    fail("baseline upper forecast endpoint changed")

if len(read_csv("data/load_sharing_8_rows.csv")) != 8:
    fail("saved spatial table does not contain eight stop/rest rows")
if len(read_csv("data/frequency_schedule.csv")) != 68:
    fail("saved high-to-low frequency schedule does not have 68 rows")

print("PASS: %d repository files hash-checked; %d evidence members verified; %d source manifest entries parsed." %
      (len(expected), verified_member_count, len(source_rows)))
print("PASS: 938 calibration comparisons; 67 compatible vectors in 50 dedup clusters; challenge 3 pass/1 fail; forecast ranges recomputed from saved values.")
print("No model, optimizer, fit, extraction, or simulation was imported or run.")
