#!/usr/bin/env python3
"""Verify key manuscript numbers from the public CSV archive."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from li_se_reduced_model import HillFit, boundary_from_fits  # noqa: E402


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def assert_close(label: str, actual: float, expected: float, tolerance: float = 5.0e-4) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")
    print(f"{label}: {actual:.6g}")


def load_hill_fits(path: Path) -> list[HillFit]:
    return [
        HillFit(
            cathode=row["cathode"],
            jmax_ma_cm2=f(row, "jmax_ma_cm2"),
            transition_pressure_mpa=f(row, "transition_pressure_mpa"),
            exponent=f(row, "exponent"),
            floor_ma_cm2=f(row, "floor_ma_cm2"),
        )
        for row in rows(path)
    ]


def main() -> None:
    source = DATA / "source_workbook_extractions"
    supplementary = DATA / "supplementary_data"
    figure_data = DATA / "manuscript_figure_data"

    ccd = rows(source / "ccd_boundary.csv")
    fits = load_hill_fits(source / "hill_boundary_fit_parameters.csv")
    n_1mpa = next(f(row, "ccd_ma_cm2") for row in ccd if row["cathode"] == "N-LCO" and f(row, "pressure_mpa") == 1.0)
    p_1mpa = next(f(row, "ccd_ma_cm2") for row in ccd if row["cathode"] == "P-LCO" and f(row, "pressure_mpa") == 1.0)
    assert_close("CCD N/P ratio at 1 MPa", n_1mpa / p_1mpa, 4.6666666667)
    assert_close("P-LCO Hill boundary at 1 MPa", float(boundary_from_fits(fits, [1.0], -1.0)[0]), 1.5564963731)
    assert_close("N-LCO Hill boundary at 1 MPa", float(boundary_from_fits(fits, [1.0], 1.0)[0]), 6.9992014257)

    ratios = {row["metric"]: row for row in rows(source / "source_workbook_consistency_ratios.csv")}
    assert_close("EIS P/N ratio", f(ratios["EIS P/N"], "value"), 1.6048174628)
    assert_close("Profilometry RMS P/N ratio", f(ratios["Profilometry RMS P/N"], "value"), 3.0651763944)

    stress_rows = {row["case"]: row for row in rows(figure_data / "figure3_stress_triad_final_void_fraction.csv")}
    assert_close("Opening-biased final void fraction", f(stress_rows["void-opening"], "mean"), 0.9348301372)
    assert_close("Neutral final void fraction", f(stress_rows["neutral"], "mean"), 0.5185537390)
    assert_close("Filling-biased final void fraction", f(stress_rows["void-filling"], "mean"), 0.1628249952)

    baseline = {row["model variant"]: row for row in rows(supplementary / "feedback_amplification" / "baseline.csv")}
    assert_close("CCD-boundary-only risk separation", f(baseline["ccd_boundary_only"], "value"), 0.3199730776)
    assert_close("Full-model risk separation", f(baseline["full_coupled_model"], "value"), 0.6298137379)
    assert_close("Full/CCD-only amplification ratio", f(baseline["full_over_ccd_boundary_only"], "value"), 1.9683335315)

    field = rows(supplementary / "feedback_amplification" / "field.csv")
    additive = [f(row, "absolute_amplification") for row in field]
    positive_count = sum(value > 0 for value in additive)
    median_additive = sorted(additive)[len(additive) // 2 - 1 : len(additive) // 2 + 1]
    assert_close("Positive additive-amplification cells", float(positive_count), 30.0, tolerance=0.0)
    assert_close("Median additive amplification", sum(median_additive) / len(median_additive), 0.1044958803)

    thresholds = rows(supplementary / "multi_current_thresholds" / "threshold_brackets.csv")
    threshold_lookup = {(f(row, "current_density_mA_cm2"), row["stress_case"]): row for row in thresholds}
    opening_4 = threshold_lookup[(4.0, "opening_biased")]
    filling_4 = threshold_lookup[(4.0, "filling_biased")]
    assert_close("4 mA pressure benefit midpoint", f(opening_4, "threshold_midpoint_MPa") - f(filling_4, "threshold_midpoint_MPa"), 3.275)

    sensitivity = rows(supplementary / "stability_rule_sensitivity" / "sensitivity_matrix.csv")
    agreements = [f(row, "leverage_map_agreement_fraction") for row in sensitivity]
    assert_close("Stability-rule agreement minimum", min(agreements), 0.8866666667)
    assert_close("Stability-rule agreement median", sorted(agreements)[len(agreements) // 2], 0.94)

    print("All public metric checks passed.")


if __name__ == "__main__":
    main()
