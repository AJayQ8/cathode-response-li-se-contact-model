"""Reduced Li|SE contact-field model used for manuscript calculations.

The routines in this module are a public implementation of the reduced model
equations used in the manuscript. They keep the same numerical constants and
definitions used for the published analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HillFit:
    """Hill-boundary fit for one cathode end member."""

    cathode: str
    jmax_ma_cm2: float
    transition_pressure_mpa: float
    exponent: float
    floor_ma_cm2: float


DEFAULT_PARAMS = {
    "cathode_stress_source_gain": 1.0,
    "pressure_stress_source_gain": 1.0,
    "void_stress_weakening_gain": 1.0,
    "stress_current_overdrive_gain": 1.0,
    "pressure_closure_gain": 1.0,
    "favorable_stress_closure_gain": 1.0,
    "subcritical_closure_gain": 1.0,
    "unfavorable_stress_opening_gain": 1.0,
    "local_overdrive_opening_gain": 1.0,
    "void_amplification_gain": 1.0,
    "global_overdrive_opening_gain": 1.0,
    "current_focusing_gain": 1.0,
    "boundary_stress_index_gain": 1.0,
    "jcrit_scale": 1.0,
    "curvature_gain": 1.0,
    "phase_relax_gain": 1.0,
    "stress_relaxation_gain": 1.0,
    "stress_diffusion_gain": 1.0,
}


def hill_boundary(fit: HillFit, pressure_mpa: np.ndarray) -> np.ndarray:
    """Evaluate one fitted CCD boundary."""

    pressure = np.asarray(pressure_mpa, dtype=float)
    return fit.floor_ma_cm2 + (fit.jmax_ma_cm2 - fit.floor_ma_cm2) / (
        1.0 + (fit.transition_pressure_mpa / pressure) ** fit.exponent
    )


def boundary_from_fits(fits: Iterable[HillFit], pressure_mpa: np.ndarray, stress_index: float) -> np.ndarray:
    """Interpolate the critical-current boundary across cathode stress index.

    `stress_index = -1` corresponds to the P-LCO-like opening-biased endpoint.
    `stress_index = +1` corresponds to the N-LCO-like filling-biased endpoint.
    """

    fit_by_name = {fit.cathode: fit for fit in fits}
    opening = hill_boundary(fit_by_name["P-LCO"], pressure_mpa)
    filling = hill_boundary(fit_by_name["N-LCO"], pressure_mpa)
    weight = np.clip((stress_index + 1.0) / 2.0, 0.0, 1.0)
    return (1.0 - weight) * opening + weight * filling


def laplace_neumann(values: np.ndarray) -> np.ndarray:
    """Five-point Laplacian with edge-padded Neumann-like boundaries."""

    padded = np.pad(values, 1, mode="edge")
    return padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * values


def smooth_noise(ny: int, nx: int, seed: int, passes: int = 5) -> np.ndarray:
    """Generate smooth deterministic roughness/noise for the contact field."""

    rng = np.random.default_rng(seed)
    values = rng.normal(size=(ny, nx))
    for _ in range(passes):
        values = (
            values
            + np.roll(values, 1, axis=0)
            + np.roll(values, -1, axis=0)
            + np.roll(values, 1, axis=1)
            + np.roll(values, -1, axis=1)
        ) / 5.0
    values -= float(values.min())
    span = float(values.max() - values.min())
    return values / span if span else values


def initial_contact_field(nx: int, ny: int, seed: int, roughness_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Create the initial Li|SE contact field and vertical coordinate."""

    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)[:, None]
    roughness = smooth_noise(1, nx, seed + 37, passes=8)[0]
    pit_depth = roughness_scale * (0.038 + 0.018 * roughness + 0.008 * rng.random(nx))
    for center, width, depth in [(0.20, 0.045, 0.085), (0.49, 0.065, 0.065), (0.78, 0.043, 0.078)]:
        pit_depth += roughness_scale * depth * np.exp(-((x - center) ** 2) / (2.0 * width**2))
    phi = 1.0 / (1.0 + np.exp(-(y - pit_depth[None, :]) / 0.014))
    return phi, y


def field_metrics(
    phi: np.ndarray,
    current_ma_cm2: float,
    jcrit_ma_cm2: float,
    current_profile: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute reduced contact-field metrics."""

    contact = np.clip(phi[0], 0.025, 1.0)
    profile = current_profile if current_profile is not None else current_ma_cm2 / contact / np.mean(1.0 / contact)
    return {
        "void_fraction": float(1.0 - np.mean(phi[: max(5, phi.shape[0] // 7)])),
        "contact_fraction": float(np.mean(phi[0] > 0.55)),
        "hotspot_factor": float(np.percentile(profile, 90) / max(current_ma_cm2, 0.1)),
        "mean_current_ratio": float(current_ma_cm2 / max(jcrit_ma_cm2, 0.1)),
        "p90_local_current_ratio": float(np.percentile(profile / max(jcrit_ma_cm2, 0.1), 90)),
        "roughness_energy": float(np.mean(np.gradient(phi, axis=1) ** 2 + np.gradient(phi, axis=0) ** 2)),
    }


def field_risk(row: dict[str, float]) -> float:
    """Manuscript field-risk metric."""

    return float(row["void_fraction"]) + 0.20 * (1.0 - float(row["contact_fraction"]))


def _merged_params(overrides: dict[str, float] | None = None) -> dict[str, float]:
    params = dict(DEFAULT_PARAMS)
    if overrides:
        params.update({key: float(value) for key, value in overrides.items()})
    return params


def simulate_contact_field(
    current_ma_cm2: float,
    pressure_mpa: float,
    stress_index: float,
    fits: Iterable[HillFit],
    seed: int,
    params: dict[str, float] | None = None,
    *,
    nx: int = 56,
    ny: int = 24,
    steps: int = 105,
    dt: float = 0.058,
) -> dict[str, float]:
    """Run the reduced contact-field model and return final metrics."""

    p = _merged_params(params)
    phi, y = initial_contact_field(nx, ny, seed, roughness_scale=1.0)
    texture = 0.82 + 0.36 * smooth_noise(ny, nx, seed + 101, passes=4)
    x = np.linspace(0.0, 1.0, nx)[None, :]
    wave = 0.85 + 0.15 * np.cos(2.0 * np.pi * x + 0.40 * np.sin(4.0 * np.pi * x))
    interface_weight = np.exp(-y / 0.17)
    bulk_weight = np.exp(-y / 0.42)
    stress_field = np.zeros_like(phi)
    boundary_stress_index = p["boundary_stress_index_gain"] * stress_index
    jcrit = float(boundary_from_fits(fits, np.array([pressure_mpa]), boundary_stress_index)[0]) * p["jcrit_scale"]
    current_profile = np.full(nx, current_ma_cm2, dtype=float)

    for _ in range(steps):
        contact = np.clip(phi[0], 0.025, 1.0)
        if p["current_focusing_gain"] <= 1.0e-12:
            current_profile = np.full(nx, current_ma_cm2, dtype=float)
        else:
            inverse_contact = np.power(1.0 / contact, p["current_focusing_gain"])
            current_profile = current_ma_cm2 * inverse_contact / np.mean(inverse_contact)

        current_ratio_surface = current_profile / max(jcrit, 0.1)
        local_overdrive = np.clip(current_ratio_surface[None, :] - 1.0, 0.0, None)
        void = 1.0 - phi
        stress_source = (
            p["cathode_stress_source_gain"] * 0.74 * stress_index * texture * wave
            + p["pressure_stress_source_gain"] * 0.11 * pressure_mpa * texture * wave
            - p["void_stress_weakening_gain"] * 0.24 * void
            - p["stress_current_overdrive_gain"] * 0.10 * local_overdrive * interface_weight
        )
        stress_field += dt * (
            p["stress_relaxation_gain"] * 0.22 * (stress_source - stress_field)
            + p["stress_diffusion_gain"] * 0.18 * laplace_neumann(stress_field)
        )
        stress_field = np.clip(stress_field, -1.55, 1.55)

        closure_drive = (
            p["pressure_closure_gain"] * 0.10 * pressure_mpa * bulk_weight
            + p["favorable_stress_closure_gain"] * 0.82 * np.clip(stress_field, 0.0, None)
            + p["subcritical_closure_gain"] * 0.24 * max(0.0, 1.0 - current_ma_cm2 / max(jcrit, 0.1))
        )
        opening_drive = (
            p["unfavorable_stress_opening_gain"] * 0.78 * np.clip(-stress_field, 0.0, None)
            + p["local_overdrive_opening_gain"]
            * 0.74
            * local_overdrive
            * (1.0 + p["void_amplification_gain"] * 0.85 * void)
            + p["global_overdrive_opening_gain"] * 0.16 * max(0.0, current_ma_cm2 / max(jcrit, 0.1) - 1.0)
        )
        curvature = p["curvature_gain"] * 0.070 * laplace_neumann(phi)
        phase_relax = p["phase_relax_gain"] * 0.014 * phi * (1.0 - phi) * (phi - 0.5)
        reaction = interface_weight * (0.135 * closure_drive * (1.0 - phi) - 0.118 * opening_drive * phi)
        phi = np.clip(phi + dt * (curvature + phase_relax + reaction), 0.0, 1.0)
        phi[-1, :] = np.maximum(phi[-1, :], 0.985)

    metrics = field_metrics(phi, current_ma_cm2, jcrit, current_profile)
    return {
        "current_ma_cm2": float(current_ma_cm2),
        "pressure_mpa": float(pressure_mpa),
        "stress_index": float(stress_index),
        "jcrit_ma_cm2": float(jcrit),
        "field_risk": field_risk(metrics),
        **metrics,
    }


def run_stress_triad(fits: Iterable[HillFit], seed: int = 9101) -> dict[str, float | bool]:
    """Run opening, neutral, and filling end-member cases."""

    cases = {"opening": -1.0, "neutral": 0.0, "filling": 1.0}
    results = {
        name: simulate_contact_field(4.0, 1.0, stress, fits, seed + offset * 17)
        for offset, (name, stress) in enumerate(cases.items())
    }
    risks = {name: float(row["field_risk"]) for name, row in results.items()}
    return {
        "opening_risk": risks["opening"],
        "neutral_risk": risks["neutral"],
        "filling_risk": risks["filling"],
        "opening_filling_separation": risks["opening"] - risks["filling"],
        "opening_neutral_margin": risks["opening"] - risks["neutral"],
        "neutral_filling_margin": risks["neutral"] - risks["filling"],
        "triad_ordering_pass": risks["opening"] > risks["neutral"] > risks["filling"],
    }
