# Reduced Interface Model Equations

The model is a reduced finite-difference phase-field-inspired interface model, not a full thermodynamic electro-chemo-mechanical phase-field framework.

## State Variables

- `phi(x,y,t)` is the reduced contact field: `phi=1` means filled/contacting interface, `phi=0` means void/open interface.
- `psi(x,y,t)` is a reduced stress-bias field.
- `s` is the nondimensional cathode stress index: `-1` for P-LCO, `0` for neutral/Z-LCO-like behavior, `+1` for N-LCO.
- `p` is stack pressure in MPa and `j` is current density in mA cm-2.

## Source-Calibrated Current Boundary

```text
j_c(p) = j_floor + (j_max - j_floor) / (1 + (p_t / p)^n)
j_crit(p,s) = (1 - w_s) j_c,P(p) + w_s j_c,N(p)
w_s = clip((s + 1) / 2, 0, 1)
```

The CCD data define the source-calibrated current scale. The claimed result is not CCD prediction; it is the void/contact morphology and pressure-current-stress leverage map built on that calibration.

## Local Current Focusing

```text
j_loc(x) = j [1 / max(phi(x,0), phi_min)] / mean_x[1 / max(phi(x,0), phi_min)]
```

This represents contact-loss-driven current focusing. The ablation robustness evaluation shows it is a morphology/local-hotspot term, not the primary source of P-LCO/N-LCO scalar separation.

## Reduced Stress-Bias Evolution

```text
dpsi/dt = lambda_psi [S(x,y,t) - psi] + D_psi laplacian(psi)
S = (a_s s + a_p p) T(x,y) - a_v (1 - phi) - a_j H(j_loc / j_crit - 1) W_i(y)
```

`T(x,y)` is fixed heterogeneity, `W_i(y)` localizes interface effects, and `H(z)=max(z,0)`.

Physical signs:

- Favorable N-LCO-like response (`s>0`) biases closure/filling.
- Unfavorable P-LCO-like response (`s<0`) biases opening.
- Pressure contributes closure.
- Existing voids reduce support.
- Local current overdrive increases opening tendency.

## Interface Evolution

```text
dphi/dt = D_phi laplacian(phi)
        + beta phi(1 - phi)(phi - 1/2)
        + W_i(y) [k_c C(1 - phi) - k_o O phi]

C = b_p p W_b(y) + b_+ H(psi) + b_sub H(1 - j / j_crit)
O = c_- H(-psi)
  + c_j H(j_loc / j_crit - 1) [1 + c_v(1 - phi)]
  + c_over H(j / j_crit - 1)
```

Positive stress bias, pressure, and subcritical current close voids. Negative stress bias, local overdrive, and global overdrive open voids. Existing voids amplify local overdrive because current crowds around remaining contact.

## Coefficient Provenance

| Term group | Coefficients used in code | Role | Provenance |
| --- | --- | --- | --- |
| CCD boundary | fitted `jmax`, `p_t`, `n`, `floor` | source-calibrated current scale | fitted to source Fig. 3d CCD rows |
| Stress source | `0.74 s`, `0.11 p`, `-0.24 void`, `-0.10 overdrive` | cathode/pressure/void/current stress bias | reduced model choices; signs fixed by physical role |
| Stress relaxation | `0.22`, `0.18` | relaxation and smoothing of `psi` | numerical reduced-model choices |
| Closure drive | `0.10 p`, `0.82 H(psi)`, `0.24 subcritical` | pressure/favorable-stress closure | reduced model choices |
| Opening drive | `0.78 H(-psi)`, `0.74 local overdrive`, `0.85 void amplification`, `0.16 global overdrive` | stress/current-driven opening | reduced model choices |
| Field regularization | `0.070 laplacian(phi)`, `0.014 double-well term` | smooth interface and two-state behavior | reduced phase-field-inspired choices |
| Reaction scaling | `0.135 closure`, `0.118 opening` | maps drives to `phi` evolution | reduced model choices |

The empirical coefficient risk is handled by sensitivity testing. Stress gain, opening gain, closure gain, current-focusing gain, and `j_crit` scale were perturbed across 48 samples. Opening/filling ordering survived in 100% of samples, with a 5th percentile risk separation of 0.361.

## Scope

The model supports a reduced mechanism/leverage-map claim. It does not support claiming a universal CCD predictor, a directly measured stress field, or a full thermodynamic multiphysics phase-field model.
