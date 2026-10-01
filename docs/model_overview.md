# Model overview

The manuscript uses a reduced contact-fraction and impedance-observation model, calibrated to the published source ratios. It does not resolve full electro-chemo-mechanical phase-field physics. The fit uses five shared parameters $(L,H_0,\eta,b,A)$ and a creep exponent $m$; separate P/N kinetic laws are not introduced.

Let $a\in(0,1]$ be the conducting-contact fraction, $J$ the nominal current density, $q_0$ the starting discharged charge, and $b_c(q)$ the processed assay pressure trace in MPa. Time $t$ is in hours, $J$ is in mA cm$^{-2}$, and $q$ is in mAh cm$^{-2}$. The length scale $L$ is in $\mu$m, $H_0$ is in h$^{-1}$, and the fixed conversion factor is $U=4.88778957\,\mu\mathrm{m}/(\mathrm{mAh}\,\mathrm{cm}^{-2})$. The pressure input and contact evolution are

$$
p(t)=2\,\mathrm{MPa}+\eta b_c(q_0+Jt),\qquad
\dot a=H_0A^{m+1}\left(\frac{p(t)}{2\,\mathrm{MPa}}\right)^m\frac{1-a}{a^m}-\frac{UJ}{La}.
$$

The resistance scale $R_*$ uses the same units as the source-assigned high-frequency composite resistance $R_h$. The observation law is

$$
R_h=R_b+\frac{C}{a},\qquad R_b=bR_*,\qquad C=A(1-b)R_*.
$$

The initial contact state is inferred from charged resistance under this assumed observation law. The pressure-history term is applied through a shared effective gain; its physical transfer is not independently identified by these fits. The baseline creep exponent is $m=6.6$; the finite sensitivity projections use $m=5.9$ and $m=7.3$. These are constitutive assumptions at contact scale. The model outputs protocol-specific predictions and decomposition comparisons; they are not unique physical-pathway identification or independent experimental validation.

The spatial comparison adds a finite-volume electrolyte current-routing calculation and compares resolved transport with its equipotential limit under local-loading and common-active-area load-sharing assumptions. The saved spatial output is small and sign-dependent for the single tested profile, parameter vector, load-sharing assumptions, and readouts. It does not bound other profiles or operating conditions.

See the manuscript and supplementary information for full equations, acquisition/readout definitions, parameter bounds, operational screens, and numerical limitations.
