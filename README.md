# Reduced Li|SE Contact-Field Model

[![DOI](https://zenodo.org/badge/1280075705.svg)](https://doi.org/10.5281/zenodo.21003701)

This repository contains the public code-data archive for the manuscript:

**Feedback-amplified cathode-response leverage in Li|solid-electrolyte contact stability**

The archive is intended to make the reduced-model calculations, processed
source-workbook extraction tables, manuscript figure data, and supplementary
analysis tables inspectable in a clean public repository.

## What Is Included

- `src/li_se_reduced_model.py`: clean implementation of the reduced Li|SE
  contact-field model used for the manuscript analyses.
- `scripts/verify_key_metrics.py`: lightweight numerical check for the main
  reported values that can be recomputed directly from the included CSV files.
- `data/source_workbook_extractions/`: processed values extracted from the
  public source workbook, including CCD-boundary, pressure-waveform,
  EIS/contact-resistance, profilometry, cycling-context, and Hill-boundary
  fit-parameter tables.
- `data/manuscript_figure_data/`: source tables behind the main manuscript
  figures and external morphology/contact-loss comparison.
- `data/supplementary_data/`: supplementary analysis tables for feedback
  amplification, threshold brackets, stability-rule sensitivity,
  term-level attribution, pressure-taper sensitivity, and Z-LCO checks.
- `data/supplementary_tables/`: model metric definitions, parameter/provenance
  tables, and simulation-parameter tables.
- `figures/`: final main and supplementary figure files.
- `docs/model_equations.md`: model equations and metric definitions.
- `docs/supplementary_information_captions.md`: journal-facing supplementary
  figure captions and information.

## What Is Not Included

The raw Moradi source workbook is not redistributed here. It should be obtained
from the original publication unless redistribution permission is confirmed.
The processed extraction tables used in the manuscript are included under
`data/source_workbook_extractions/`.

This archive also excludes intermediate working reports, old manuscript
packages, deployment files, and other development-only folders.

## Quick Start

Create a Python environment and install the lightweight dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run the public metric check:

```sh
python scripts/verify_key_metrics.py
```

The script prints the key values used in the manuscript and verifies that they
match the included CSV data within small rounding tolerances.

## Scope

The model is a reduced, source-calibrated Li|SE contact-field model. It is not a
full electro-chemo-mechanical phase-field solver. The manuscript uses it to
separate CCD-boundary interpolation from dynamic stress-response/contact-field
feedback and to map pressure-current regimes under the stated assumptions.

## License

This repository uses file-type licensing:

- Code in `src/` and `scripts/` is licensed under the MIT License.
- Data, figures, and documentation are licensed under the Creative Commons
  Attribution 4.0 International License (CC BY 4.0).

The raw Moradi source workbook is not included in this repository and is not
licensed by the authors of this repository. See `LICENSE` for details.

## Citation

Please cite the versioned Zenodo archive for the release you used:

AJ. **Feedback-amplified cathode-response leverage in Li|solid-electrolyte
contact stability: code/data reproducibility archive**. Version 1.0.0. Zenodo.
https://doi.org/10.5281/zenodo.21003702

For machine-readable citation metadata, see `CITATION.cff`. When the associated
manuscript has a final citation, cite both the manuscript and this archive.
