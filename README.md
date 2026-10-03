# Cathode-response leverage in Li|solid-electrolyte contact stability: modeling and falsifiable impedance tests

**Current manuscript:** *Cathode-response leverage in Li|solid-electrolyte contact stability: modeling and falsifiable impedance tests*

**Archive version:** 1.1.0 (2026-10-03), [doi:10.5281/zenodo.23120547](https://doi.org/10.5281/zenodo.23120547).

This repository contains the processed tables, final figures, saved numerical records, and selected source snapshots associated with the current manuscript. The current analysis evaluates a reduced contact/impedance model against one published LCO-cell study and makes protocol-specific, assumption-conditional predictions. It is not a full electro-chemo-mechanical phase-field solver or an independently validated cell model.

## Main results represented in these files

- N-LCO's uncapped source critical-current densities correspond to +0.77 and +2.74 MPa of extra pressure on the P-LCO fit. This is 34–144 times the maximum paired pressure difference in the Li-free assay. The comparison quantifies the leverage but compares different experimental protocols.
- In the tested spatial case, current redistribution changes the P/N contrast by less than 0.07 percentage points; its sign depends on load sharing and readout.
- Under the specified Test 1 protocol, retained baseline-exponent predictions span a P-minus-N contrast of 0.99–5.62 percentage points. Endpoint-exponent predictions span 2.69–4.17 percentage points. These are finite sampled ranges, not confidence intervals or certified bounds. The sampled lower edge depends on the 0.03 compatibility tolerance.
- Crossing pressure histories with starting states shows a trade-off between in-window history and inherited contact state. The factorial decomposition covers 53 baseline-exponent records; it does not cover the additional minimum-search records or the endpoint-exponent projections.

For cohort definitions, row descriptions, and checksums, see [`data/README.md`](data/README.md). The numerical-source map is in [`provenance/NUMERICAL_SOURCES.csv`](provenance/NUMERICAL_SOURCES.csv).

## Contents

- `data/`: 13 processed CSV/JSON tables supplied with the current manuscript.
- `figures/`: six main figures and one supplementary figure, each as PDF and PNG.
- `evidence/`: compact ZIP bundles of the exact archived fit, projection, optimizer, spatial, and spectral-readout records used to check the tables. Each bundle contains an `EVIDENCE_MANIFEST.json` with SHA-256 hashes and the frozen archive hash. Two operational manifests omit only machine-local job receipt/log path keys; their original and exported hashes are both recorded.
- `source/route_a/`: selected exact Python source snapshots from the frozen analysis archive, with hashes in `source/SOURCE_MANIFEST.csv`.
- `scripts/verify_saved_outputs.py`: standard-library-only integrity and saved-table check. It does not run or import the scientific model or solver.
- `scripts/build_spatial_figures.py`: saved-table figure builder for main Figure 3 and supplementary Figure S1; it reads from `data/` and writes optional regenerations under ignored `figures/regenerated/`, leaving the frozen final figures untouched.
- `docs/`: concise model and reproduction notes.
- `provenance/`: export/source metadata and the numerical claim-to-source map.

## Verify the saved files

With Python 3, run from the repository root:

```sh
python3 scripts/verify_saved_outputs.py
```

The verifier checks repository file hashes, evidence-archive member hashes, CSV row counts, forecast ranges, challenge residual screens, and Python syntax without importing archived scripts. No third-party Python packages are needed for this check. Rebuilding the two spatial figures requires Matplotlib; the other final figures are supplied as outputs.

## Reproduction scope

The source snapshots preserve exact calculation and audit scripts, and the evidence ZIPs retain the saved records that support the published tables. Most archived run scripts depend on the AJ Physics worktree layout, shared-compute launcher, and runtime modules that are not included here; they are supplied for source inspection and provenance, not as a turnkey solver package. This repository's default verification is deliberately limited to saved-file integrity and arithmetic. It does not re-run fitting, optimization, extraction, or simulation.

The full frozen Route A output archive is separately available at [numerical-evidence.zip](https://aj-physics-qq6btmr96-iscoot.vercel.app/review/cathode-submission-20260927-d64c45f08d114e2b/numerical-evidence.zip). This curated repository contains only the selected source and output records listed in its manifests.

## Source data

Processed measurements derive from Moradi, Zahiri, and Braun, *Nature Communications* 16, 9266 (2025), [doi:10.1038/s41467-025-64358-2](https://doi.org/10.1038/s41467-025-64358-2). The third-party source workbook is not redistributed. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for attribution and scope.

## Version history

Version 1.1.0 archives the revised manuscript's source snapshots, processed data and saved results at [10.5281/zenodo.23120547](https://doi.org/10.5281/zenodo.23120547). Its scientific files are unchanged from repository commit `071f52cd55932346dc7647e8fd4e09b094679521`; this update adds the archive citation metadata and refreshed checksums.

The annotated `v1.0.0` tag is retained as historical provenance. Its Zenodo DOI, [10.5281/zenodo.21003702](https://doi.org/10.5281/zenodo.21003702), identifies the older code archive. The historical `v1.0.0` tag object is `d94008a8d2e50062266484e60e886ab9e7895198`; it peels to commit `91fc8e118b4e02ad6a8676379224f547bfc58267`. The public main commit before the revised source/data update was `8d7741057618a0dae88ab7177eeb8ff6fda4ef00`.

## License

Code, data, figures, and documentation use the file-type licenses described in [`LICENSE`](LICENSE). The raw source workbook and other third-party materials remain excluded and are not relicensed here.
