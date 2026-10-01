# Reproduction and verification

## Saved-file verification

Run the standard-library check from this directory:

```sh
python3 scripts/verify_saved_outputs.py
```

It verifies `SHA256SUMS.txt`, checks each evidence ZIP and its member manifest, validates selected CSV row counts, recomputes challenge maximum residuals from the saved 14-row residual table, checks the forecast range summary against the saved compatible vectors and accepted challenge records, and parses the included archived Python source files with `ast`. It never imports a model module or starts an optimizer, simulation, extraction, or fit.

## Figure regeneration

Main Figure 3 and Supplementary Figure S1 can be regenerated from `data/load_sharing_8_rows.csv` using `scripts/build_spatial_figures.py`. That optional step requires Python and Matplotlib, reads only saved values, and writes outputs under `figures/regenerated/` so supplied final figures remain unchanged. It does not call the scientific solver. The remaining manuscript figures are included as final PDF/PNG outputs; their full figure-production chain is not represented by this script.

## Full analysis environment

`source/route_a/` preserves source snapshots, and the `evidence/` bundles preserve selected saved cases, fits, and checks. The full Route A execution depended on the original shared-compute launcher, worktree-level modules and paths, and study input layout. Those runtime components are not all part of this curated repository, so this repository does not promise a clean-room end-to-end solver rerun. The 0.99–5.62 pp baseline-exponent forecast range is a finite sample of retained compatible records, not a certified range over all possible parameters or models. The failed challenge return remains in the evidence bundle and is not converted to a forecast.

The raw third-party source workbook is excluded; processed values are provided in `data/`, with citation details in `DATA_SOURCES.md`.
