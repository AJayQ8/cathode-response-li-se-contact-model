# Processed tables

These 13 CSV/JSON files are byte-for-byte copies of the saved tables used by the current manuscript. `SHA256SUMS.txt` at the repository root records their hashes. Generated LaTeX table fragments and the paper-building environment are not included here.

## File guide

| File | Contents |
|---|---|
| `calibration_938_rows.csv` | The 938 model-versus-source ratio comparisons for the 67-vector calibration ensemble. Each row identifies the extraction, cathode/rate/stage, target and prediction, and source fit record. |
| `compatible_vectors.csv` | The 67 saved compatible calibration vectors: 50 `original50` settings/parameter vectors, 14 `endpoint_m` projections, and three additional `post50` baseline-exponent records. `parameter_source_path` and `case_or_projection_path` map into `evidence/compatible-vector-inputs.zip`; `parameter_source_path` and calibration `fit_source` paths map into the evidence bundles using the recorded project paths. |
| `forecast_ranges.csv` | Forecast counts and extrema grouped by exponent and extraction. At $m=6.6$, 53 original baseline records plus three accepted challenge outputs produce 56 records. The 14 endpoint-exponent records are separate. |
| `challenge_forecasts.csv` | The four bounded-search outputs. Three pass the unchanged all-14 screen and have forecast readouts; the published-extraction return fails the screen and has no forecast. Search outcomes are preserved in `evidence/minimum-search.zip`. |
| `challenge_all14_residuals.csv` | The 14 target residuals for each of the four challenge outputs. |
| `factor_arithmetic.json` | Saved fixed-contact pressure-factor arithmetic at the three tested creep exponents and selected transfer gains. This diagnostic is not the full dynamic forecast. |
| `ccd_arithmetic.json` | Saved Hill-fit inverse-pressure comparisons for the uncapped source observations and the distinct capacity-capped observations. |
| `pressure_arithmetic.json` | Saved pressure-window and paired P/N pressure-difference summaries for the two charge registrations. |
| `source_pressure_traces.csv` | Processed source pressure trace samples used to interpolate pressure over discharged charge. |
| `known_protocols.csv` | Protocol-specific source quantities used to construct the retained forecast scenarios. |
| `frequency_schedule.csv` | The ordered 68-frequency acquisition schedule and its saved timing fields. |
| `forecast_circuit_templates.json` | Saved circuit-template and algebraic reconstruction checks used for the forecast readout. |
| `load_sharing_8_rows.csv` | The eight saved spatial stop/rest outputs (two load-sharing assumptions, two charge points, and two readouts); these support main Figure 3. |

## Cohorts and limits

`original50`, `post50`, and `endpoint_m` are archive-group labels preserved to make the source records traceable. The 50 `original50` plus three `post50` records form the 53-record baseline-exponent factorial subset; the 14 `endpoint_m` records are the projections at $m=5.9$ or $m=7.3$. The saved deduplication groups the 67 compatibility records into 50 clusters. The current forecast table adds three further minimum-search outputs to the 53 baseline records, giving 56 baseline-exponent forecasts and 14 endpoint-exponent forecasts (70 total). The three new challenge outputs are not part of the 53-record factorial decomposition.

All fit compatibility refers to the fixed 0.03 absolute-ratio screen on the 14 source ratios. The displayed ranges are finite saved samples. The minimum-search challenge is not a global optimization certificate, and the close-to-screen accepted results mean the lower sampled edge depends on the chosen tolerance. The failed published-extraction return is kept visible and is not used to create a forecast.

## Saved evidence

- `evidence/calibration-fit-records.zip` contains the source fit records named by `calibration_938_rows.csv` and the saved final arithmetic/read-off table.
- `evidence/compatible-vector-inputs.zip` contains the parameter and case/projection source records named by `compatible_vectors.csv` and their stage manifests.
- `evidence/minimum-search.zip` contains all three search histories, optimizer returns, fit records, and the failed published return.
- `evidence/spatial-current.zip` contains saved load-sharing current-routing outputs.
- `evidence/spectral-readout-01.zip` and `evidence/spectral-readout-02.zip` contain the 53 per-vector spectral readout records, split for download size.

Each ZIP has a member-level SHA-256 index. Exact members retain their frozen-archive SHA-256; two operational manifests have machine-local job receipt/log path keys removed, with both original and exported hashes recorded. The frozen archive is identified in `provenance/EXPORT_MANIFEST.json`.
