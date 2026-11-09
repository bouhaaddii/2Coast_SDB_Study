# Tide-Aware Multi-Temporal Satellite-Derived Bathymetry Across Two Moroccan Coasts

This repository contains the code, study inputs, validation outputs, and figures for the Anza/Agadir (Atlantic) and Ras El Ma/Ras Kebdana (Mediterranean) satellite-derived bathymetry study.

The complete analysis through maps, figures, transfer experiments, and uncertainty diagnostics was run by the author on 9 November 2026. All depth values remain referenced to each site's own local hydrographic zero. No absolute datum tie between the two ports is asserted.

## Main findings from the current run

The table reports mean RMSE across five buffered spatial folds for the 30-day, all-scene, quality-weighted composite with modeled GOT5.6 anomalies applied. HLS Mixed combines harmonized Sentinel-2 and Landsat observations; S30-only uses Sentinel-2.

| Site | Scene pool | Stumpf log-ratio | Random Forest | HistGradientBoosting | RBF-SVR |
|---|---|---:|---:|---:|---:|
| Anza / Agadir | S30-only (25 scenes) | 6.603 m | 1.290 m | 1.265 m | 1.116 m |
| Anza / Agadir | HLS Mixed (33 scenes) | 5.802 m | 1.132 m | 1.103 m | 1.131 m |
| Ras El Ma / Ras Kebdana | S30-only (12 scenes) | 1.436 m | 0.624 m | 0.629 m | 0.516 m |
| Ras El Ma / Ras Kebdana | HLS Mixed (20 scenes) | 2.890 m | 0.636 m | 0.619 m | 0.472 m |

These are site-specific results under the present sampling, optical conditions, and validation design. They do not establish universal model rankings. The HLS mixture improves the HistGradientBoosting RMSE modestly at Ras El Ma and more clearly at Anza; performance differs by model and site.

A paired block-bootstrap tide ablation found no material RMSE change for the HLS Mixed HistGradientBoosting model: Anza tide-minus-no-tide ΔRMSE = −0.0004 m (95% interval −0.0091 to 0.0081 m); Ras El Ma ΔRMSE = +0.0013 m (−0.0106 to 0.0127 m). The intervals include zero. GOT5.6 is therefore treated as a tested adjustment, not a demonstrated accuracy improvement. There is no local tide-gauge validation.

## Study inputs

- `Data/Survey/` contains the supplied bathymetric point exports and survey plans. Ras El Ma point data are included as supplied by the author.
- `Data/Satellite/` contains the clipped NASA/USGS Harmonized Landsat and Sentinel-2 v2.0 image inputs used in the analysis.
- `Data/Tide_Model/GOT5.6/` contains the GOT5.6 model constituents used by `pyTMD`. The unused GOT5.5 test package is not included.
- `Data/Open_Bathymetry/` contains EMODnet and GEBCO context layers. These are not validation truth because their lineage, resolution, and vertical-datum compatibility have not been established.

The preparation step removes exact duplicate XYZ records before constructing pixel labels. Inside the analyzed image chips, it retained 23,178 Anza soundings and 372,654 Ras El Ma soundings; 10,595 exact XYZ duplicate rows were removed from the Ras El Ma chip. The resulting sample table contains 2,660 Anza and 5,837 Ras El Ma submerged 30 m pixels.

HLS v2.0 source collections and identifiers are recorded in `Config/experiment.json` and `Results/Prepared/scene_quality.csv`. The project uses blue, green, red, and near-infrared reflectance with the HLS Fmask quality layer. GOT5.6 values are acquisition anomalies relative to the modeled local survey-day mean; they do not correct absolute port datum, meteorological surge, or wave setup.

## Methods and validation

- Primary validation: five buffered spatial folds, 500 m spatial blocks, and a 90 m training-exclusion buffer.
- Primary optical composites: 30-day window, quality-weighted mean, with a 14-day sensitivity analysis.
- Models: Stumpf log-ratio, Random Forest, HistGradientBoosting, and a fixed RBF-SVR benchmark (`C=10`, `epsilon=0.1 m`, `gamma=scale`).
- Additional analyses: paired tide ablation, nested scene-count selection, cross-site transfer with target-site calibration, depth-stratified errors, and block-bootstrap intervals.
- Full-data bathymetric maps are supported only within 90 m of valid survey samples. Use held-out spatial predictions and metrics for accuracy claims; the fitted maps are visualization products.

The two hydrographic zeros are local and not tied to a common vertical datum. Cross-site absolute-depth transfer therefore requires target-site calibration and must not be interpreted as datum-free transfer. The available DWG plans and XYZ exports do not quantify the survey instrument, vertical control, water-level reduction, local datum realization, or correction uncertainties. A complete survey TVU/U95 cannot be calculated from the available records. See the diagnostics in `Results/Additional_Analyses/survey_vertical_uncertainty_budget.csv` and `survey_window_tide_timing_diagnostic.csv`.

## Reproduce the analysis

Use Windows PowerShell from the repository root. Create the environment and install the recorded dependencies:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r ".\Config\requirements.txt"
```

Run the main pipeline in this order, waiting for each command to finish successfully:

```powershell
& ".\.venv\Scripts\python.exe" ".\Scripts\00_check_inputs.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\01_prepare_samples.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\05_estimate_tides.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\02_run_validation.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\06_tide_ablation.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\10_run_svr_extension.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\11_nested_scene_selection.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\07_cross_site_transfer.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\08_uncertainty_analysis.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\03_generate_supported_maps.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\04_make_figures.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\09_make_extended_figures.py"
```

Available-data diagnostics can be run separately:

```powershell
& ".\.venv\Scripts\python.exe" ".\Scripts\12_vertical_uncertainty_assessment.py"
& ".\.venv\Scripts\python.exe" ".\Scripts\13_label_support_diagnostic.py"
```

`00_download_and_clip_hls.py` can rebuild missing HLS chips. `00_download_atl24.py` is optional; ATL24 observations are not used as validation truth in the current analysis.

## Output files

- `Results/Prepared/`: deduplicated pixel labels and acquisition-quality records.
- `Results/Metrics/`: spatial-fold scores and scene-count summaries.
- `Results/Predictions/`: out-of-fold predictions and supported full-data depth maps.
- `Results/Additional_Analyses/`: tide, transfer, nested selection, bootstrap, and data-support diagnostics.
- `Results/Tides/`: GOT5.6 acquisition and survey-window calculations.
- `Results/Figures/`: publication-oriented figure pairs in PNG and PDF.
- `Results/Logs/run_metadata.json`: run settings and tide/datum limitations.

## Data and reuse

Third-party satellite, tide-model, and open-bathymetry inputs remain subject to their original source terms. The repository does not add a separate reuse license for the author-supplied bathymetric survey data. Confirm the journal's data-license requirement before citing this repository as an openly licensed data archive.
