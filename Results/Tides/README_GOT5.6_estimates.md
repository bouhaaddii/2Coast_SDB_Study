# GOT5.6 tide estimates for Anza and Ras El Ma

These files contain predicted astronomical tidal elevations from the NASA GSFC
GOT5.6 model, calculated with pyTMD. Raw heights use the model harmonic
reference. The pipeline uses each acquisition's anomaly relative to the modeled
mean over that site's local survey date. This removes the unknown constant
model-reference offset; it does not establish an absolute tie to either port's
Hydrographic Zero.

## Coordinates and timing

The model point for each site is the WGS84 position obtained by transforming the
median projected X/Y coordinates of that site's bathymetric sounding dataset.
The table `modelled_tide_overpasses_GOT5.6.csv` contains the estimate at each HLS
acquisition time in UTC. Survey plans supply dates but not survey times, so
`modelled_survey_day_tide_range_GOT5.6.csv` gives a 10-minute-sampled range across
the local civil survey date in Africa/Casablanca time.

## Coastal, timing, and vertical-datum limits

GOT5 is a global 1/8-degree harmonic atlas. For these coastal points, pyTMD was
allowed to extrapolate from the nearest valid wet model node only when it was
within 10 km. This is a regional astronomical estimate, not a gauge observation;
it does not include local wind setup, atmospheric pressure residuals, wave setup,
or run-up. The survey plans provide dates but not survey times. The daily mean is
therefore an explicit reference assumption; the full-day min/max range is
provided to show the possible within-day variation. The `local_ZH_level_m` field
remains empty because no absolute tie to the port zeros is asserted. The new
`tide_anomaly_m_relative_to_survey_day_mean` field in `Config/tide_levels.csv`
contains only relative anomalies for the model's intended depth-plus-tide
adjustment.

## Model files and references

GOT5.6 combines the four additional degree-3 constituents in NASA's GOT5.6
archive with the base GOT5.5 constituents. GOT5.5 files used here are from the
pyTMD developer model-data mirror on Zenodo (record 10.5281/zenodo.22970362;
the record states that the models remain the property of their developers and
may be used for scientific purposes). The GOT5.6 supplement was fetched from
NASA GSFC. Cite the original GOT model documentation and software, not this
calculation note, in any manuscript.

- NASA GSFC, Ocean Tide Models: https://earth.gsfc.nasa.gov/geo/data/ocean-tide-models
- Ray, R. D. (2025), GOT5 technical memorandum, NASA TM-20250002085:
  https://sedvme.gsfc.nasa.gov/sed/content/uploadFiles/publication_files/GOT5-TechMemo.pdf
- pyTMD documentation: https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html
- pyTMD Zenodo model-data record: https://doi.org/10.5281/zenodo.22970362
