# HyperWind-Now

A physics-constrained ML + data assimilation system for low-altitude urban wind forecasting over DFW. Built as a portfolio project targeting operational drone and UAM weather applications.

---

# Motivation

Low-altitude urban wind forecasting (0–500m AGL) is the hardest unsolved problem in drone and eVTOL operations. Standard NWP models operate at 3–25 km resolution — far too coarse to capture the building wakes, terrain channeling, and microburst outflows that threaten small aircraft. This project builds a prototype end-to-end system that addresses this gap using ML nowcasting, physics constraints, real-time data assimilation, and operational NWP ingestion.

---

# System Architecture

```
HRRR Forecast (NOAA AWS S3, 3 km, hourly) + ASOS Surface Observations
                          ↓
        Module 5: HRRR Ingestion Pipeline  [real-time]
        (Herbie fetch → LCC reprojection → QC → NetCDF)
                          ↓
        Module 1: Data Pipeline
        (feature engineering, QC, normalization)
                          ↓
        Module 2: TrajGRU Wind Nowcasting
        (0–12 hour wind speed forecast)
                          ↓
        Module 3: Physics Constraint Layer
        (divergence correction + microburst detection)
                          ↓
        Module 4: EnKF Data Assimilation
        (real-time ASOS observation update)
                          ↓
        Calibrated Wind Hazard Forecast
        (CLEAR TO FLY / NO-FLY alert)
```

---

# Results

### Module 2: TrajGRU Wind Nowcasting

| Lead Time | RMSE (m/s) | Skill vs Persistence |
|-----------|------------|----------------------|
| 3h        | 0.741      | 29.5%                |
| 6h        | 1.173      | 23.9%                |
| 9h        | 1.488      | 17.8%                |
| 12h       | 1.634      | 19.0%                |

### Module 4: EnKF Assimilation (2021-04-06T21:00 UTC)

| Station | Obs (m/s) | ERA5 Background | Innovation |
|---------|-----------|-----------------|------------|
| DFW     | 11.40     | 10.23           | +1.17      |
| DAL     | 9.10      | 10.14           | -1.04      |
| FTW     | 11.38     | 10.87           | +0.51      |
| AFW     | 12.15     | 10.24           | +1.91      |
| DTO     | 11.63     | 9.88            | +1.75      |
| RBD     | 10.34     | 10.14           | +0.20      |

### Module 5: HRRR Real-Time Pipeline (2021-04-06T21:00 UTC)

| Step | Status | Detail |
|------|--------|--------|
| Fetch | ✅ | 11 variables from `s3://noaa-hrrr-bdp-pds` in ~9s |
| Reproject | ✅ | LCC → 0.25° lat/lon, wind rotation corrected |
| QC | ✅ | PASS — 0 warnings, 0 failures |
| Export | ✅ | 41.8 KB ERA5-schema NetCDF |

### Diagnostic Plots

| Module | Plot |
|--------|------|
| Module 1 | ERA5 wind fields + ASOS time series |
| Module 2 | TrajGRU training curves + forecast skill vs persistence |
| Module 3 | Physics correction + divergence maps + microburst risk |
| Module 4 | EnKF background vs analysis + innovations + ensemble spread |
| Module 5 | HRRR DFW snapshot + resolution comparison + architecture diagram |

See the `results/` folder for all diagnostic plots.

---

## Domain

| Parameter | Value |
|-----------|-------|
| Region | Dallas-Fort Worth, Texas |
| Latitude | 32.25° – 33.25° N |
| Longitude | 97.75° – 96.50° W |
| Pressure levels | 1000, 975, 950 hPa |
| ASOS stations | DFW, DAL, FTW, AFW, DTO, RBD |
| Output resolution | 0.25° lat/lon (~28 km, ERA5-compatible) |
| Domain grid | 5 × 6 cells |

---

## Data Sources

| Source | Variables | Period |
|--------|-----------|--------|
| ERA5 (CDS API) | u10, v10, msl, u/v at 1000/975/950 hPa | 2020–2021 |
| HRRR (NOAA AWS S3) | u10, v10, msl, gust, t2m, u/v at 1000/975/950 hPa | Real-time |
| ASOS (Iowa Mesonet) | wind speed, wind direction | 2020–2021 |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `module1_data_pipeline.ipynb` | ERA5 + ASOS download, QC, feature engineering |
| `module2_trajgru.ipynb` | TrajGRU model training and evaluation |
| `module3_physics.ipynb` | Divergence correction and microburst detection |
| `module4_enkf.ipynb` | Ensemble Kalman Filter data assimilation |
| `module5_hrrr_ingestion.ipynb` | HRRR real-time ingestion pipeline |

---

## Installation

```bash
conda create -n hyperwind python=3.11
conda activate hyperwind
pip install climetlab xarray numpy pandas torch requests
pip install cartopy scikit-learn tqdm scipy matplotlib netCDF4
pip install herbie-data cfgrib pyproj eccodes
```

A CDS API key is required for ERA5 access: https://cds.climate.copernicus.eu/api-how-to

HRRR data is fetched from NOAA's public AWS S3 bucket — no credentials required.

---

## Key Techniques

- **TrajGRU** — Trajectory Gated Recurrent Unit for sequential wind nowcasting
- **Percentile feature engineering** — spatial distribution compression (9 quantiles)
- **Physics constraints** — iterative divergence correction enforcing mass conservation
- **Microburst detection** — surface divergence threshold alerting (FAA-inspired)
- **Ensemble Kalman Filter** — 50-member ensemble, nearest-neighbor observation operator
- **HRRR ingestion** — Herbie byte-range S3 fetch, LCC→lat/lon reprojection, wind rotation correction
- **Temporal train/val/test split** — prevents data leakage in time series ML

---

## Relevance to Operational Drone Weather

This system directly addresses the core challenges in low-altitude UAM weather:

1. **Real-time NWP** — Module 5 replaces ERA5 reanalysis with HRRR operational forecasts (3 km, hourly, ~45 min latency)
2. **Nowcasting** — 0–12h wind forecasts at 3h resolution for flight planning
3. **Physics consistency** — mass-conserving wind fields reduce false alarms
4. **Real-time correction** — EnKF assimilates sparse surface observations to correct coarse NWP background, mimicking operational data assimilation cycles
5. **Hazard alerting** — microburst detection provides binary CLEAR/NO-FLY output suitable for drone operator decision support

---

## ERA5 vs HRRR

| | ERA5 (Modules 1–4) | HRRR (Module 5) |
|---|---|---|
| Resolution | ~28 km | ~3 km native → 0.25° output |
| Update cycle | Monthly reanalysis | Hourly real-time |
| Availability | 3–5 month delay | ~45 min latency |
| Radar assimilation | No | Yes (15-min cycle) |
| Wind gust field | No | Yes |
| Credentials | CDS API key | None |
| Operational | No | **Yes** |

---

## Author

**Emmanuel Ibekwe**
Email: ibekwemmanuel@gmail.com

---

## Acknowledgements

Built using techniques from the ECMWF Machine Learning in Weather and Climate MOOC.
ERA5 data provided by the Copernicus Climate Data Store.
ASOS data provided by the Iowa Environmental Mesonet.
HRRR data provided by NOAA and hosted on AWS S3.
