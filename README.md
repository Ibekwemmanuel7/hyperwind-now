\# HyperWind-Now
\*Physics-Constrained ML + Data Assimilation for Low-Altitude Urban Wind Forecasting over DFW\*
Built as a portfolio project targeting operational drone and UAM weather applications,
inspired by operational low-altitude weather intelligence pipelines used in drone and UAM decision-support systems.
---

\## Motivation
---
Low-altitude urban wind forecasting (0–500m AGL) is the hardest unsolved problem
in drone and eVTOL operations. Standard NWP models operate at 3–25km resolution —
far too coarse to capture the building wakes, terrain channeling, and microburst
outflows that threaten small aircraft. This project builds a prototype end-to-end
system that addresses this gap using ML nowcasting, physics constraints, and
real-time data assimilation.

---
\## System Architecture
---
ERA5 Reanalysis + ASOS Surface Observations

&nbsp;             ↓

&nbsp;  Module 1: Data Pipeline

&nbsp;  (feature engineering, QC, normalization)

&nbsp;             ↓

&nbsp;  Module 2: TrajGRU Wind Nowcasting

&nbsp;  (0–12 hour wind speed forecast)

&nbsp;             ↓

&nbsp;  Module 3: Physics Constraint Layer

&nbsp;  (divergence correction + microburst detection)

&nbsp;             ↓

&nbsp;  Module 4: EnKF Data Assimilation

&nbsp;  (real-time ASOS observation update)

&nbsp;             ↓

&nbsp;  Calibrated Wind Hazard Forecast

&nbsp;  (CLEAR TO FLY / NO-FLY alert)

---
\## Results
\### Module 2: TrajGRU Wind Nowcasting
| Lead Time | RMSE (m/s) | Skill vs Persistence |

|-----------|------------|----------------------|
| 3h        | 0.741      | 29.5%                |
| 6h        | 1.173      | 23.9%                |
| 9h        | 1.488      | 17.8%                |
| 12h       | 1.634      | 19.0%                |

\### Module 4: EnKF Assimilation (2021-04-06T21:00 UTC)
| Station | Obs (m/s) | ERA5 Background | Innovation |
|---------|-----------|-----------------|------------|
| DFW     | 11.40     | 10.23           | +1.17      |
| DAL     | 9.10      | 10.14           | -1.04      |
| FTW     | 11.38     | 10.87           | +0.51      |
| AFW     | 12.15     | 10.24           | +1.91      |
| DTO     | 11.63     | 9.88            | +1.75      |
| RBD     | 10.34     | 10.14           | +0.20      |
---

\## Results Plots
| Module | Plot |
|--------|------|
| Module 1 | ERA5 wind fields + ASOS time series |
| Module 2 | TrajGRU training curves + forecast skill vs persistence |
| Module 3 | Physics correction + divergence maps + microburst risk |
| Module 4 | EnKF background vs analysis + innovations + ensemble spread |
See the `results/` folder for all diagnostic plots.

---
\## Domain
\*\*Dallas-Fort Worth, Texas\*\*
\- Latitude: 32.25° – 33.25° N
\- Longitude: 97.75° – 96.50° W
\- Pressure levels: 1000, 975, 950 hPa
\- ASOS stations: DFW, DAL, FTW, AFW, DTO, RBD
\- Spatial resolution: 0.25° ERA5 (~28 km)
\- Domain grid: 5 × 6 cells

---

\## Data Sources
| Source | Variables | Period |
|--------|-----------|--------|
| ERA5 (CDS API) | u10, v10, msl, u/v at 1000/975/950 hPa | 2020–2021 |
| ASOS (Iowa Mesonet) | wind speed, wind direction | 2020–2021 |
---

\## Notebooks
| Notebook | Description |
|----------|-------------|
| `module1\_data\_pipeline.ipynb` | ERA5 + ASOS download, QC, feature engineering |
| `module2\_trajgru.ipynb` | TrajGRU model training and evaluation |
| `module3\_physics.ipynb` | Divergence correction and microburst detection |
| `module4\_enkf.ipynb` | Ensemble Kalman Filter data assimilation |
---

\## Installation
```bash
conda create -n hyperwind python=3.11
conda activate hyperwind
pip install climetlab xarray numpy pandas torch requests
pip install cartopy scikit-learn tqdm scipy matplotlib netCDF4
```
You will also need a CDS API key for ERA5 access:
https://cds.climate.copernicus.eu/api-how-to
---
\## Key Techniques
\- \*\*TrajGRU\*\* — Trajectory Gated Recurrent Unit for sequential wind nowcasting
\- \*\*Percentile feature engineering\*\* — spatial distribution compression (9 quantiles)
\- \*\*Physics constraints\*\* — iterative divergence correction enforcing mass conservation
\- \*\*Microburst detection\*\* — surface divergence threshold alerting (FAA-inspired)
\- \*\*Ensemble Kalman Filter\*\* — 50-member ensemble, nearest-neighbor observation operator
\- \*\*Temporal train/val/test split\*\* — prevents data leakage in time series ML

---
\## Relevance to Operational Drone Weather
This system directly addresses the core challenges in low-altitude UAM weather:
1\. \*\*Nowcasting\*\* — 0–12h wind forecasts at 3h resolution for flight planning
2\. \*\*Physics consistency\*\* — mass-conserving wind fields reduce false alarms
3\. \*\*Real-time correction\*\* — EnKF assimilates sparse surface observations to correct
&nbsp;  coarse NWP background, mimicking operational data assimilation cycles
4\. \*\*Hazard alerting\*\* — microburst detection provides binary CLEAR/NO-FLY output

&nbsp;  suitable for drone operator decision support
---
\## Author
\*\*Emmanuel Ibekwe\*\*
\- Email: ibekwemmanuel@gmail.com
---
## Limitations

This prototype uses ERA5 reanalysis rather than operational NWP
and therefore does not yet represent real-time forecasting skill.

Future work will integrate HRRR or ECMWF IFS forecasts.

\## Acknowledgements
Built using techniques from the ECMWF Machine Learning in Weather and Climate MOOC.
ERA5 data provided by the Copernicus Climate Data Store.
ASOS data provided by the Iowa Environmental Mesonet.
