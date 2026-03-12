"""
run_live.py — HyperWind-Now Operational Demo
============================================
Fetches the latest HRRR cycle, runs EnKF assimilation with live ASOS
observations, applies microburst detection, and prints a CLEAR TO FLY
or NO-FLY decision for the DFW domain.

Usage:
    python run_live.py                  # latest available HRRR cycle
    python run_live.py --cycle "2021-04-06 21:00"  # specific cycle
    python run_live.py --plot           # show diagnostic plot
"""

import argparse
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'
RAW_DIR  = DATA_DIR / 'hrrr_raw'
PROC_DIR = DATA_DIR / 'processed'
for d in [RAW_DIR, PROC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Domain (matches all modules) ──────────────────────────────────────────────
LAT = np.array([33.25, 33.0, 32.75, 32.5, 32.25])   # descending, ERA5 convention
LON = np.array([-97.75, -97.5, -97.25, -97.0, -96.75, -96.5])
DX  = 111000 * np.cos(np.radians(32.8)) * 0.25       # metres per grid cell
DY  = 111000 * 0.25

STATIONS = {
    'DFW': (32.90, -97.04),
    'DAL': (32.85, -96.85),
    'FTW': (32.82, -97.36),
    'AFW': (32.99, -97.32),
    'DTO': (33.20, -97.18),
    'RBD': (32.68, -96.87),
}

# ── EnKF config (matches Module 4) ───────────────────────────────────────────
N_ENSEMBLE = 50
OBS_ERROR  = 0.5    # m/s
BG_ERROR   = 1.5    # m/s

# ── ASOS config ───────────────────────────────────────────────────────────────
ASOS_STATION_IDS = {
    'DFW': 'DFW',
    'DAL': 'DAL',
    'FTW': 'FTW',
    'AFW': 'AFW',
    'DTO': 'DTO',
    'RBD': 'RBD',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: HRRR Fetch (Module 5)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_hrrr(cycle_dt: datetime) -> Path:
    """
    Fetch HRRR background for cycle_dt using Module 5 pipeline.
    Returns path to processed NetCDF.
    """
    # Check cache first
    fname    = f'hrrr_dfw_{cycle_dt:%Y%m%d_%H%M}Z_f00.nc'
    out_path = PROC_DIR / fname
    if out_path.exists():
        print(f'  [cache] {fname}')
        return out_path

    print(f'  Fetching HRRR {cycle_dt:%Y-%m-%d %H:%M} UTC from AWS S3...')
    try:
        from herbie import Herbie
        from scipy.interpolate import RegularGridInterpolator

        # Search strings (Module 5)
        SFC_SEARCH = '|'.join([
            ':UGRD:10 m above ground:',
            ':VGRD:10 m above ground:',
            ':MSLMA:',
            ':GUST:surface:',
            ':TMP:2 m above ground:',
        ])
        PRS_SEARCH = '|'.join([
            ':UGRD:1000 mb:', ':VGRD:1000 mb:',
            ':UGRD:975 mb:',  ':VGRD:975 mb:',
            ':UGRD:950 mb:',  ':VGRD:950 mb:',
        ])

        def herbie_fetch(product, search):
            H  = Herbie(cycle_dt.strftime('%Y-%m-%d %H:%M'),
                        model='hrrr', product=product, fxx=0,
                        save_dir=str(RAW_DIR), verbose=False)
            ds = H.xarray(search, remove_grib=True)
            return xr.merge(ds, compat='override') if isinstance(ds, list) else ds

        sfc = herbie_fetch('sfc', SFC_SEARCH)
        prs = herbie_fetch('prs', PRS_SEARCH)

        # Rename sfc variables
        rename_sfc = {}
        for v in list(sfc.data_vars):
            s = str(sfc[v].attrs.get('GRIB_shortName', '')).lower()
            l = sfc[v].attrs.get('GRIB_level', None)
            if s == '10u' or (s in ('u','ugrd') and l == 10): rename_sfc[v] = 'u10'
            elif s == '10v' or (s in ('v','vgrd') and l == 10): rename_sfc[v] = 'v10'
            elif s == '2t' or (s == 'tmp' and l == 2): rename_sfc[v] = 't2m'
            elif 'gust' in s: rename_sfc[v] = 'gust'
            elif s in ('msl','prmsl','mslma') or 'mslma' in v.lower(): rename_sfc[v] = 'msl'
        sfc = sfc.rename({k: nv for k, nv in rename_sfc.items() if k in sfc.data_vars})

        # Unstack prs levels
        prs_out = {}
        for v in list(prs.data_vars):
            arr   = prs[v]
            short = str(arr.attrs.get('GRIB_shortName', '')).lower()
            is_u  = short in ('u','ugrd')
            is_v  = short in ('v','vgrd')
            for dim in arr.dims:
                if 'isobar' in dim.lower() or dim in ('level','isobaricInhPa'):
                    for i, lev in enumerate(arr.coords[dim].values):
                        li = int(lev)
                        if li in [1000, 975, 950]:
                            prefix = 'u' if is_u else 'v'
                            sliced = arr.isel({dim: i}).drop_vars(
                                [c for c in arr.isel({dim:i}).coords if dim in c],
                                errors='ignore')
                            prs_out[f'{prefix}{li}'] = sliced
        prs = xr.Dataset(prs_out, attrs=prs.attrs)

        merged = xr.merge([sfc, prs], compat='override')

        # Reproject to ERA5 0.25-deg grid
        lat_coord = next(c for c in merged.coords if 'lat' in c.lower())
        lon_coord = next(c for c in merged.coords if 'lon' in c.lower())
        h_lat = merged.coords[lat_coord].values
        h_lon = merged.coords[lon_coord].values
        h_lon = np.where(h_lon > 180, h_lon - 360, h_lon)

        OUT_LON, OUT_LAT = np.meshgrid(LON, LAT[::-1])  # ascending for interp

        buf = 0.5
        mask = ((h_lat >= LAT.min()-buf) & (h_lat <= LAT.max()+buf) &
                (h_lon >= LON.min()-buf) & (h_lon <= LON.max()+buf))
        rows, cols = np.where(mask)
        r0,r1,c0,c1 = rows.min(), rows.max()+1, cols.min(), cols.max()+1
        sub_lat = h_lat[r0:r1, c0:c1]
        sub_lon = h_lon[r0:r1, c0:c1]

        out_vars = {}
        for varname in list(merged.data_vars):
            data = merged[varname].values
            while data.ndim > 2 and data.shape[0] == 1:
                data = data.squeeze(0)
            if data.ndim != 2:
                continue
            sub = data[r0:r1, c0:c1]
            lat_1d = np.mean(sub_lat, axis=1)
            lon_1d = np.mean(sub_lon, axis=0)
            si, sj = np.argsort(lat_1d), np.argsort(lon_1d)
            d_s = sub[np.ix_(si,sj)].copy()
            bad = ~np.isfinite(d_s)
            if bad.any(): d_s[bad] = np.nanmean(d_s)
            fn = RegularGridInterpolator(
                (lat_1d[si], lon_1d[sj]), d_s,
                method='linear', bounds_error=False, fill_value=np.nan)
            pts = np.column_stack([OUT_LAT.ravel(), OUT_LON.ravel()])
            out_vars[varname] = fn(pts).reshape(OUT_LAT.shape).astype(np.float32)

        # Build output dataset with ERA5 lat convention (descending)
        out_ds = xr.Dataset(
            {k: xr.DataArray(v[::-1], dims=['latitude','longitude'])
             for k, v in out_vars.items()},
            coords={
                'latitude':  ('latitude',  LAT),
                'longitude': ('longitude', LON),
            },
            attrs={
                'source':     'HRRR',
                'cycle':      cycle_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'valid_time': cycle_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            }
        )
        out_ds.to_netcdf(out_path)
        print(f'  Saved: {fname}  ({out_path.stat().st_size/1024:.1f} KB)')
        return out_path

    except Exception as e:
        print(f'  HRRR fetch failed: {e}')
        print('  Falling back to ERA5 background...')
        return _era5_fallback(cycle_dt)


def _era5_fallback(cycle_dt: datetime) -> Path:
    """Use ERA5 if HRRR unavailable."""
    tag = cycle_dt.strftime('%Y_%m')
    p   = PROC_DIR / f'era5_sfc_{tag}.nc'
    if p.exists():
        print(f'  Using ERA5 fallback: {p.name}')
        return p
    raise FileNotFoundError(
        f'No HRRR and no ERA5 fallback found for {cycle_dt:%Y-%m}. '
        f'Run Module 1 first to download ERA5 data.'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: ASOS Observations (live from Iowa Mesonet API)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_asos_obs(cycle_dt: datetime, window_min: int = 60) -> dict:
    """
    Fetch live ASOS wind observations from Iowa Environmental Mesonet.
    Falls back to cached CSV if network unavailable.
    Returns dict: station -> wind speed m/s.
    """
    print(f'  Fetching ASOS observations for {cycle_dt:%Y-%m-%d %H:%M} UTC...')

    t_start = (cycle_dt - timedelta(minutes=window_min)).strftime('%Y%m%d%H%M')
    t_end   = (cycle_dt + timedelta(minutes=window_min)).strftime('%Y%m%d%H%M')
    stn_str = ','.join(ASOS_STATION_IDS.values())

    url = (
        f'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
        f'?station={stn_str}'
        f'&data=sknt'
        f'&year1={t_start[:4]}&month1={t_start[4:6]}&day1={t_start[6:8]}'
        f'&hour1={t_start[8:10]}&minute1={t_start[10:12]}'
        f'&year2={t_end[:4]}&month2={t_end[4:6]}&day2={t_end[6:8]}'
        f'&hour2={t_end[8:10]}&minute2={t_end[10:12]}'
        f'&tz=utc&format=comma&latlon=no&missing=M&trace=T&direct=no&report_type=1'
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        lines = [l for l in resp.text.strip().split('\n')
                 if not l.startswith('#') and l.strip()]
        if len(lines) < 2:
            raise ValueError('Empty API response')

        df = pd.read_csv(
            __import__('io').StringIO('\n'.join(lines)),
            parse_dates=['valid']
        )
        df.columns = [c.strip().lower() for c in df.columns]
        df['sknt'] = pd.to_numeric(df['sknt'], errors='coerce')
        df = df.dropna(subset=['sknt'])
        df['wspd_ms'] = df['sknt'] * 0.51444

        target = pd.Timestamp(cycle_dt, tz='UTC')
        results = {}
        for name, sid in ASOS_STATION_IDS.items():
            sub = df[df['station'].str.strip().str.upper() == sid.upper()].copy()
            if sub.empty:
                print(f'    {name}: no live obs')
                results[name] = None
                continue
            sub['dt'] = (pd.to_datetime(sub['valid'], utc=True) - target).abs()
            row = sub.loc[sub['dt'].idxmin()]
            results[name] = round(float(row['wspd_ms']), 2)
            print(f'    {name}: {results[name]:.2f} m/s')
        return results

    except Exception as e:
        print(f'  Live ASOS fetch failed ({e}), trying cached CSV...')
        return _asos_fallback(cycle_dt)


def _asos_fallback(cycle_dt: datetime) -> dict:
    """Use cached monthly CSV if live fetch fails."""
    tag  = cycle_dt.strftime('%Y_%m')
    path = PROC_DIR / f'asos_{tag}.csv'
    if not path.exists():
        print(f'  No cached ASOS data for {tag}')
        return {s: None for s in STATIONS}

    df = pd.read_csv(path, parse_dates=['valid_time'])
    df = df.set_index('valid_time').sort_index()
    target = pd.Timestamp(cycle_dt)
    window = df[
        (df.index >= target - pd.Timedelta('1h')) &
        (df.index <= target + pd.Timedelta('1h'))
    ]
    results = {}
    for name in STATIONS:
        sub = window[window['station'] == name]
        if sub.empty:
            results[name] = None
        else:
            sub = sub.copy()
            sub['dt'] = (sub.index - target).map(abs)
            results[name] = round(float(sub.loc[sub['dt'].idxmin(), 'wind_speed_mps']), 2)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: EnKF Analysis (Module 4 logic)
# ═══════════════════════════════════════════════════════════════════════════════

def build_H(station_names: list) -> np.ndarray:
    """Build observation operator mapping grid to station locations."""
    n_grid = len(LAT) * len(LON)
    H      = np.zeros((len(station_names), n_grid))
    for i, name in enumerate(station_names):
        s_lat, s_lon = STATIONS[name]
        li = int(np.argmin(np.abs(LAT - s_lat)))
        lj = int(np.argmin(np.abs(LON - s_lon)))
        H[i, li * len(LON) + lj] = 1.0
    return H


def enkf_analysis(x_bg, H, y_obs,
                  obs_error=OBS_ERROR, bg_error=BG_ERROR,
                  n_ensemble=N_ENSEMBLE, seed=42):
    """EnKF analysis step — identical to Module 4."""
    rng      = np.random.default_rng(seed)
    n_grid   = len(x_bg)
    ensemble = x_bg + rng.normal(0, bg_error, (n_ensemble, n_grid))
    x_mean   = ensemble.mean(axis=0)
    X_pert   = ensemble - x_mean
    P_b      = (X_pert.T @ X_pert) / (n_ensemble - 1)
    R        = np.eye(len(y_obs)) * obs_error**2
    HPH_T    = H @ P_b @ H.T
    K        = P_b @ H.T @ np.linalg.inv(HPH_T + R)
    innov    = y_obs - H @ x_bg
    x_an     = x_bg + K @ innov
    return x_an, innov


def run_enkf(nc_path: Path, obs: dict):
    """
    Load background from NetCDF, run EnKF with ASOS obs.
    Returns (ws_bg_grid, ws_an_grid, u_an_grid, v_an_grid, innovations, obs_stations).
    """
    ds = xr.open_dataset(nc_path)

    # Extract u10, v10 on (lat, lon) grid
    u10 = ds['u10'].values if 'u10' in ds.data_vars else None
    v10 = ds['v10'].values if 'v10' in ds.data_vars else None

    # Handle time dimension if present
    if u10 is not None and u10.ndim == 3: u10 = u10[0]
    if v10 is not None and v10.ndim == 3: v10 = v10[0]

    # Fallback: try u/v at 1000 hPa
    if u10 is None and 'u' in ds.data_vars:
        u10 = ds['u'].values[0, -1] if ds['u'].ndim == 4 else ds['u'].values[-1]
        v10 = ds['v'].values[0, -1] if ds['v'].ndim == 4 else ds['v'].values[-1]

    source = ds.attrs.get('source', 'unknown')
    ds.close()

    if u10 is None:
        raise ValueError(f'Could not find wind components in {nc_path}')

    # Ensure shape matches grid
    if u10.shape != (len(LAT), len(LON)):
        raise ValueError(f'Wind shape {u10.shape} != grid ({len(LAT)},{len(LON)})')

    ws_flat = np.sqrt(u10.flatten()**2 + v10.flatten()**2)

    # Build obs vector from available stations only
    obs_stations = [n for n, v in obs.items() if v is not None]
    if not obs_stations:
        raise ValueError('No ASOS observations available')

    y_obs = np.array([obs[n] for n in obs_stations])
    H     = build_H(obs_stations)

    ws_an_flat, innov = enkf_analysis(ws_flat, H, y_obs)

    ws_bg_grid = ws_flat.reshape(len(LAT), len(LON))
    ws_an_grid = ws_an_flat.reshape(len(LAT), len(LON))

    # Scale u/v to match corrected wind speed (preserve direction)
    scale     = np.where(ws_bg_grid > 0.01, ws_an_grid / ws_bg_grid, 1.0)
    u_an_grid = u10 * scale
    v_an_grid = v10 * scale

    return ws_bg_grid, ws_an_grid, u_an_grid, v_an_grid, innov, obs_stations, source


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Physics / Microburst Detection (Module 3)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_microburst(u, v, threshold=0.003):
    """Surface divergence microburst detection — identical to Module 3."""
    du_dx = np.gradient(u, DX, axis=1)
    dv_dy = np.gradient(v, DY, axis=0)
    div   = du_dx + dv_dy
    max_d = float(np.max(div))
    idx   = np.unravel_index(np.argmax(div), div.shape)
    loc   = (float(LAT[idx[0]]), float(LON[idx[1]]))
    alert = max_d > threshold
    return alert, div, max_d, loc


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Print Decision
# ═══════════════════════════════════════════════════════════════════════════════

def print_decision(
    cycle_dt, source, obs, obs_stations,
    ws_bg_grid, ws_an_grid, innov,
    alert, div_field, max_div, mb_loc,
):
    W = 60
    print()
    print('=' * W)
    print('  HyperWind-Now | DFW Low-Altitude Wind Hazard Forecast')
    print('=' * W)
    print(f'  Cycle   : {cycle_dt:%Y-%m-%d %H:%M} UTC')
    print(f'  Source  : {source} background + live ASOS')
    print(f'  Run at  : {datetime.utcnow():%Y-%m-%d %H:%M} UTC')
    print('-' * W)
    print(f'  {"Station":<8}  {"Obs":>7}  {"Background":>11}  {"Innovation":>11}')
    print(f'  {"-"*8}  {"-"*7}  {"-"*11}  {"-"*11}')
    for i, name in enumerate(obs_stations):
        li = int(np.argmin(np.abs(LAT - STATIONS[name][0])))
        lj = int(np.argmin(np.abs(LON - STATIONS[name][1])))
        bg = ws_bg_grid[li, lj]
        print(f'  {name:<8}  {obs[name]:>6.2f}  {bg:>10.2f}  {innov[i]:>+10.2f}  m/s')
    print('-' * W)
    bg_mean  = np.mean([ws_bg_grid[
        int(np.argmin(np.abs(LAT - STATIONS[n][0]))),
        int(np.argmin(np.abs(LON - STATIONS[n][1])))
    ] for n in obs_stations])
    an_mean  = np.mean([ws_an_grid[
        int(np.argmin(np.abs(LAT - STATIONS[n][0]))),
        int(np.argmin(np.abs(LON - STATIONS[n][1])))
    ] for n in obs_stations])
    print(f'  Domain mean background : {bg_mean:.2f} m/s')
    print(f'  Domain mean analysis   : {an_mean:.2f} m/s')
    print(f'  Max surface divergence : {max_div:.2e} s⁻¹')
    print(f'  Microburst location    : lat={mb_loc[0]:.2f}, lon={mb_loc[1]:.2f}')
    print('=' * W)

    if alert:
        print()
        print('  ⚠️  WARNING: MICROBURST DETECTED')
        print(f'     Divergence {max_div:.2e} s⁻¹ exceeds threshold 0.003 s⁻¹')
        print(f'     Location: lat={mb_loc[0]:.2f}, lon={mb_loc[1]:.2f}')
        print()
        print('  ██████████████████████████████████████████')
        print('  ██                                      ██')
        print('  ██           NO-FLY ZONE                ██')
        print('  ██    Do not launch drones at this time  ██')
        print('  ██                                      ██')
        print('  ██████████████████████████████████████████')
    else:
        print()
        print('  ✅  CLEAR TO FLY')
        print(f'     No microburst detected (max div = {max_div:.2e} s⁻¹)')
        print(f'     Mean analysis wind speed: {an_mean:.2f} m/s over DFW')
        print()
        print('  ██████████████████████████████████████████')
        print('  ██                                      ██')
        print('  ██          CLEAR TO FLY                ██')
        print('  ██    Conditions within normal limits    ██')
        print('  ██                                      ██')
        print('  ██████████████████████████████████████████')
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Optional: Diagnostic Plot
# ═══════════════════════════════════════════════════════════════════════════════

def make_plot(cycle_dt, ws_bg_grid, ws_an_grid, u_an_grid, v_an_grid,
              div_field, alert, mb_loc, obs, obs_stations):
    import matplotlib.pyplot as plt

    BG, PAN = '#0d1117', '#161b22'
    extent  = [LON.min(), LON.max(), LAT.min(), LAT.max()]
    vmax    = max(ws_bg_grid.max(), ws_an_grid.max(), 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig.suptitle(
        f'HyperWind-Now | DFW Wind Analysis\n{cycle_dt:%Y-%m-%d %H:%M} UTC',
        color='white', fontsize=13, fontweight='bold'
    )

    titles = ['HRRR Background', 'EnKF Analysis', 'Microburst Risk']
    datas  = [ws_bg_grid, ws_an_grid,
              np.clip(div_field * 1000, 0, None)]
    cmaps  = ['YlOrRd', 'YlOrRd', 'hot_r']
    vmaxes = [vmax, vmax, max(float(np.clip(div_field*1000,0,None).max()), 0.001)]

    for ax, data, title, cmap, vm in zip(axes, datas, titles, cmaps, vmaxes):
        ax.set_facecolor(PAN)
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vm,
                       origin='upper', extent=extent, aspect='auto')
        plt.colorbar(im, ax=ax, label='m/s' if 'Risk' not in title else '×10⁻³ s⁻¹')
        ax.set_title(title, color='white', fontsize=10)
        ax.set_xlabel('Longitude', color='#8b949e', fontsize=8)
        ax.set_ylabel('Latitude',  color='#8b949e', fontsize=8)
        ax.tick_params(colors='#8b949e', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

        for name in obs_stations:
            s_lat, s_lon = STATIONS[name]
            ax.plot(s_lon, s_lat, 'o', color='#58a6ff', ms=6, zorder=5)
            val = obs[name]
            label = f'{name}\n{val:.1f}' if val is not None else name
            ax.text(s_lon+0.03, s_lat+0.02, label,
                    color='#58a6ff', fontsize=6.5, fontweight='bold', zorder=6)

    if alert:
        axes[2].plot(mb_loc[1], mb_loc[0], '*', color='cyan',
                     ms=15, zorder=7, label='ALERT')
        axes[2].legend(facecolor='#21262d', labelcolor='white', fontsize=9)

    plt.tight_layout()
    out = ROOT_DIR / 'results' / f'live_{cycle_dt:%Y%m%d_%H%M}Z.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'  Plot saved: {out}')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def latest_hrrr_cycle() -> datetime:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


def main():
    parser = argparse.ArgumentParser(description='HyperWind-Now live forecast')
    parser.add_argument('--cycle', type=str, default=None,
                        help='Cycle datetime e.g. "2021-04-06 21:00"')
    parser.add_argument('--plot', action='store_true',
                        help='Show diagnostic plot')
    args = parser.parse_args()

    if args.cycle:
        cycle_dt = datetime.strptime(args.cycle, '%Y-%m-%d %H:%M')
    else:
        cycle_dt = latest_hrrr_cycle()

    print()
    print('HyperWind-Now | Starting live pipeline')
    print(f'Cycle: {cycle_dt:%Y-%m-%d %H:%M} UTC')
    print()

    # Step 1: HRRR background
    print('[1/4] Fetching HRRR background...')
    t0 = time.time()
    nc_path = fetch_hrrr(cycle_dt)
    print(f'      Done in {time.time()-t0:.1f}s')

    # Step 2: ASOS observations
    print('[2/4] Fetching ASOS observations...')
    obs = fetch_asos_obs(cycle_dt)

    # Step 3: EnKF
    print('[3/4] Running EnKF assimilation...')
    ws_bg, ws_an, u_an, v_an, innov, obs_stns, source = run_enkf(nc_path, obs)
    print(f'      {len(obs_stns)} stations assimilated: {obs_stns}')

    # Step 4: Physics
    print('[4/4] Running microburst detection...')
    alert, div, max_div, mb_loc = detect_microburst(u_an, v_an)

    # Decision
    print_decision(cycle_dt, source, obs, obs_stns,
                   ws_bg, ws_an, innov, alert, div, max_div, mb_loc)

    if args.plot:
        make_plot(cycle_dt, ws_bg, ws_an, u_an, v_an,
                  div, alert, mb_loc, obs, obs_stns)

    return 0 if not alert else 1


if __name__ == '__main__':
    sys.exit(main())
