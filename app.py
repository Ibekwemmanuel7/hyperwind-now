"""
app.py — HyperWind-Now Streamlit Dashboard
==========================================
Run with:
    streamlit run app.py

Displays a live HRRR-driven wind hazard forecast for the DFW domain.
Reuses all logic from run_live.py — no duplication.
"""

import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

# ── Import pipeline functions from run_live.py ────────────────────────────────
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from run_live import (
    fetch_hrrr, fetch_asos_obs, run_enkf, detect_microburst,
    latest_hrrr_cycle, STATIONS, LAT, LON,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title='HyperWind-Now | DFW Wind Hazard',
    page_icon='🌬️',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .block-container { padding-top: 1rem; }
    .alert-box {
        border-radius: 8px;
        padding: 20px 24px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        margin: 12px 0;
    }
    .clear  { background: #1a3a1a; border: 2px solid #3fb950; color: #3fb950; }
    .nofly  { background: #3a1a1a; border: 2px solid #f85149; color: #f85149; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('## 🌬️ HyperWind-Now')
    st.markdown('**DFW Low-Altitude Wind Hazard**')
    st.markdown('---')

    mode = st.radio('Cycle mode', ['Latest available', 'Custom cycle'])

    if mode == 'Latest available':
        cycle_dt = latest_hrrr_cycle()
        st.info(f'Latest: {cycle_dt:%Y-%m-%d %H:%M} UTC')
    else:
        date_val = st.date_input('Date', value=datetime(2021, 4, 6).date())
        hour_val = st.selectbox('Hour (UTC)', list(range(24)), index=21)
        cycle_dt = datetime(date_val.year, date_val.month, date_val.day, hour_val)

    st.markdown('---')
    run_btn = st.button('▶  Run Forecast', use_container_width=True, type='primary')

    st.markdown('---')
    st.markdown('**Pipeline**')
    st.markdown('1. HRRR fetch (AWS S3)')
    st.markdown('2. ASOS observations')
    st.markdown('3. EnKF assimilation')
    st.markdown('4. Microburst detection')
    st.markdown('---')
    st.markdown('**Domain**')
    st.markdown('32.25–33.25°N, 96.5–97.75°W')
    st.markdown('6 ASOS stations · 50-member EnKF')
    st.markdown('---')
    st.caption('Emmanuel Ibekwe | HyperWind-Now')


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('# 🌬️ HyperWind-Now')
st.markdown('#### Physics-Constrained ML Wind Hazard Forecast | Dallas-Fort Worth')
st.markdown('---')

if not run_btn:
    st.markdown("""
    <div style="text-align:center; padding: 60px; color: #8b949e;">
        <div style="font-size: 4rem;">🌬️</div>
        <div style="font-size: 1.2rem; margin-top: 16px;">
            Select a cycle and click <strong>Run Forecast</strong> to start
        </div>
        <div style="font-size: 0.9rem; margin-top: 8px;">
            Pipeline: HRRR (AWS S3) → EnKF assimilation → Microburst detection → CLEAR / NO-FLY
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# Run pipeline
# ═══════════════════════════════════════════════════════════════════════════════

progress = st.progress(0, text='Starting pipeline...')
status   = st.empty()

try:
    status.markdown('**[1/4]** Fetching HRRR background...')
    t0      = time.time()
    nc_path = fetch_hrrr(cycle_dt)
    t_fetch = time.time() - t0
    progress.progress(25, text='HRRR fetched')

    status.markdown('**[2/4]** Fetching ASOS observations...')
    obs = fetch_asos_obs(cycle_dt)
    progress.progress(50, text='ASOS loaded')

    status.markdown('**[3/4]** Running EnKF assimilation...')
    ws_bg, ws_an, u_an, v_an, innov, obs_stns, source = run_enkf(nc_path, obs)
    progress.progress(75, text='EnKF complete')

    status.markdown('**[4/4]** Running microburst detection...')
    alert, div_field, max_div, mb_loc = detect_microburst(u_an, v_an)
    progress.progress(100, text='Done')
    status.empty()
    progress.empty()

except Exception as e:
    progress.empty()
    status.empty()
    st.error(f'Pipeline error: {e}')
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# Decision banner
# ═══════════════════════════════════════════════════════════════════════════════

if alert:
    st.markdown(
        '<div class="alert-box nofly">⚠️ NO-FLY ZONE — MICROBURST DETECTED</div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="alert-box clear">✅ CLEAR TO FLY</div>',
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Top metrics
# ═══════════════════════════════════════════════════════════════════════════════

bg_mean = float(np.mean([
    ws_bg[int(np.argmin(np.abs(LAT - STATIONS[n][0]))),
          int(np.argmin(np.abs(LON - STATIONS[n][1])))]
    for n in obs_stns
]))
an_mean = float(np.mean([
    ws_an[int(np.argmin(np.abs(LAT - STATIONS[n][0]))),
          int(np.argmin(np.abs(LON - STATIONS[n][1])))]
    for n in obs_stns
]))

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric('Cycle', f'{cycle_dt:%Y-%m-%d %H:%M}Z')
with c2:
    st.metric('Source', source)
with c3:
    st.metric('Bg Wind', f'{bg_mean:.2f} m/s')
with c4:
    st.metric('Analysis Wind', f'{an_mean:.2f} m/s', delta=f'{an_mean-bg_mean:+.2f}')
with c5:
    st.metric('Max Divergence', f'{max_div:.2e} s⁻¹',
              delta='ALERT' if alert else 'OK',
              delta_color='inverse' if alert else 'normal')

st.markdown('---')


# ═══════════════════════════════════════════════════════════════════════════════
# Main wind plots
# ═══════════════════════════════════════════════════════════════════════════════

BG, PAN = '#0d1117', '#161b22'
extent  = [LON.min(), LON.max(), LAT.min(), LAT.max()]
vmax    = max(float(ws_bg.max()), float(ws_an.max()), 0.1)

fig = plt.figure(figsize=(18, 6), facecolor=BG)
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3,
                        left=0.05, right=0.97, top=0.88, bottom=0.12)

panels = [
    (ws_bg,                              'YlOrRd', vmax,  'HRRR Background (m/s)'),
    (ws_an,                              'YlOrRd', vmax,  'EnKF Analysis (m/s)'),
    (np.clip(div_field * 1000, 0, None), 'hot_r',  None,  'Microburst Risk (×10⁻³ s⁻¹)'),
]

for i, (data, cmap, vm, title) in enumerate(panels):
    ax = fig.add_subplot(gs[i])
    ax.set_facecolor(PAN)
    if vm is None:
        vm = max(float(data.max()), 0.001)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vm,
                   origin='upper', extent=extent, aspect='auto')
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    for name in obs_stns:
        s_lat, s_lon = STATIONS[name]
        ax.plot(s_lon, s_lat, 'o', color='#58a6ff', ms=7, zorder=5)
        val = obs.get(name)
        lbl = f'{name}\n{val:.1f}' if val else name
        ax.text(s_lon+0.03, s_lat+0.02, lbl,
                color='#58a6ff', fontsize=7, fontweight='bold', zorder=6)

    if alert and i == 2:
        ax.plot(mb_loc[1], mb_loc[0], '*', color='cyan', ms=18, zorder=7)

    ax.set_title(title, color='white', fontsize=11, pad=8)
    ax.set_xlabel('Longitude', color='#8b949e', fontsize=8)
    ax.set_ylabel('Latitude',  color='#8b949e', fontsize=8)
    ax.tick_params(colors='#8b949e', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')

fig.suptitle(
    f'HyperWind-Now | DFW Wind Analysis | {cycle_dt:%Y-%m-%d %H:%M} UTC',
    color='white', fontsize=13, fontweight='bold')
st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Station table + Innovation bar chart
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('---')
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('#### Station Observations & Innovations')
    rows = []
    for i, name in enumerate(obs_stns):
        li  = int(np.argmin(np.abs(LAT - STATIONS[name][0])))
        lj  = int(np.argmin(np.abs(LON - STATIONS[name][1])))
        bg  = float(ws_bg[li, lj])
        inn = float(innov[i])
        rows.append({
            'Station':     name,
            'Obs (m/s)':   round(float(obs[name]), 2),
            'Bg (m/s)':    round(bg, 2),
            'Innovation':  round(inn, 2),
            'Status':      '↑ obs > bg' if inn > 0 else '↓ obs < bg',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with col_right:
    st.markdown('#### Innovation by Station')
    fig2, ax2 = plt.subplots(figsize=(7, 4), facecolor=BG)
    ax2.set_facecolor(PAN)
    colors = ['#3fb950' if v >= 0 else '#f78166' for v in innov]
    bars   = ax2.bar(obs_stns, innov, color=colors, alpha=0.85,
                     edgecolor='#30363d', linewidth=0.8, zorder=3)
    ax2.axhline(0, color='white', lw=1, alpha=0.5)
    for bar, val in zip(bars, innov):
        ypos = val + 0.05 if val >= 0 else val - 0.15
        ax2.text(bar.get_x() + bar.get_width()/2, ypos,
                 f'{val:+.2f}', ha='center',
                 va='bottom' if val >= 0 else 'top',
                 color='white', fontsize=8.5, fontweight='bold')
    ax2.set_ylabel('Obs − Background (m/s)', color='#8b949e', fontsize=9)
    ax2.set_xlabel('Station', color='#8b949e', fontsize=9)
    ax2.tick_params(colors='#8b949e', labelsize=9)
    ax2.yaxis.grid(True, color='#30363d', lw=0.5, zorder=0)
    ax2.set_axisbelow(True)
    for sp in ax2.spines.values():
        sp.set_edgecolor('#30363d')
    fig2.patch.set_facecolor(BG)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('---')
fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.caption(f'Pipeline completed at {datetime.utcnow():%Y-%m-%d %H:%M} UTC')
with fc2:
    st.caption(f'HRRR fetch: {t_fetch:.1f}s · EnKF: 50 members · Threshold: 0.003 s⁻¹')
with fc3:
    st.caption('HyperWind-Now · Emmanuel Ibekwe · github.com/Ibekwemmanuel7/hyperwind-now')
