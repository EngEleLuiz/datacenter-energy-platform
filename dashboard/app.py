"""
dashboard/app.py
================
Datacenter Energy Intelligence Platform — Executive Dashboard
Visualises GFL vs GFM inverter dynamics with 5 academic features:
  1. Virtual Inertia (VSM)
  2. Black-Start Capability
  3. Active Harmonic Compensation
  4. Droop Control (P/f, Q/V)
  5. Weak-Grid Stability Map
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
import math

from data_generator.server_simulator import ServerSimulator
from data_generator.ups_inverter_simulator import (
    UPSSimulator, InverterSimulator,
    NOMINAL_FREQ_HZ, NOMINAL_VOLTAGE_V,
    IEEE1547_ROCOF_LIMIT, IEEE519_THD_LIMIT_PCT,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DC Energy Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
DARK_BG       = "#0A0E1A"
CARD_BG       = "#111827"
CARD_BORDER   = "#1F2937"
ACCENT_CYAN   = "#00D4FF"
ACCENT_GREEN  = "#00FF9F"
ACCENT_RED    = "#FF4757"
ACCENT_AMBER  = "#FFB020"
ACCENT_PURPLE = "#A855F7"
TEXT_PRIMARY  = "#F1F5F9"
TEXT_MUTED    = "#64748B"

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'JetBrains Mono', monospace", color=TEXT_PRIMARY, size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'JetBrains Mono', monospace;
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
}}

.main .block-container {{ padding: 1rem 2rem; max-width: 100%; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1117 0%, #0A0E1A 100%);
    border-right: 1px solid {CARD_BORDER};
}}

/* KPI Cards */
.kpi-card {{
    background: linear-gradient(135deg, {CARD_BG} 0%, #0D1B2A 100%);
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {ACCENT_CYAN}, {ACCENT_GREEN});
}}
.kpi-card.red::before  {{ background: linear-gradient(90deg, {ACCENT_RED}, {ACCENT_AMBER}); }}
.kpi-card.amber::before {{ background: linear-gradient(90deg, {ACCENT_AMBER}, {ACCENT_PURPLE}); }}
.kpi-card.purple::before {{ background: linear-gradient(90deg, {ACCENT_PURPLE}, {ACCENT_CYAN}); }}
.kpi-label  {{ font-size: 0.68rem; color: {TEXT_MUTED}; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }}
.kpi-value  {{ font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; color: {TEXT_PRIMARY}; line-height: 1; }}
.kpi-sub    {{ font-size: 0.70rem; color: {TEXT_MUTED}; margin-top: 4px; }}
.kpi-badge  {{ display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 0.65rem; font-weight: 600; margin-top: 6px; }}
.badge-ok   {{ background: rgba(0,255,159,0.15); color: {ACCENT_GREEN}; border: 1px solid rgba(0,255,159,0.3); }}
.badge-warn {{ background: rgba(255,176,32,0.15); color: {ACCENT_AMBER}; border: 1px solid rgba(255,176,32,0.3); }}
.badge-err  {{ background: rgba(255,71,87,0.15);  color: {ACCENT_RED};   border: 1px solid rgba(255,71,87,0.3); }}

/* Section headers */
.section-header {{
    display: flex; align-items: center; gap: 10px;
    border-bottom: 1px solid {CARD_BORDER};
    padding-bottom: 8px; margin-bottom: 20px; margin-top: 10px;
}}
.section-header h3 {{
    font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
    color: {ACCENT_CYAN}; letter-spacing: 0.05em; margin: 0;
}}
.section-header .tag {{
    font-size: 0.60rem; padding: 2px 8px; border-radius: 4px;
    background: rgba(0,212,255,0.12); color: {ACCENT_CYAN};
    border: 1px solid rgba(0,212,255,0.25); letter-spacing: 0.08em;
}}

/* Feature badge */
.feature-pill {{
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.62rem; font-weight: 600; letter-spacing: 0.08em;
    margin: 2px; text-transform: uppercase;
}}
.pill-vsm    {{ background:rgba(0,212,255,0.15); color:{ACCENT_CYAN};   border:1px solid rgba(0,212,255,0.3); }}
.pill-bs     {{ background:rgba(255,176,32,0.15); color:{ACCENT_AMBER}; border:1px solid rgba(255,176,32,0.3); }}
.pill-harm   {{ background:rgba(168,85,247,0.15); color:{ACCENT_PURPLE};border:1px solid rgba(168,85,247,0.3); }}
.pill-droop  {{ background:rgba(0,255,159,0.15);  color:{ACCENT_GREEN}; border:1px solid rgba(0,255,159,0.3); }}
.pill-wg     {{ background:rgba(255,71,87,0.15);  color:{ACCENT_RED};   border:1px solid rgba(255,71,87,0.3); }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
::-webkit-scrollbar-thumb {{ background: {CARD_BORDER}; border-radius: 2px; }}

/* Hide Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ── Helper: plotly layout ─────────────────────────────────────────────────────
def apply_theme(fig, height=380, title=""):
    fig.update_layout(
        height=height, title=title,
        **PLOTLY_THEME,
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor=CARD_BORDER,
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(gridcolor=CARD_BORDER, zerolinecolor=CARD_BORDER),
        yaxis=dict(gridcolor=CARD_BORDER, zerolinecolor=CARD_BORDER),
    )
    return fig


def kpi(label, value, sub="", variant="", badge="", badge_type="ok"):
    badge_html = f'<div class="kpi-badge badge-{badge_type}">{badge}</div>' if badge else ""
    st.markdown(f"""
    <div class="kpi-card {variant}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def section(icon, title, tag=""):
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    st.markdown(f"""
    <div class="section-header">
        <h3>{icon} {title}</h3>
        {tag_html}
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                    color:{ACCENT_CYAN};letter-spacing:0.04em;">⚡ DC ENERGY</div>
        <div style="font-size:0.65rem;color:{TEXT_MUTED};letter-spacing:0.15em;
                    text-transform:uppercase;margin-top:2px;">Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.7rem;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Parameters</div>", unsafe_allow_html=True)

    num_servers    = st.slider("Servers",            50, 200, 100, 10)
    num_inverters  = st.slider("Inverters",           2,   8,   4)
    islanding_prob = st.slider("Islanding Prob.",  0.00, 0.15, 0.05, 0.01)
    bs_prob        = st.slider("Black-Start Prob.", 0.00, 0.03, 0.01, 0.005)
    scr_min        = st.slider("SCR Min",           0.5,  3.0,  1.0, 0.5)
    history_hours  = st.slider("History (hours)",     1,  24,    6)

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.7rem;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio("", [
        "Overview",
        "① Virtual Inertia",
        "② Black-Start",
        "③ Harmonics",
        "④ Droop Control",
        "⑤ Weak-Grid Stability",
    ], label_visibility="collapsed")

    st.markdown("---")
    refresh = st.button("⟳  Refresh Data", use_container_width=True)
    st.markdown(f"<div style='font-size:0.62rem;color:{TEXT_MUTED};text-align:center;margin-top:6px'>{datetime.now().strftime('%H:%M:%S')} UTC</div>", unsafe_allow_html=True)

    # Feature pills
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div>
        <span class="feature-pill pill-vsm">VSM</span>
        <span class="feature-pill pill-bs">Black-Start</span>
        <span class="feature-pill pill-harm">Harmonics</span>
        <span class="feature-pill pill-droop">Droop</span>
        <span class="feature-pill pill-wg">Weak-Grid</span>
    </div>
    """, unsafe_allow_html=True)


# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner="Generating telemetry…")
def load_data(num_servers, num_inverters, islanding_prob, bs_prob, scr_min, hours, _r):
    srv_sim = ServerSimulator(num_servers=num_servers, num_racks=10,
                              fault_probability=0.01, random_seed=None)
    ups_sim = UPSSimulator(num_ups=4, vsm_inertia_H=5.0)
    inv_sim = InverterSimulator(
        num_inverters=num_inverters,
        islanding_probability=islanding_prob,
        black_start_probability=bs_prob,
        scr_range=(scr_min, 12.0),
        vsm_inertia_H_range=(2.0, 10.0),
        droop_kw_hz=20.0,
        droop_kvar_v=5.0,
    )
    start  = datetime.now(timezone.utc) - timedelta(hours=hours)
    steps  = hours * 12
    srv_r, ups_r, inv_r = [], [], []
    for i in range(steps):
        ts = start + timedelta(minutes=5 * i)
        srv_r.extend([asdict(r) for r in srv_sim.generate_snapshot(ts)])
        ups_r.extend([asdict(r) for r in ups_sim.generate_snapshot(ts)])
        inv_r.extend([asdict(r) for r in inv_sim.generate_snapshot(ts)])

    df_s = pd.DataFrame(srv_r)
    df_u = pd.DataFrame(ups_r)
    df_i = pd.DataFrame(inv_r)
    for d in [df_s, df_u, df_i]:
        d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)
    return df_s, df_u, df_i

df_srv, df_ups, df_inv = load_data(
    num_servers, num_inverters, islanding_prob, bs_prob, scr_min, history_hours, refresh
)

latest_srv = df_srv[df_srv["timestamp_utc"] == df_srv["timestamp_utc"].max()]
df_gfl     = df_inv[df_inv["control_mode"] == "GFL"]
df_gfm     = df_inv[df_inv["control_mode"] == "GFM"]
df_gfx     = df_inv[df_inv["control_mode"].isin(["GFL", "GFM"])]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f"""
    <div style="margin-bottom:24px">
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
                    background:linear-gradient(90deg,{ACCENT_CYAN},{ACCENT_GREEN});
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    letter-spacing:0.02em;">
            Datacenter Microgrid Intelligence
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px;letter-spacing:0.08em">
            REAL-TIME GFL/GFM INVERTER ANALYTICS · IEEE 1547 · IEEE 519 · VIRTUAL SYNCHRONOUS MACHINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_kw    = latest_srv["power_draw_w"].sum() / 1000
    avg_pue     = latest_srv["pue_contribution"].mean()
    anomalies   = latest_srv["is_anomaly"].sum()
    gfm_count   = (df_inv["control_mode"] == "GFM").sum()
    bs_events   = df_inv["black_start_active"].sum()
    ieee_viol   = (df_gfl["rocof_hz_per_s"].abs() > IEEE1547_ROCOF_LIMIT).mean()

    with c1: kpi("IT Power", f"{total_kw:.0f}", "kW total", badge="Live", badge_type="ok")
    with c2: kpi("Avg PUE", f"{avg_pue:.4f}", "Power Usage Effectiveness", badge="Nominal", badge_type="ok")
    with c3: kpi("Anomalies", str(anomalies), "active faults", "red",
                 badge="Alert" if anomalies > 0 else "Clear",
                 badge_type="err" if anomalies > 0 else "ok")
    with c4: kpi("GFM Active", str(gfm_count), "inverter snapshots", "amber", badge="Grid-Forming", badge_type="warn")
    with c5: kpi("Black-Start", str(bs_events), "events detected", "purple", badge="GFM Only", badge_type="warn")
    with c6: kpi("IEEE Violations", f"{ieee_viol:.1%}", "GFL ROCOF > 0.5 Hz/s", "red",
                 badge="Critical" if ieee_viol > 0.05 else "OK",
                 badge_type="err" if ieee_viol > 0.05 else "ok")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("📈", "Power & Frequency Timeline", "48H OVERVIEW")
        dc_pwr = df_srv.groupby("timestamp_utc")["power_draw_w"].sum().reset_index()
        dc_pwr["kw"] = dc_pwr["power_draw_w"] / 1000
        inv_freq = df_inv.groupby("timestamp_utc")["output_frequency_hz"].mean().reset_index()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.06, row_heights=[0.6, 0.4])
        fig.add_trace(go.Scatter(
            x=dc_pwr["timestamp_utc"], y=dc_pwr["kw"],
            fill="tozeroy", name="IT Power (kW)",
            line=dict(color=ACCENT_CYAN, width=1.5),
            fillcolor=f"rgba(0,212,255,0.08)"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=inv_freq["timestamp_utc"], y=inv_freq["output_frequency_hz"],
            name="Grid Freq (Hz)", line=dict(color=ACCENT_GREEN, width=1.5)
        ), row=2, col=1)
        fig.add_hline(y=NOMINAL_FREQ_HZ, line_dash="dash",
                      line_color="rgba(255,255,255,0.2)", row=2, col=1)
        fig.update_layout(height=320, **PLOTLY_THEME)
        fig.update_xaxes(gridcolor=CARD_BORDER)
        fig.update_yaxes(gridcolor=CARD_BORDER)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section("🔵", "Control Mode Distribution", "REAL-TIME")
        mode_counts = df_inv["control_mode"].value_counts().reset_index()
        mode_counts.columns = ["mode", "count"]
        color_map = {"GFL": ACCENT_RED, "GFM": ACCENT_GREEN,
                     "transitioning": ACCENT_AMBER, "black_start": ACCENT_PURPLE}
        fig2 = go.Figure(go.Pie(
            labels=mode_counts["mode"],
            values=mode_counts["count"],
            hole=0.65,
            marker=dict(colors=[color_map.get(m, TEXT_MUTED) for m in mode_counts["mode"]],
                        line=dict(color=DARK_BG, width=3)),
            textfont=dict(size=11),
        ))
        fig2.add_annotation(text="Inverters", x=0.5, y=0.55,
                            font=dict(size=10, color=TEXT_MUTED), showarrow=False)
        fig2.add_annotation(text=str(len(df_inv["inverter_id"].unique())), x=0.5, y=0.42,
                            font=dict(size=28, color=TEXT_PRIMARY,
                                      family="Syne, sans-serif"), showarrow=False)
        fig2.update_layout(height=320, showlegend=True, **PLOTLY_THEME,
                           legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig2, use_container_width=True)

    # Stability overview heatmap
    section("🗺️", "Stability Overview — All Inverters", "MIDDLEBROOK CRITERION")
    stab_pivot = df_inv.groupby(["inverter_id", "stability_flag"])["scr"].count().unstack(fill_value=0)
    stab_pct   = stab_pivot.div(stab_pivot.sum(axis=1), axis=0) * 100
    flag_order = [c for c in ["stable", "marginal", "unstable"] if c in stab_pct.columns]
    fig3 = go.Figure()
    bar_colors = {"stable": ACCENT_GREEN, "marginal": ACCENT_AMBER, "unstable": ACCENT_RED}
    for flag in flag_order:
        if flag in stab_pct.columns:
            fig3.add_trace(go.Bar(
                name=flag.capitalize(), x=stab_pct.index,
                y=stab_pct[flag],
                marker_color=bar_colors[flag],
                marker_line=dict(width=0),
            ))
    fig3.update_layout(barmode="stack", height=220, **PLOTLY_THEME,
                       xaxis_title="Inverter", yaxis_title="% of Time")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VIRTUAL INERTIA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "① Virtual Inertia":
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_CYAN}">
            ① Virtual Synchronous Machine (VSM)
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            GFM inverters emulate synchronous generator inertia via the swing equation:
            &nbsp;<b style="color:{ACCENT_CYAN}">2H/ω₀ · dω/dt = P_mech − P_elec − D·Δω</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    gfl_rocof_mean = df_gfl["rocof_hz_per_s"].abs().mean()
    gfm_rocof_mean = df_gfm["rocof_hz_per_s"].abs().mean()
    ieee_gfl = (df_gfl["rocof_hz_per_s"].abs() > IEEE1547_ROCOF_LIMIT).mean()
    ieee_gfm = (df_gfm["rocof_hz_per_s"].abs() > IEEE1547_ROCOF_LIMIT).mean()
    with c1: kpi("GFL Mean |ROCOF|", f"{gfl_rocof_mean:.3f}", "Hz/s", "red",
                 badge="IEEE 1547", badge_type="err")
    with c2: kpi("GFM Mean |ROCOF|", f"{gfm_rocof_mean:.3f}", "Hz/s", "",
                 badge="IEEE 1547", badge_type="ok")
    with c3: kpi("GFL Violations", f"{ieee_gfl:.1%}", "ROCOF > 0.5 Hz/s", "red",
                 badge_type="err", badge="Non-Compliant" if ieee_gfl > 0 else "OK")
    with c4: kpi("GFM Violations", f"{ieee_gfm:.1%}", "ROCOF > 0.5 Hz/s", "",
                 badge_type="ok", badge="Compliant")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("📊", "ROCOF Distribution", "GFL vs GFM")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_gfl["rocof_hz_per_s"], name="GFL",
            marker_color=ACCENT_RED, opacity=0.75, nbinsx=60,
        ))
        fig.add_trace(go.Histogram(
            x=df_gfm["rocof_hz_per_s"], name="GFM",
            marker_color=ACCENT_GREEN, opacity=0.75, nbinsx=60,
        ))
        for sign in [1, -1]:
            fig.add_vline(x=sign * IEEE1547_ROCOF_LIMIT, line_dash="dash",
                          line_color="rgba(255,255,255,0.4)",
                          annotation_text="IEEE 1547" if sign == 1 else "",
                          annotation_font_size=9)
        fig.update_layout(barmode="overlay", **PLOTLY_THEME, height=320,
                          xaxis_title="ROCOF (Hz/s)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("⚡", "Inertia Constant H vs ROCOF at Nadir", "VSM THEORY")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_gfm["virtual_inertia_H"],
            y=df_gfm["rocof_at_nadir"].abs(),
            mode="markers",
            marker=dict(
                color=df_gfm["virtual_inertia_power_kw"].abs(),
                colorscale=[[0, ACCENT_PURPLE], [0.5, ACCENT_CYAN], [1, ACCENT_GREEN]],
                size=5, opacity=0.6,
                colorbar=dict(title="VSM Power (kW)", thickness=10),
            ),
            name="GFM inverters",
        ))
        fig2.update_layout(**PLOTLY_THEME, height=320,
                           xaxis_title="H constant (s)",
                           yaxis_title="|ROCOF at Nadir| (Hz/s)")
        st.plotly_chart(fig2, use_container_width=True)

    section("📉", "Virtual Inertia Power Injection Over Time", "VSM RESPONSE")
    fig3 = go.Figure()
    for inv_id in df_gfm["inverter_id"].unique():
        sub = df_gfm[df_gfm["inverter_id"] == inv_id]
        fig3.add_trace(go.Scatter(
            x=sub["timestamp_utc"], y=sub["virtual_inertia_power_kw"],
            name=inv_id, mode="lines", line=dict(width=1.2)
        ))
    fig3.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
    fig3.update_layout(**PLOTLY_THEME, height=260,
                       xaxis_title="Time", yaxis_title="VSM Power (kW)")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BLACK-START
# ══════════════════════════════════════════════════════════════════════════════
elif page == "② Black-Start":
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_AMBER}">
            ② Black-Start Capability
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            GFM forms voltage from scratch after a blackout.
            &nbsp;<b style="color:{ACCENT_AMBER}">GFL cannot black-start — it requires an external voltage reference for PLL lock.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    bs_events_df = df_inv[df_inv["black_start_stage"] > 0]
    stage_labels = {0: "Idle", 1: "Pre-Charge", 2: "Voltage Ramp",
                    3: "Load Pickup", 4: "Complete"}
    stage_colors = {0: TEXT_MUTED, 1: ACCENT_RED, 2: ACCENT_AMBER,
                    3: ACCENT_CYAN, 4: ACCENT_GREEN}

    c1, c2, c3, c4 = st.columns(4)
    total_bs   = df_inv["black_start_active"].sum()
    completed  = (df_inv["black_start_stage"] == 4).sum()
    max_loads  = df_inv["loads_reconnected"].max()
    gfl_bs     = df_gfl["black_start_active"].sum()
    with c1: kpi("Black-Start Events", str(total_bs), "GFM activations", "amber")
    with c2: kpi("Completed", str(completed), "full restorations", "", badge="Stage 4", badge_type="ok")
    with c3: kpi("Max Loads Restored", str(max_loads), "critical loads", "purple")
    with c4: kpi("GFL Black-Start", str(gfl_bs), "always zero", "red",
                 badge="Not Capable", badge_type="err")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("🔄", "Black-Start Stage Distribution", "ALL INVERTERS")
        stage_cnt = df_inv["black_start_stage"].value_counts().sort_index().reset_index()
        stage_cnt.columns = ["stage", "count"]
        stage_cnt["label"] = stage_cnt["stage"].map(stage_labels)
        stage_cnt["color"] = stage_cnt["stage"].map(stage_colors)
        fig = go.Figure(go.Bar(
            x=stage_cnt["label"], y=stage_cnt["count"],
            marker_color=stage_cnt["color"],
            marker_line=dict(width=0),
            text=stage_cnt["count"], textposition="outside",
        ))
        fig.update_layout(**PLOTLY_THEME, height=320,
                          xaxis_title="Stage", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("⚡", "Voltage Ramp Profile", "STAGE 2 — GFM ONLY")
        df_ramp = df_inv[df_inv["black_start_stage"] == 2].copy()
        if len(df_ramp) > 0:
            fig2 = go.Figure()
            for inv_id in df_ramp["inverter_id"].unique():
                sub = df_ramp[df_ramp["inverter_id"] == inv_id].reset_index(drop=True)
                fig2.add_trace(go.Scatter(
                    x=sub.index * 5, y=sub["black_start_voltage_pct"],
                    name=inv_id, mode="lines+markers",
                    marker=dict(size=4), line=dict(width=2)
                ))
            fig2.add_hline(y=100, line_dash="dash",
                           line_color="rgba(255,255,255,0.3)",
                           annotation_text="480V Nominal")
            fig2.update_layout(**PLOTLY_THEME, height=320,
                               xaxis_title="Time (minutes)",
                               yaxis_title="Bus Voltage (% nominal)")
        else:
            fig2 = go.Figure()
            fig2.add_annotation(text="No voltage ramp data yet.<br>Increase Black-Start Prob.",
                                x=0.5, y=0.5, showarrow=False,
                                font=dict(color=TEXT_MUTED, size=13))
            fig2.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    section("📦", "Load Reconnection Progress", "STAGE 3 — CRITICAL LOAD PICKUP")
    df_lp = df_inv[df_inv["black_start_stage"] == 3].copy()
    if len(df_lp) > 0:
        fig3 = go.Figure()
        for inv_id in df_lp["inverter_id"].unique():
            sub = df_lp[df_lp["inverter_id"] == inv_id]
            fig3.add_trace(go.Scatter(
                x=sub["timestamp_utc"], y=sub["loads_reconnected"],
                name=inv_id, mode="lines+markers",
                line=dict(width=2), marker=dict(size=5)
            ))
        fig3.add_hline(y=8, line_dash="dash", line_color=ACCENT_GREEN,
                       annotation_text="All Critical Loads (8)")
        fig3.update_layout(**PLOTLY_THEME, height=240,
                           xaxis_title="Time", yaxis_title="Loads Reconnected")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No load-pickup events in current window. Increase history window or black-start probability.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HARMONICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "③ Harmonics":
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_PURPLE}">
            ③ Active Harmonic Compensation
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            GFM virtual impedance shaping eliminates 5th, 7th, 11th, 13th harmonics from datacenter SMPS loads.
            &nbsp;<b style="color:{ACCENT_PURPLE}">IEEE 519-2022: THD ≤ 5% at PCC.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    gfl_thd  = df_gfl["thd_percent"].mean()
    gfm_thd  = df_gfm["thd_percent"].mean()
    gfl_comp = (df_gfl["thd_percent"] <= IEEE519_THD_LIMIT_PCT).mean()
    gfm_comp = (df_gfm["thd_percent"] <= IEEE519_THD_LIMIT_PCT).mean()
    df_comp  = df_gfm[df_gfm["harmonic_compensation_active"]]
    reduction = 0.0
    if len(df_comp) > 0:
        reduction = ((df_comp["thd_before_compensation_pct"] - df_comp["thd_percent"]) /
                      df_comp["thd_before_compensation_pct"].clip(lower=0.01)).mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("GFL Mean THD", f"{gfl_thd:.2f}%", "no compensation", "red",
                 badge="Non-Compliant" if gfl_thd > IEEE519_THD_LIMIT_PCT else "OK",
                 badge_type="err" if gfl_thd > IEEE519_THD_LIMIT_PCT else "ok")
    with c2: kpi("GFM Mean THD", f"{gfm_thd:.2f}%", "after APF compensation", "",
                 badge="Compliant", badge_type="ok")
    with c3: kpi("THD Reduction", f"{reduction:.1%}", "GFM active filter", "purple",
                 badge="APF Active", badge_type="warn")
    with c4: kpi("IEEE 519 GFM", f"{gfm_comp:.1%}", "compliance rate", "",
                 badge="Compliant", badge_type="ok")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("📊", "THD Distribution — GFL vs GFM", "IEEE 519 COMPLIANCE")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_gfl["thd_percent"], name="GFL (no compensation)",
            marker_color=ACCENT_RED, opacity=0.75, nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=df_gfm["thd_percent"], name="GFM (with APF)",
            marker_color=ACCENT_GREEN, opacity=0.75, nbinsx=50,
        ))
        fig.add_vline(x=IEEE519_THD_LIMIT_PCT, line_dash="dash",
                      line_color="rgba(255,255,255,0.4)",
                      annotation_text="IEEE 519 (5%)", annotation_font_size=9)
        fig.update_layout(barmode="overlay", **PLOTLY_THEME, height=320,
                          xaxis_title="THD (%)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("🔬", "Before vs After Compensation", "GFM ACTIVE POWER FILTER")
        if len(df_comp) > 0:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_comp["thd_before_compensation_pct"],
                y=df_comp["thd_percent"],
                mode="markers",
                marker=dict(
                    color=df_comp["harmonic_injection_a"],
                    colorscale=[[0, ACCENT_PURPLE], [1, ACCENT_CYAN]],
                    size=5, opacity=0.5,
                    colorbar=dict(title="Injection (A)", thickness=10),
                ),
                name="THD before→after",
            ))
            max_v = df_comp["thd_before_compensation_pct"].max()
            fig2.add_trace(go.Scatter(
                x=[0, max_v], y=[0, max_v], mode="lines",
                line=dict(dash="dash", color="rgba(255,255,255,0.2)"),
                name="No compensation", showlegend=True,
            ))
            fig2.add_hline(y=IEEE519_THD_LIMIT_PCT, line_dash="dot",
                           line_color=ACCENT_GREEN,
                           annotation_text="IEEE 519 limit")
            fig2.update_layout(**PLOTLY_THEME, height=320,
                               xaxis_title="THD Before (%)", yaxis_title="THD After (%)")
        else:
            fig2 = go.Figure()
            fig2.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    section("🎸", "Dominant Harmonic Order & Injection Current", "5th / 7th / 11th / 13th")
    col3, col4 = st.columns(2)
    with col3:
        harm_cnt = df_inv["dominant_harmonic_order"].value_counts().sort_index().reset_index()
        harm_cnt.columns = ["order", "count"]
        harm_colors = {5: ACCENT_RED, 7: ACCENT_AMBER, 11: ACCENT_PURPLE, 13: ACCENT_CYAN}
        fig3 = go.Figure(go.Bar(
            x=[f"{o}th harmonic" for o in harm_cnt["order"]],
            y=harm_cnt["count"],
            marker_color=[harm_colors.get(o, TEXT_MUTED) for o in harm_cnt["order"]],
            marker_line=dict(width=0),
        ))
        fig3.update_layout(**PLOTLY_THEME, height=240, yaxis_title="Occurrences")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        if len(df_comp) > 0:
            fig4 = go.Figure(go.Histogram(
                x=df_comp["harmonic_injection_a"],
                marker_color=ACCENT_PURPLE, nbinsx=40
            ))
            fig4.update_layout(**PLOTLY_THEME, height=240,
                               xaxis_title="Injection Current (A)", yaxis_title="Count")
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DROOP CONTROL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "④ Droop Control":
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_GREEN}">
            ④ Droop Control — Autonomous Power Sharing
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            Multiple GFM inverters share load without communication:
            &nbsp;<b style="color:{ACCENT_GREEN}">f = f₀ − kp·(P − P₀) &nbsp;|&nbsp; V = V₀ − kq·(Q − Q₀)</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sharing_err = df_gfm.groupby("timestamp_utc")["output_active_power_kw"].agg(
        lambda x: x.std() / x.mean() * 100 if x.mean() != 0 else 0
    )
    mean_err = sharing_err.mean()

    c1, c2, c3, c4 = st.columns(4)
    kp = df_gfm["droop_kw_per_hz"].iloc[0] if len(df_gfm) > 0 else 20.0
    kq = df_gfm["droop_kvar_per_v"].iloc[0] if len(df_gfm) > 0 else 5.0
    with c1: kpi("Droop kp", f"{kp:.0f}", "kW / Hz", badge="P/f", badge_type="ok")
    with c2: kpi("Droop kq", f"{kq:.0f}", "kVAr / V", badge="Q/V", badge_type="ok")
    with c3: kpi("Sharing Error", f"{mean_err:.1f}%", "std/mean of P",
                 badge="Good" if mean_err < 5 else "High",
                 badge_type="ok" if mean_err < 5 else "warn")
    with c4: kpi("GFM Inverters", str(df_gfm["inverter_id"].nunique()), "parallel units", "purple")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("📈", "P/f Droop Characteristic", "ACTIVE POWER vs FREQUENCY")
        fig = go.Figure()
        palette = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_PURPLE,
                   ACCENT_RED, "#FF9FF3", "#54A0FF", "#5F27CD"]
        for j, inv_id in enumerate(df_gfm["inverter_id"].unique()):
            sub = df_gfm[df_gfm["inverter_id"] == inv_id]
            sub = sub.sample(min(300, len(sub)))
            fig.add_trace(go.Scatter(
                x=sub["output_active_power_kw"], y=sub["output_frequency_hz"],
                mode="markers", name=inv_id,
                marker=dict(color=palette[j % len(palette)], size=4, opacity=0.5)
            ))
        fig.add_hline(y=NOMINAL_FREQ_HZ, line_dash="dash",
                      line_color="rgba(255,255,255,0.2)", annotation_text="f₀ = 60 Hz")
        fig.update_layout(**PLOTLY_THEME, height=320,
                          xaxis_title="Active Power (kW)", yaxis_title="Frequency (Hz)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("📉", "Q/V Droop Characteristic", "REACTIVE POWER vs VOLTAGE")
        fig2 = go.Figure()
        for j, inv_id in enumerate(df_gfm["inverter_id"].unique()):
            sub = df_gfm[df_gfm["inverter_id"] == inv_id]
            sub = sub.sample(min(300, len(sub)))
            fig2.add_trace(go.Scatter(
                x=sub["output_reactive_power_kvar"], y=sub["voltage_deviation_pu"],
                mode="markers", name=inv_id,
                marker=dict(color=palette[j % len(palette)], size=4, opacity=0.5)
            ))
        fig2.add_hline(y=0, line_dash="dash",
                       line_color="rgba(255,255,255,0.2)", annotation_text="V₀ nominal")
        fig2.update_layout(**PLOTLY_THEME, height=320,
                           xaxis_title="Reactive Power (kVAr)", yaxis_title="Voltage Deviation (p.u.)")
        st.plotly_chart(fig2, use_container_width=True)

    section("⚖️", "Load Sharing Error Over Time", "TARGET < 5%")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=sharing_err.index, y=sharing_err.values,
        fill="tozeroy", name="Sharing Error (%)",
        line=dict(color=ACCENT_AMBER, width=1.5),
        fillcolor="rgba(255,176,32,0.08)"
    ))
    fig3.add_hline(y=5.0, line_dash="dash", line_color=ACCENT_RED,
                   annotation_text="5% target")
    fig3.update_layout(**PLOTLY_THEME, height=240,
                       xaxis_title="Time", yaxis_title="Sharing Error (%)")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — WEAK-GRID STABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⑤ Weak-Grid Stability":
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{ACCENT_RED}">
            ⑤ Weak-Grid Stability Map
        </div>
        <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:4px">
            At SCR &lt; 3, GFL PLL destabilises. GFM remains stable for all SCR values.
            &nbsp;<b style="color:{ACCENT_RED}">Middlebrook: |Z_grid / Z_inv| &gt; 1 → unstable.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    gfl_stable = (df_gfl["stability_flag"] == "stable").mean()
    gfm_stable = (df_gfm["stability_flag"] == "stable").mean()
    gfl_scr    = df_gfl["scr"].mean()
    gfm_scr    = df_gfm["scr"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("GFL Stable", f"{gfl_stable:.1%}", "of operating time", "red",
                 badge="Unstable Risk", badge_type="err")
    with c2: kpi("GFM Stable", f"{gfm_stable:.1%}", "of operating time", "",
                 badge="Always Stable", badge_type="ok")
    with c3: kpi("GFL Mean SCR", f"{gfl_scr:.2f}", "Short-Circuit Ratio", "red")
    with c4: kpi("GFM Mean SCR", f"{gfm_scr:.2f}", "Short-Circuit Ratio")

    st.markdown("<br>", unsafe_allow_html=True)

    # Theoretical stability map
    section("🗺️", "Stability Map — SCR × Phase Margin", "MIDDLEBROOK CRITERION")
    scr_sweep = np.linspace(0.5, 12.0, 400)

    def gfl_pm(s): return float(np.clip(90.0 * (1.0 - np.exp(-0.3 * (s - 1.0))), -30, 90))
    def gfm_margin(s):
        Zg = 1.0 / max(s, 0.01)
        Zi = 0.12
        return float(-20.0 * np.log10(Zg / Zi))

    gfl_pm_vals  = [gfl_pm(s) for s in scr_sweep]
    gfm_imp_vals = [gfm_margin(s) for s in scr_sweep]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["PLL Phase Margin vs SCR (GFL)",
                                        "Impedance Stability Margin — Middlebrook (dB)"])

    # Zone fills
    zones = [(0.5, 1.5, ACCENT_RED, 0.15), (1.5, 3.0, ACCENT_AMBER, 0.12), (3.0, 12.0, ACCENT_GREEN, 0.08)]
    for xmin, xmax, color, alpha in zones:
        mask = [(xmin <= s < xmax) for s in scr_sweep]
        xs   = [s for s, m in zip(scr_sweep, mask) if m]
        ys_t = [gfl_pm(s) for s in xs]
        rgb  = px.colors.hex_to_rgb(color)
        fill = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"
        if xs:
            fig.add_trace(go.Scatter(
                x=xs + xs[::-1], y=ys_t + [-30] * len(xs),
                fill="toself", fillcolor=fill,
                line=dict(width=0), showlegend=False, mode="lines"
            ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=scr_sweep, y=gfl_pm_vals, name="GFL PLL Phase Margin (°)",
        line=dict(color=ACCENT_RED, width=2.5)
    ), row=1, col=1)
    fig.add_hline(y=45, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text="PM=45° target", row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color=ACCENT_RED,
                  annotation_text="PM=0° (unstable)", row=1, col=1)

    # GFM stability band
    fig.add_trace(go.Scatter(
        x=[0.5, 12.0], y=[80, 80], name="GFM (stable for all SCR)",
        line=dict(color=ACCENT_GREEN, width=2.5, dash="dot")
    ), row=1, col=1)

    # Simulation scatter — GFL
    fig.add_trace(go.Scatter(
        x=df_gfl["scr"], y=df_gfl["pll_stability_margin_deg"],
        mode="markers", name="GFL (simulated)",
        marker=dict(color=ACCENT_RED, size=4, opacity=0.25)
    ), row=1, col=1)

    # Right panel: impedance margin
    fig.add_trace(go.Scatter(
        x=scr_sweep, y=gfm_imp_vals, name="GFM |Zgrid/Zinv| (dB)",
        fill="tozeroy", fillcolor="rgba(0,255,159,0.08)",
        line=dict(color=ACCENT_GREEN, width=2.5)
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color=ACCENT_RED,
                  annotation_text="Middlebrook limit (0 dB)", row=1, col=2)

    fig.add_trace(go.Scatter(
        x=df_gfl["scr"], y=df_gfl["impedance_stability_margin_db"],
        mode="markers", name="GFL (simulated)",
        marker=dict(color=ACCENT_RED, size=4, opacity=0.25),
        showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=df_gfm["scr"], y=df_gfm["impedance_stability_margin_db"],
        mode="markers", name="GFM (simulated)",
        marker=dict(color=ACCENT_GREEN, size=4, opacity=0.25),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(height=420, **PLOTLY_THEME)
    fig.update_xaxes(title_text="Short-Circuit Ratio (SCR)", gridcolor=CARD_BORDER)
    fig.update_yaxes(title_text="Phase Margin (°)", row=1, col=1, gridcolor=CARD_BORDER)
    fig.update_yaxes(title_text="Stability Margin (dB)", row=1, col=2, gridcolor=CARD_BORDER)
    st.plotly_chart(fig, use_container_width=True)

    # Stability breakdown per inverter
    section("📋", "Stability Breakdown per Inverter", "SIMULATED DATA")
    stab_df = (df_inv.groupby(["inverter_id", "control_mode", "stability_flag"])
               .size().reset_index(name="count"))
    stab_pct = stab_df.copy()
    totals = stab_pct.groupby("inverter_id")["count"].transform("sum")
    stab_pct["pct"] = (stab_pct["count"] / totals * 100).round(1)
    sc = {f: c for f, c in
          [("stable", ACCENT_GREEN), ("marginal", ACCENT_AMBER), ("unstable", ACCENT_RED)]}
    fig2 = go.Figure()
    for flag in ["stable", "marginal", "unstable"]:
        sub = stab_pct[stab_pct["stability_flag"] == flag]
        if len(sub):
            fig2.add_trace(go.Bar(
                name=flag.capitalize(), x=sub["inverter_id"], y=sub["pct"],
                marker_color=sc[flag], marker_line=dict(width=0),
                text=sub["pct"].apply(lambda v: f"{v:.0f}%"),
                textposition="inside",
            ))
    fig2.update_layout(barmode="stack", **PLOTLY_THEME, height=240,
                       xaxis_title="Inverter", yaxis_title="% of Time")
    st.plotly_chart(fig2, use_container_width=True)
