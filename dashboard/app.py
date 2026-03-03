import sys, os
sys.path.insert(0, os.path.abspath('..'))

# Ensure project root is in path (for Streamlit Cloud)
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

from data_generator.server_simulator import ServerSimulator
from data_generator.ups_inverter_simulator import UPSSimulator, InverterSimulator


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Datacenter Energy Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #3498db;
    }
    .metric-card.warning { border-left-color: #f39c12; }
    .metric-card.danger  { border-left-color: #e74c3c; }
    .metric-card.success { border-left-color: #2ecc71; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/server.png", width=60)
    st.title("⚡ DC Energy Platform")
    st.markdown("---")

    st.subheader("Simulation Parameters")
    num_servers    = st.slider("Servers",         50, 200, 100, step=10)
    num_racks      = st.slider("Racks",            5,  20,  10)
    fault_prob     = st.slider("Fault Probability", 0.0, 0.10, 0.01, step=0.005)
    islanding_prob = st.slider("Islanding Probability", 0.0, 0.15, 0.05, step=0.01)
    history_hours  = st.slider("History Window (hours)", 1, 24, 6)

    st.markdown("---")
    st.subheader("Navigation")
    page = st.radio("Page", [
        "🏠 Overview",
        "🖥️ Server Telemetry",
        "⚡ Inverter Analysis",
        "🔋 UPS Monitor",
        "📈 PUE Trends",
    ])

    st.markdown("---")
    refresh = st.button("🔄 Refresh Data", use_container_width=True)
    st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")


# ── Data generation (cached) ──────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner="Generating telemetry...")
def load_data(num_servers, num_racks, fault_prob, islanding_prob, history_hours, _refresh):
    srv_sim = ServerSimulator(num_servers=num_servers, num_racks=num_racks,
                               fault_probability=fault_prob, random_seed=None)
    ups_sim = UPSSimulator(num_ups=4)
    inv_sim = InverterSimulator(num_inverters=4,
                                 islanding_probability=islanding_prob)

    start  = datetime.now(timezone.utc) - timedelta(hours=history_hours)
    steps  = history_hours * 12   # 5-min intervals
    srv_records, ups_records, inv_records = [], [], []

    for i in range(steps):
        ts = start + timedelta(minutes=5 * i)
        srv_records.extend([asdict(r) for r in srv_sim.generate_snapshot(ts)])
        ups_records.extend([asdict(r) for r in ups_sim.generate_snapshot(ts)])
        inv_records.extend([asdict(r) for r in inv_sim.generate_snapshot(ts)])

    df_srv = pd.DataFrame(srv_records)
    df_ups = pd.DataFrame(ups_records)
    df_inv = pd.DataFrame(inv_records)

    for df in [df_srv, df_ups, df_inv]:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)

    return df_srv, df_ups, df_inv

df_srv, df_ups, df_inv = load_data(
    num_servers, num_racks, fault_prob, islanding_prob,
    history_hours, refresh
)

# Latest snapshot
latest_ts  = df_srv['timestamp_utc'].max()
df_latest  = df_srv[df_srv['timestamp_utc'] == latest_ts]
df_inv_lat = df_inv[df_inv['timestamp_utc'] == df_inv['timestamp_utc'].max()]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("⚡ Datacenter Energy Intelligence Platform")
    st.caption("Real-time monitoring & ML-powered analytics")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    total_power = df_latest['power_draw_w'].sum() / 1000
    avg_pue     = df_latest['pue_contribution'].mean()
    anomaly_ct  = df_latest['is_anomaly'].sum()
    avg_cpu     = df_latest['cpu_utilization'].mean()
    avg_temp    = df_latest['cpu_temp_c'].mean()
    critical_ct = (df_latest['cooling_state'] == 'critical').sum()

    col1.metric("Total IT Power",  f"{total_power:.1f} kW")
    col2.metric("Average PUE",     f"{avg_pue:.4f}")
    col3.metric("Active Servers",  f"{len(df_latest)}")
    col4.metric("Avg CPU Util",    f"{avg_cpu:.1%}")
    col5.metric("Avg CPU Temp",    f"{avg_temp:.1f} °C")
    col6.metric("⚠️ Anomalies",    f"{anomaly_ct}",
                delta=f"{anomaly_ct} alerts",
                delta_color="inverse" if anomaly_ct > 0 else "normal")

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # Power over time
        dc_power = (df_srv.groupby('timestamp_utc')['power_draw_w']
                    .sum().reset_index())
        dc_power['power_kw'] = dc_power['power_draw_w'] / 1000
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dc_power['timestamp_utc'], y=dc_power['power_kw'],
            fill='tozeroy', name='Total IT Power',
            line=dict(color='#e74c3c', width=2)))
        fig.update_layout(
            title='Total Datacenter Power Consumption (kW)',
            template='plotly_dark', height=320,
            margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Cooling state donut
        cooling_counts = df_latest['cooling_state'].value_counts().reset_index()
        cooling_counts.columns = ['state', 'count']
        colors = {'normal': '#2ecc71', 'elevated': '#f39c12',
                  'warning': '#e67e22', 'critical': '#e74c3c'}
        fig2 = px.pie(cooling_counts, values='count', names='state',
                      hole=0.55, title='Cooling State Distribution',
                      color='state', color_discrete_map=colors)
        fig2.update_layout(template='plotly_dark', height=320,
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Rack heatmap
    st.subheader("🌡️ Rack Thermal Heatmap")
    rack_pivot = (df_latest.groupby('rack_id')['cpu_temp_c']
                  .mean().reset_index().sort_values('rack_id'))
    fig3 = px.bar(rack_pivot, x='rack_id', y='cpu_temp_c',
                  color='cpu_temp_c', color_continuous_scale='YlOrRd',
                  title='Average CPU Temperature per Rack (Latest Snapshot)',
                  labels={'cpu_temp_c': 'Avg CPU Temp (°C)', 'rack_id': 'Rack'})
    fig3.update_layout(template='plotly_dark', height=300,
                       margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SERVER TELEMETRY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🖥️ Server Telemetry":
    st.title("🖥️ Server Telemetry")

    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly Rate", f"{df_latest['is_anomaly'].mean():.2%}")
    col2.metric("Avg Power",    f"{df_latest['power_draw_w'].mean():.1f} W")
    col3.metric("Max CPU Temp", f"{df_latest['cpu_temp_c'].max():.1f} °C")

    # CPU Util vs Power scatter
    fig = px.scatter(
        df_latest.sample(min(500, len(df_latest))),
        x='cpu_utilization', y='power_draw_w',
        color='is_anomaly',
        color_discrete_map={False: '#3498db', True: '#e74c3c'},
        size='cpu_temp_c', hover_data=['server_id', 'rack_id', 'cooling_state'],
        title='CPU Utilization vs Power Draw — Anomalies Highlighted',
        labels={'cpu_utilization': 'CPU Utilization',
                'power_draw_w': 'Power Draw (W)'},
        template='plotly_dark'
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Anomalies table
    st.subheader("🚨 Active Anomalies")
    anomalies = df_latest[df_latest['is_anomaly']].sort_values(
        'cpu_temp_c', ascending=False
    )[['server_id', 'rack_id', 'cpu_temp_c', 'power_draw_w',
       'cpu_utilization', 'cooling_state', 'fault_type']].head(20)

    if len(anomalies) > 0:
        st.dataframe(anomalies.style.background_gradient(
            subset=['cpu_temp_c'], cmap='Reds'), use_container_width=True)
    else:
        st.success("✅ No anomalies detected in current snapshot.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INVERTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Inverter Analysis":
    st.title("⚡ GFL vs GFM Inverter Analysis")
    st.caption("Core research topic — Grid-Following vs Grid-Forming dynamics")

    # Mode distribution
    mode_counts = df_inv_lat['control_mode'].value_counts().reset_index()
    mode_counts.columns = ['mode', 'count']
    col1, col2, col3 = st.columns(3)
    for mode, color, col in [('GFL','#e74c3c',col1),
                               ('GFM','#2ecc71',col2),
                               ('transitioning','#f39c12',col3)]:
        n = mode_counts[mode_counts['mode']==mode]['count'].sum()
        col.metric(f"{mode} Inverters", f"{n}")

    # ROCOF and THD over time
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=['ROCOF (Hz/s) — IEEE 1547 limit: 0.5',
                        'THD (%) — IEEE 519 limit: 5%'],
        vertical_spacing=0.1)

    colors = {'GFL': '#e74c3c', 'GFM': '#2ecc71', 'transitioning': '#f39c12'}
    for inv_id in df_inv['inverter_id'].unique():
        df_i = df_inv[df_inv['inverter_id'] == inv_id]
        mode = df_i['control_mode'].mode()[0]
        fig.add_trace(go.Scatter(
            x=df_i['timestamp_utc'], y=df_i['rocof_hz_per_s'],
            name=f"{inv_id} ({mode})",
            line=dict(color=colors.get(mode, 'gray'), width=1.5)),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_i['timestamp_utc'], y=df_i['thd_percent'],
            name=f"{inv_id}", showlegend=False,
            line=dict(color=colors.get(mode, 'gray'), width=1.5)),
            row=2, col=1)

    fig.add_hline(y=0.5, line_dash='dash', line_color='white',
                  annotation_text='IEEE 1547 (0.5 Hz/s)', row=1, col=1)
    fig.add_hline(y=5.0, line_dash='dash', line_color='white',
                  annotation_text='IEEE 519 (5%)', row=2, col=1)
    fig.update_layout(height=550, template='plotly_dark',
                      title='Inverter Power Quality Metrics Over Time')
    st.plotly_chart(fig, use_container_width=True)

    # GFL vs GFM comparison box
    df_gfx = df_inv[df_inv['control_mode'].isin(['GFL', 'GFM'])]
    fig2 = px.box(df_gfx, x='control_mode', y='rocof_hz_per_s',
                  color='control_mode',
                  color_discrete_map={'GFL': '#e74c3c', 'GFM': '#2ecc71'},
                  points='outliers', title='ROCOF Distribution — GFL vs GFM',
                  template='plotly_dark')
    fig2.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Islanding events table
    islanding = df_inv[df_inv['islanding_detected']].sort_values(
        'timestamp_utc', ascending=False
    )[['timestamp_utc', 'inverter_id', 'control_mode',
       'rocof_hz_per_s', 'freq_deviation_hz', 'thd_percent']].head(15)

    st.subheader(f"🔌 Islanding Events ({len(islanding)} detected)")
    if len(islanding) > 0:
        st.dataframe(islanding, use_container_width=True)
    else:
        st.info("No islanding events in current window.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPS MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔋 UPS Monitor":
    st.title("🔋 UPS Monitor")

    col1, col2, col3 = st.columns(3)
    avg_soc  = df_ups[df_ups['timestamp_utc'] == df_ups['timestamp_utc'].max()]['battery_soc'].mean()
    on_batt  = (df_ups['ups_mode'] == 'battery').sum()
    faults   = (df_ups['ups_mode'] == 'fault').sum()
    col1.metric("Avg Battery SoC", f"{avg_soc:.1%}")
    col2.metric("On-Battery Events", f"{on_batt}")
    col3.metric("Fault Events", f"{faults}", delta_color="inverse")

    fig = px.line(df_ups, x='timestamp_utc', y='battery_soc',
                  color='ups_id', title='Battery State of Charge — All UPS Units',
                  labels={'battery_soc': 'SoC', 'timestamp_utc': 'Time'},
                  template='plotly_dark')
    fig.add_hline(y=0.20, line_dash='dash', line_color='red',
                  annotation_text='Low Battery Alarm (20%)')
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PUE TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 PUE Trends":
    st.title("📈 PUE Trends & Energy Efficiency")

    pue_ts = (df_srv.groupby('timestamp_utc')['pue_contribution']
              .mean().reset_index())
    pue_ts.columns = ['timestamp_utc', 'avg_pue']

    current_pue = pue_ts['avg_pue'].iloc[-1]
    min_pue     = pue_ts['avg_pue'].min()
    max_pue     = pue_ts['avg_pue'].max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Current PUE",  f"{current_pue:.4f}")
    col2.metric("Best PUE",     f"{min_pue:.4f}")
    col3.metric("Worst PUE",    f"{max_pue:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pue_ts['timestamp_utc'], y=pue_ts['avg_pue'],
        fill='tozeroy', name='PUE',
        line=dict(color='#2ecc71', width=2)))
    fig.add_hline(y=pue_ts['avg_pue'].mean(), line_dash='dash',
                  line_color='gray',
                  annotation_text=f"Mean: {pue_ts['avg_pue'].mean():.4f}")
    fig.update_layout(title='PUE Over Time',
                      xaxis_title='Time', yaxis_title='PUE',
                      template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # PUE by hour of day
    df_srv['hour'] = df_srv['timestamp_utc'].dt.hour
    pue_hour = df_srv.groupby('hour')['pue_contribution'].mean().reset_index()
    fig2 = px.bar(pue_hour, x='hour', y='pue_contribution',
                  color='pue_contribution', color_continuous_scale='RdYlGn_r',
                  title='Average PUE by Hour of Day',
                  labels={'pue_contribution': 'Avg PUE', 'hour': 'Hour (UTC)'},
                  template='plotly_dark')
    fig2.update_layout(height=340)
    st.plotly_chart(fig2, use_container_width=True)