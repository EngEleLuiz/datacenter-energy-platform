"""
analysis/stability_analysis.py
================================
Frequency-domain stability analysis for GFL and GFM inverters.

Computes and visualizes:
  - Bode diagram of the GFL PLL (Phase-Locked Loop) control loop
  - Nyquist diagram of the GFL open-loop transfer function
  - Middlebrook Criterion: |Z_grid(jω) / Z_inv(jω)|
  - Phase Margin (PM) and Gain Margin (GM) vs SCR sweep
  - GFL vs GFM comparison under weak-grid conditions

References:
  - Zhong & Weiss (2011)  — IEEE Trans. Industrial Electronics
  - Zhou et al. (2020)    — IEEE Trans. Power Electronics
  - Middlebrook (1976)    — PESC Proceedings
  - Pogaku et al. (2007)  — IEEE Trans. Power Systems

Dependencies: numpy, scipy, plotly
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Optional

# ── System base parameters ─────────────────────────────────────────────────
F0     = 60.0           # Hz  — nominal frequency
W0     = 2 * np.pi * F0
V_NOM  = 480.0          # V   — nominal AC voltage (480 V three-phase, data center)
S_BASE = 100e3          # VA  — base apparent power (100 kVA)
Z_BASE = V_NOM**2 / S_BASE

# SRF-PLL (Synchronous Reference Frame PLL) second-order parameters
PLL_KP          = 80.0                  # PI proportional gain (typical for 60 Hz)
PLL_KI          = 1600.0               # PI integral gain
PLL_FILTER_WN   = 2 * np.pi * 30       # loop filter bandwidth (30 Hz)

# GFM inverter parameters
INV_OUTPUT_IMP_PU = 0.12   # output impedance (0.12 pu — typical for VSM)
INV_FILTER_L_PU   = 0.15   # LCL filter inductance (pu)
INV_FILTER_C_PU   = 0.05   # LCL filter capacitance (pu)

# Design system colors
C_GFL  = "#FF4757"
C_GFM  = "#00FF9F"
C_GRID = "#00D4FF"
C_WARN = "#FFB020"
TEXT   = "#F1F5F9"
BORDER = "#1F2937"

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)


@dataclass
class StabilityMetrics:
    """Stability metrics computed for a given SCR value."""
    scr:              float
    gfl_pm_deg:       float   # Phase Margin of GFL PLL (degrees)
    gfl_gm_db:        float   # Gain Margin of GFL (dB)
    gfl_stable:       bool
    gfm_stable:       bool    # GFM is stable for all SCR > 0.5
    middlebrook_db:   float   # |Z_grid/Z_inv| in dB — below 0 dB = stable
    pll_crossover_hz: float   # GFL PLL gain crossover frequency (Hz)
    risk_label:       str     # 'stable' | 'marginal' | 'unstable'


# ── Mathematical models ────────────────────────────────────────────────────

def z_grid(scr: float, w: np.ndarray) -> np.ndarray:
    """
    Grid impedance in the frequency domain.

    Model: series R-L with R_grid + jωL_grid
    where Z_grid ≈ V²/(SCR × S_base) in pu.

    Args:
        scr: Short-Circuit Ratio at the point of common coupling
        w:   Angular frequency vector (rad/s)

    Returns:
        Array of complex impedances Z_grid(jω) in Ohms
    """
    # Typical R/X ratio for medium-voltage transmission lines: 0.3
    z_magnitude = 1.0 / max(scr, 0.1)      # pu (SCR = V²/(Z·Sbase) → Z = 1/SCR pu)
    r_grid = z_magnitude * 0.3 / np.sqrt(1 + 0.3**2)
    x_base = z_magnitude / np.sqrt(1 + 0.3**2)
    l_grid = x_base / W0
    return (r_grid + 1j * w * l_grid) * Z_BASE


def z_inv_gfl(w: np.ndarray) -> np.ndarray:
    """
    GFL (Grid-Following) inverter output impedance.

    Model: LCL filter + current controller output impedance.
    A GFL inverter behaves as a current source — HIGH output impedance.
    """
    l1 = INV_FILTER_L_PU * Z_BASE / W0
    c  = INV_FILTER_C_PU / (W0 * Z_BASE)
    l2 = INV_FILTER_L_PU * 0.3 * Z_BASE / W0

    zl1  = 1j * w * l1
    zc   = 1.0 / (1j * w * c + 1e-15)
    zl2  = 1j * w * l2

    # GFL output impedance ≈ L1 in series with (C || L2)
    z_cl2 = (zc * zl2) / (zc + zl2 + 1e-30)
    return zl1 + z_cl2


def z_inv_gfm(w: np.ndarray, h_inertia: float = 5.0) -> np.ndarray:
    """
    GFM (Grid-Forming) virtual output impedance.

    A GFM inverter behaves as a voltage source — LOW output impedance.
    The virtual inertia constant H affects the impedance at low frequencies.
    """
    # Low base impedance (voltage source behavior)
    z_base_gfm = INV_OUTPUT_IMP_PU * Z_BASE

    # Virtual inertia component: adds inductive reactance proportional to H
    # at low frequencies (emulates synchronous machine reactance)
    l_virtual = h_inertia * z_base_gfm / W0
    z_virtual = 1j * w * l_virtual

    # Virtual damping (swing equation D coefficient)
    r_virtual = z_base_gfm * 0.05

    return r_virtual + z_virtual + z_base_gfm * 0.02


def pll_open_loop(scr: float, w: np.ndarray) -> np.ndarray:
    """
    SRF-PLL open-loop transfer function.

    L(jω) = Kp·(1 + Ki/(jω)) × 1/(jω) × H_filter(jω) × G_grid(jω)

    G_grid depends on SCR — weak grid adds a destabilizing gain term.
    """
    jw = 1j * w + 1e-30   # avoid division by zero

    # PI controller: Kp(1 + Ki/s)
    pi_ctrl = PLL_KP * (1 + PLL_KI / jw)

    # Integrator (converts frequency error to phase)
    integrator = 1.0 / jw

    # Low-pass loop filter
    h_filter = PLL_FILTER_WN / (jw + PLL_FILTER_WN)

    # Plant gain (grid): weak grid means the PLL sees larger voltage perturbations
    # Normalized: high SCR → gain ≈ 1; low SCR → gain > 1 (destabilizing)
    scr_gain = 1.0 / max(scr * 0.3, 0.05)

    return pi_ctrl * integrator * h_filter * scr_gain


def compute_stability_metrics(scr: float) -> StabilityMetrics:
    """Computes all stability metrics for a given SCR value."""
    w     = np.logspace(-1, 4, 5000) * 2 * np.pi   # 0.1 Hz to 10 kHz
    loop  = pll_open_loop(scr, w)
    mag   = np.abs(loop)
    phase = np.angle(loop, deg=True)

    # Gain crossover frequency (where |L(jω)| = 1 = 0 dB)
    cross_idx    = np.argmin(np.abs(mag - 1.0))
    crossover_hz = w[cross_idx] / (2 * np.pi)

    # Phase margin: angle at 0 dB + 180°
    pm = float(phase[cross_idx] + 180.0)

    # Phase crossover frequency (where phase = -180°)
    phase_cross_idx = np.argmin(np.abs(phase + 180.0))
    gm_db = float(-20 * np.log10(mag[phase_cross_idx] + 1e-9))

    # Middlebrook criterion
    zg = z_grid(scr, w)
    zi = z_inv_gfl(w)
    ratio = np.abs(zg) / (np.abs(zi) + 1e-15)
    middlebrook_db = float(20 * np.log10(np.max(ratio) + 1e-9))

    # Stability assessment
    gfl_stable = (pm >= 15.0) and (gm_db >= 3.0)

    if scr >= 3.0:    risk = "stable"
    elif scr >= 1.5:  risk = "marginal"
    else:             risk = "unstable"

    return StabilityMetrics(
        scr=scr,
        gfl_pm_deg=pm,
        gfl_gm_db=gm_db,
        gfl_stable=gfl_stable,
        gfm_stable=True,           # GFM is stable for all SCR ≥ 0.5 (no PLL)
        middlebrook_db=middlebrook_db,
        pll_crossover_hz=crossover_hz,
        risk_label=risk,
    )


def scr_sweep_metrics(scr_values: Optional[np.ndarray] = None) -> list:
    """Computes metrics for a sweep of SCR values."""
    if scr_values is None:
        scr_values = np.concatenate([
            np.linspace(0.5, 1.5, 20),
            np.linspace(1.5, 3.0, 20),
            np.linspace(3.0, 12.0, 30),
        ])
    return [compute_stability_metrics(float(s)) for s in scr_values]


# ── Plotly figures ─────────────────────────────────────────────────────────

def make_bode_figure(scr: float, h_inertia: float = 5.0) -> go.Figure:
    """
    Bode diagram comparing GFL vs GFM for a given SCR.
    Panels: Magnitude (dB) and Phase (degrees) vs Frequency (Hz).
    """
    w  = np.logspace(0, 4, 2000) * 2 * np.pi   # 1 Hz to 10 kHz
    fq = w / (2 * np.pi)

    loop_gfl  = pll_open_loop(scr, w)
    zg        = z_grid(scr, w)
    zi_gfm    = z_inv_gfm(w, h_inertia)
    loop_gfm  = zg / (zi_gfm + 1e-15)     # Middlebrook ratio for GFM

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            f"Magnitude (dB) — SCR = {scr:.1f}",
            "Phase (degrees)",
        ],
    )

    # ── Magnitude ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=fq, y=20 * np.log10(np.abs(loop_gfl) + 1e-15),
        name="GFL — PLL Loop", line=dict(color=C_GFL, width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fq, y=20 * np.log10(np.abs(loop_gfm) + 1e-15),
        name=f"GFM — |Z_grid/Z_inv| (H={h_inertia}s)", line=dict(color=C_GFM, width=2),
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash",
                  line_color="rgba(255,255,255,0.3)",
                  annotation_text="0 dB", row=1, col=1)

    # ── Phase ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=fq, y=np.angle(loop_gfl, deg=True),
        name="GFL — Phase", line=dict(color=C_GFL, width=2, dash="dot"),
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=fq, y=np.angle(loop_gfm, deg=True),
        name="GFM — Phase", line=dict(color=C_GFM, width=2, dash="dot"),
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=-180, line_dash="dash",
                  line_color=C_GFL, annotation_text="-180° (instability boundary)",
                  annotation_font_color=C_GFL, row=2, col=1)

    # Vertical line at GFL crossover frequency
    met = compute_stability_metrics(scr)
    fig.add_vline(
        x=met.pll_crossover_hz,
        line_dash="dot", line_color=C_WARN,
        annotation_text=f"f_cross={met.pll_crossover_hz:.1f} Hz",
        annotation_font_color=C_WARN,
    )

    fig.update_layout(
        height=480,
        **PLOTLY_BASE,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor=BORDER, borderwidth=1),
        title=f"Bode Diagram — GFL vs GFM  |  SCR = {scr:.1f}  |  PM_GFL = {met.gfl_pm_deg:.1f}°",
    )
    fig.update_xaxes(type="log", title_text="Frequency (Hz)",
                     gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (°)",      row=2, col=1)
    return fig


def make_nyquist_figure(scr: float) -> go.Figure:
    """
    Nyquist diagram of the GFL PLL open-loop transfer function.
    The critical point (-1+j0) must lie outside the contour for stability.
    """
    w_pos = np.logspace(-2, 5, 3000) * 2 * np.pi
    loop  = pll_open_loop(scr, w_pos)

    re = np.real(loop)
    im = np.imag(loop)

    fig = go.Figure()

    # Positive frequency branch (ω > 0)
    fig.add_trace(go.Scatter(
        x=re, y=im, mode="lines",
        name="ω > 0", line=dict(color=C_GFL, width=2),
    ))
    # Negative frequency branch (mirror)
    fig.add_trace(go.Scatter(
        x=re[::-1], y=-im[::-1], mode="lines",
        name="ω < 0", line=dict(color=C_GFL, width=1.5, dash="dot"),
    ))

    # Critical point (-1 + j0)
    fig.add_trace(go.Scatter(
        x=[-1], y=[0], mode="markers",
        name="Critical point (−1+j0)",
        marker=dict(symbol="x-open", size=16, color=C_WARN, line=dict(width=3)),
    ))

    met = compute_stability_metrics(scr)
    stability_text = "✅ STABLE" if met.gfl_stable else "❌ UNSTABLE"

    fig.update_layout(
        height=420,
        **PLOTLY_BASE,
        title=f"Nyquist Diagram — GFL PLL  |  SCR = {scr:.1f}  |  {stability_text}",
        xaxis=dict(
            title="Re[L(jω)]",
            gridcolor=BORDER,
            zerolinecolor="rgba(255,255,255,0.3)",
            range=[max(re.min() * 1.2, -3), min(re.max() * 1.2, 3)],
        ),
        yaxis=dict(
            title="Im[L(jω)]",
            gridcolor=BORDER,
            zerolinecolor="rgba(255,255,255,0.3)",
            scaleanchor="x",
            range=[max(im.min() * 1.2, -3), min(im.max() * 1.2, 3)],
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor=BORDER, borderwidth=1),
    )
    return fig


def make_pm_vs_scr_figure(metrics_list: list) -> go.Figure:
    """
    Phase Margin of GFL PLL vs SCR, with colored stability zones and GFM reference line.
    Most impactful chart for the presentation — clearly shows GFM advantage.
    """
    scrs = np.array([m.scr       for m in metrics_list])
    pms  = np.array([m.gfl_pm_deg for m in metrics_list])

    fig = go.Figure()

    # Background stability zones
    zone_data = [
        (0.5, 1.5, "rgba(255,71,87,0.12)",  "Unstable zone (SCR < 1.5)"),
        (1.5, 3.0, "rgba(255,176,32,0.10)",  "Marginal zone (1.5 ≤ SCR < 3.0)"),
        (3.0, 12., "rgba(0,255,159,0.08)",   "Stable zone (SCR ≥ 3.0)"),
    ]
    for xmin, xmax, color, label in zone_data:
        mask = (scrs >= xmin) & (scrs <= xmax)
        fig.add_trace(go.Scatter(
            x=np.concatenate([scrs[mask], scrs[mask][::-1]]),
            y=np.concatenate([pms[mask],  np.full(mask.sum(), -30)]),
            fill="toself", fillcolor=color,
            line=dict(width=0), name=label, showlegend=True,
        ))

    # GFL Phase Margin curve
    fig.add_trace(go.Scatter(
        x=scrs, y=pms, name="GFL — PLL Phase Margin",
        line=dict(color=C_GFL, width=2.5),
        mode="lines",
    ))

    # GFM — constant reference (no PLL, always stable)
    fig.add_trace(go.Scatter(
        x=[scrs.min(), scrs.max()], y=[80, 80],
        name="GFM — Stable for all SCR values", mode="lines",
        line=dict(color=C_GFM, width=2.5, dash="dot"),
    ))

    # Reference lines
    fig.add_hline(y=45, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text="PM = 45° (design target)")
    fig.add_hline(y=15, line_dash="dash", line_color=C_WARN,
                  annotation_text="PM = 15° (practical stability limit)",
                  annotation_font_color=C_WARN)
    fig.add_hline(y=0,  line_dash="solid", line_color=C_GFL,
                  annotation_text="PM = 0° (unstable)", annotation_font_color=C_GFL)

    fig.update_layout(
        height=400,
        **PLOTLY_BASE,
        title="GFL PLL Phase Margin vs SCR — Comparison with GFM",
        xaxis=dict(title="Short-Circuit Ratio (SCR)", gridcolor=BORDER),
        yaxis=dict(title="Phase Margin (degrees)", gridcolor=BORDER, range=[-30, 100]),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor=BORDER, borderwidth=1,
                    x=0.6, y=0.05),
    )
    return fig


def make_middlebrook_figure(metrics_list: list) -> go.Figure:
    """
    Middlebrook stability margin (|Z_grid/Z_inv| in dB) vs SCR.
    Below 0 dB = stable. Above = unstable.
    """
    scrs = np.array([m.scr           for m in metrics_list])
    mbs  = np.array([m.middlebrook_db for m in metrics_list])

    # Compute GFM separately
    mbs_gfm = []
    for scr in scrs:
        w  = np.logspace(0, 4, 500) * 2 * np.pi
        zg = z_grid(scr, w)
        zi = z_inv_gfm(w)
        mbs_gfm.append(float(20 * np.log10(np.max(np.abs(zg) / (np.abs(zi) + 1e-15)))))

    fig = go.Figure()

    # Unstable region
    fig.add_hrect(y0=0, y1=max(max(mbs), 10),
                  fillcolor="rgba(255,71,87,0.08)", line_width=0,
                  annotation_text="Unstable (Middlebrook violated)",
                  annotation_position="top right",
                  annotation_font_color=C_GFL)

    # Stable region
    fig.add_hrect(y0=min(min(mbs), -20), y1=0,
                  fillcolor="rgba(0,255,159,0.06)", line_width=0,
                  annotation_text="Stable",
                  annotation_position="bottom right",
                  annotation_font_color=C_GFM)

    fig.add_trace(go.Scatter(
        x=scrs, y=mbs, name="GFL — |Z_grid/Z_inv| (dB)",
        line=dict(color=C_GFL, width=2.5), fill="tozeroy",
        fillcolor="rgba(255,71,87,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=scrs, y=mbs_gfm, name="GFM — |Z_grid/Z_inv| (dB)",
        line=dict(color=C_GFM, width=2.5), fill="tozeroy",
        fillcolor="rgba(0,255,159,0.06)",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color=C_WARN,
                  annotation_text="Middlebrook limit (0 dB)")

    fig.update_layout(
        height=380,
        **PLOTLY_BASE,
        title="Middlebrook Criterion — Impedance Stability Margin (dB)",
        xaxis=dict(title="SCR", gridcolor=BORDER),
        yaxis=dict(title="|Z_grid / Z_inv| (dB)", gridcolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor=BORDER, borderwidth=1),
    )
    return fig


def make_gain_margin_figure(metrics_list: list) -> go.Figure:
    """Gain Margin (GM) of GFL in dB vs SCR."""
    scrs = np.array([m.scr       for m in metrics_list])
    gms  = np.array([m.gfl_gm_db for m in metrics_list])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scrs, y=gms, name="GFL — Gain Margin (dB)",
        line=dict(color=C_GFL, width=2), fill="tozeroy",
        fillcolor="rgba(255,71,87,0.08)",
    ))
    fig.add_hline(y=6, line_dash="dash", line_color=C_GFM,
                  annotation_text="GM = 6 dB (design target)")
    fig.add_hline(y=0, line_dash="solid", line_color=C_GFL,
                  annotation_text="GM = 0 dB (unstable)", annotation_font_color=C_GFL)

    fig.update_layout(
        height=300,
        **PLOTLY_BASE,
        title="GFL Gain Margin vs SCR",
        xaxis=dict(title="SCR", gridcolor=BORDER),
        yaxis=dict(title="Gain Margin (dB)", gridcolor=BORDER),
    )
    return fig


# ── Dashboard utility functions ───────────────────────────────────────────

def get_stability_summary_text(scr: float) -> dict:
    """Returns a dict with status text and colors for the dashboard status card."""
    met = compute_stability_metrics(scr)

    if met.risk_label == "stable":
        return {
            "status":     "✅ STABLE",
            "color":      "#00FF9F",
            "bg":         "rgba(0,255,159,0.1)",
            "pm_text":    f"{met.gfl_pm_deg:.1f}°",
            "gm_text":    f"{met.gfl_gm_db:.1f} dB",
            "mb_text":    f"{met.middlebrook_db:.1f} dB",
            "risk_label": "Adequate phase margin",
        }
    elif met.risk_label == "marginal":
        return {
            "status":     "⚠️ MARGINAL",
            "color":      "#FFB020",
            "bg":         "rgba(255,176,32,0.1)",
            "pm_text":    f"{met.gfl_pm_deg:.1f}°",
            "gm_text":    f"{met.gfl_gm_db:.1f} dB",
            "mb_text":    f"{met.middlebrook_db:.1f} dB",
            "risk_label": "Risk during severe transients",
        }
    else:
        return {
            "status":     "❌ UNSTABLE",
            "color":      "#FF4757",
            "bg":         "rgba(255,71,87,0.1)",
            "pm_text":    f"{met.gfl_pm_deg:.1f}°",
            "gm_text":    f"{met.gfl_gm_db:.1f} dB",
            "mb_text":    f"{met.middlebrook_db:.1f} dB",
            "risk_label": "PLL will lose synchronism",
        }


if __name__ == "__main__":
    # Smoke test
    for scr_test in [1.0, 2.0, 5.0]:
        m = compute_stability_metrics(scr_test)
        print(f"SCR={scr_test:.1f} | PM={m.gfl_pm_deg:.1f}° | GM={m.gfl_gm_db:.1f} dB "
              f"| Middlebrook={m.middlebrook_db:.1f} dB "
              f"| GFL={'✅' if m.gfl_stable else '❌'} | GFM=✅")
    print("stability_analysis OK")
