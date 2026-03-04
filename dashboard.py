"""
Monte Carlo Dashboard — Streamlit
-----------------------------------
Run with:   streamlit run dashboard.py
Requires:   pip install streamlit plotly numpy pandas
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from simulation import SimulationConfig, SimulationResult, run_simulation, compute_stats, percentile_band


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Monte Carlo Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root {
    --bg:        #1a1714;
    --surface:   #211e1a;
    --border:    #333028;
    --accent:    #cc785c;
    --accent2:   #d4a574;
    --accent3:   #8fab6e;
    --text:      #e8e0d5;
    --muted:     #7a7060;
    --gain:      #8fab6e;
    --loss:      #c26b5a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Source Sans 3', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--accent) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"]  { font-family: 'Playfair Display', serif; font-size: 1.6rem !important; color: var(--accent) !important; }
[data-testid="stMetricLabel"]  { font-size: 0.65rem !important; letter-spacing: 0.12em; color: var(--muted) !important; text-transform: uppercase; font-family: 'Source Sans 3', sans-serif; }
[data-testid="stMetricDelta"]  { font-size: 0.75rem !important; }

/* ── Section headers ── */
.section-head {
    font-family: 'Arial Narrow', Arial, sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 18px;
}

/* ── Hero title ── */
.hero {
    font-family: 'Arial Narrow', Arial, sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1.2;
    color: var(--text);
    letter-spacing: 0em;
}
.hero span { color: var(--accent); }

.subtitle {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin-top: 4px;
    text-transform: uppercase;
    font-family: 'Source Sans 3', sans-serif;
}

/* ── Tag badge ── */
.badge {
    display: inline-block;
    background: rgba(204,120,92,0.10);
    border: 1px solid rgba(204,120,92,0.30);
    color: var(--accent);
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 3px;
    margin-right: 6px;
    font-family: 'Source Sans 3', sans-serif;
}

/* ── Plotly chart containers ── */
.js-plotly-plot { border-radius: 6px; }

/* ── Streamlit overrides ── */
.stSelectbox > div > div { background: var(--surface) !important; border-color: var(--border) !important; }
div[data-baseweb="select"] { background: var(--surface) !important; }
.stNumberInput input, .stTextInput input { background: var(--surface) !important; border-color: var(--border) !important; color: var(--text) !important; }
hr { border-color: var(--border) !important; }
.block-container { padding-top: 4rem; padding-bottom: 4rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1a1714",
    plot_bgcolor="#1a1714",
    font=dict(family="Source Sans 3, sans-serif", color="#e8e0d5", size=11),
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(gridcolor="#333028", linecolor="#333028", zerolinecolor="#333028"),
    yaxis=dict(gridcolor="#333028", linecolor="#333028", zerolinecolor="#333028"),
    legend=dict(bgcolor="rgba(33,30,26,0.8)", bordercolor="#333028", borderwidth=1),
)

ACCENT   = "#cc785c"
ACCENT2  = "#d4a574"
ACCENT3  = "#8fab6e"
GAIN     = "#8fab6e"
LOSS     = "#c26b5a"


# ─────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-head">⚙ Parameters</div>', unsafe_allow_html=True)

    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()

    start_price = st.number_input(
        "Current Price ($)", min_value=1.0, max_value=10_000.0,
        value=195.0, step=0.5
    )

    st.markdown("---")
    st.markdown('<div class="section-head">Market Assumptions</div>', unsafe_allow_html=True)

    mu = st.slider(
        "Annualised Drift (μ)", min_value=-0.30, max_value=0.50,
        value=0.10, step=0.01,
        help="Expected annual return, e.g. 0.10 = 10%"
    )

    sigma = st.slider(
        "Annualised Volatility (σ)", min_value=0.05, max_value=1.00,
        value=0.25, step=0.01,
        help="Annual standard deviation, e.g. 0.25 = 25%"
    )

    st.markdown("---")
    st.markdown('<div class="section-head">Simulation Settings</div>', unsafe_allow_html=True)

    days = st.slider("Trading Days", min_value=21, max_value=504, value=252, step=21)
    n_paths = st.select_slider(
        "Simulated Paths",
        options=[500, 1_000, 2_500, 5_000, 10_000],
        value=2_500,
    )
    seed = st.number_input("Random Seed", value=42, step=1)

    run_btn = st.button("▶  Run Simulation", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.58rem;color:#4a6070;line-height:1.7">'
        'Model: Geometric Brownian Motion<br>'
        'dS = μS dt + σS dW<br><br>'
        'Not financial advice.'
        '</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

col_title, col_badges = st.columns([3, 2])
with col_title:
    st.markdown(
        f'<div class="hero">Monte <span>Carlo</span><br>Simulator</div>'
        f'<div class="subtitle">Geometric Brownian Motion · Equity Path Forecasting</div>',
        unsafe_allow_html=True
    )
with col_badges:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        f'<span class="badge">GBM</span>'
        f'<span class="badge">Stochastic</span>'
        f'<span class="badge">{n_paths:,} Paths</span>'
        f'<span class="badge">{days}D Horizon</span>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RUN SIMULATION
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_run(ticker, start_price, mu, sigma, days, n_paths, seed):
    cfg = SimulationConfig(
        ticker=ticker,
        start_price=start_price,
        mu=mu,
        sigma=sigma,
        days=days,
        n_paths=n_paths,
        seed=int(seed),
    )
    result = run_simulation(cfg)
    stats  = compute_stats(result)
    return result, stats


if "result" not in st.session_state or run_btn:
    with st.spinner("Running simulation..."):
        result, stats = cached_run(ticker, start_price, mu, sigma, days, n_paths, seed)
    st.session_state.result = result
    st.session_state.stats  = stats
else:
    result = st.session_state.result
    stats  = st.session_state.stats


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────

st.markdown('<div class="section-head">Key Statistics</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)

def pct(v): return f"{v*100:+.1f}%"
def price(v): return f"${v:,.2f}"

k1.metric("Mean Price",    price(stats["mean_price"]),   pct((stats["mean_price"] - start_price) / start_price))
k2.metric("Median Price",  price(stats["median_price"]), pct((stats["median_price"] - start_price) / start_price))
k3.metric("5th Pctile",   price(stats["p5"]),            pct((stats["p5"] - start_price) / start_price))
k4.metric("95th Pctile",  price(stats["p95"]),           pct((stats["p95"] - start_price) / start_price))
k5.metric("Prob. Profit",  f"{stats['prob_profit']*100:.1f}%")
k6.metric("VaR (95%)",     f"{stats['var_95']*100:.1f}%", help="5th percentile of returns")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHART 1 — PATH FAN
# ─────────────────────────────────────────────

st.markdown('<div class="section-head">Price Path Simulation</div>', unsafe_allow_html=True)

days_axis = np.arange(result.paths.shape[0])

def build_fan_chart(result: SimulationResult) -> go.Figure:
    fig = go.Figure()

    # Shaded percentile bands
    bands = [
        (5,  95, "rgba(204,120,92,0.06)", "5–95th"),
        (15, 85, "rgba(204,120,92,0.10)", "15–85th"),
        (25, 75, "rgba(204,120,92,0.16)", "25–75th"),
    ]
    for lo, hi, fill_color, label in bands:
        lower, upper = percentile_band(result, lo, hi)
        fig.add_trace(go.Scatter(
            x=days_axis, y=upper,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=days_axis, y=lower,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=fill_color,
            name=label, hoverinfo="skip",
        ))

    # A handful of individual paths (visual texture)
    sample_idx = np.random.default_rng(99).integers(0, result.config.n_paths, size=60)
    path_colors = [
        "rgba(204,120,92,0.85)",
        "rgba(212,165,116,0.85)",
        "rgba(143,171,110,0.85)",
        "rgba(180,150,100,0.85)",
        "rgba(160,130,110,0.85)",
    ]
    for i, idx in enumerate(sample_idx):
        fig.add_trace(go.Scatter(
            x=days_axis, y=result.paths[:, idx],
            mode="lines",
            line=dict(color=path_colors[i % len(path_colors)], width=1.2),
            showlegend=False, hoverinfo="skip",
        ))

    # Median and mean
    median_path = np.median(result.paths, axis=1)
    mean_path   = np.mean(result.paths,   axis=1)

    fig.add_trace(go.Scatter(
        x=days_axis, y=median_path,
        mode="lines", name="Median",
        line=dict(color=ACCENT, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=days_axis, y=mean_path,
        mode="lines", name="Mean",
        line=dict(color=ACCENT2, width=1.5, dash="dot"),
    ))

    # Start price reference
    fig.add_hline(
        y=result.config.start_price,
        line=dict(color="#4a6070", width=1, dash="dash"),
        annotation_text="Entry", annotation_font_color="#4a6070",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"{result.config.ticker} — {result.config.n_paths:,} Simulated Paths", font=dict(size=13, color="#fff")),
        xaxis_title="Trading Days",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=420,
    )
    return fig

st.plotly_chart(build_fan_chart(result), use_container_width=True)


# ─────────────────────────────────────────────
# CHARTS ROW 2 — Distribution + Return CDF
# ─────────────────────────────────────────────

col_hist, col_cdf = st.columns(2)

# ── Final Price Distribution
with col_hist:
    st.markdown('<div class="section-head">Final Price Distribution</div>', unsafe_allow_html=True)

    fp = result.final_prices

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=fp,
        nbinsx=80,
        marker=dict(
            color=ACCENT,
            opacity=0.7,
            line=dict(color="#080c12", width=0.3),
        ),
        name="Price Distribution",
    ))

    # Vertical reference lines
    for val, color, label in [
        (start_price,     "#4a6070", "Entry"),
        (stats["p5"],     LOSS,      "P5"),
        (stats["median_price"], ACCENT, "Median"),
        (stats["p95"],    GAIN,      "P95"),
    ]:
        fig_hist.add_vline(x=val, line=dict(color=color, width=1.2, dash="dash"),
                           annotation_text=label, annotation_font_color=color,
                           annotation_font_size=9)

    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Distribution of Terminal Prices", font=dict(size=12, color="#fff")),
        xaxis_title="Price (USD)", yaxis_title="Frequency",
        height=340, showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ── Return CDF
with col_cdf:
    st.markdown('<div class="section-head">Return Cumulative Distribution</div>', unsafe_allow_html=True)

    returns = (fp - start_price) / start_price
    sorted_r = np.sort(returns)
    cdf      = np.arange(1, len(sorted_r)+1) / len(sorted_r)

    fig_cdf = go.Figure()

    # Fill profit / loss zones
    loss_mask  = sorted_r <  0
    profit_mask = sorted_r >= 0

    fig_cdf.add_trace(go.Scatter(
        x=sorted_r[loss_mask], y=cdf[loss_mask],
        mode="lines", line=dict(color=LOSS, width=2),
        fill="tozeroy", fillcolor=f"rgba(194,107,90,0.12)",
        name="Loss Zone",
    ))
    fig_cdf.add_trace(go.Scatter(
        x=sorted_r[profit_mask], y=cdf[profit_mask],
        mode="lines", line=dict(color=GAIN, width=2),
        fill="tozeroy", fillcolor=f"rgba(143,171,110,0.10)",
        name="Profit Zone",
    ))

    fig_cdf.add_vline(x=0, line=dict(color="#4a6070", width=1, dash="dash"))

    fig_cdf.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Cumulative Return Probability", font=dict(size=12, color="#fff")),
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        xaxis_title="Return",
        yaxis_title="Cumulative Probability",
        height=340,
    )
    st.plotly_chart(fig_cdf, use_container_width=True)


# ─────────────────────────────────────────────
# CHART 3 — Volatility over time (rolling std)
# ─────────────────────────────────────────────

st.markdown('<div class="section-head">Simulated Uncertainty Over Time</div>', unsafe_allow_html=True)

col_vol, col_prob = st.columns(2)

with col_vol:
    cross_sectional_std  = np.std(result.paths, axis=1)
    cross_sectional_mean = np.mean(result.paths, axis=1)

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=days_axis, y=cross_sectional_std,
        mode="lines", name="Cross-sectional Std",
        line=dict(color=ACCENT2, width=2),
        fill="tozeroy", fillcolor="rgba(212,165,116,0.10)",
    ))
    fig_vol.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Price Dispersion (σ across paths)", font=dict(size=12, color="#fff")),
        xaxis_title="Trading Day", yaxis_title="Std Dev of Price ($)",
        height=300,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with col_prob:
    # Rolling prob above entry price at each day
    prob_above = np.mean(result.paths > start_price, axis=1)

    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(
        x=days_axis, y=prob_above,
        mode="lines", name="P(price > entry)",
        line=dict(color=GAIN, width=2),
        fill="tozeroy", fillcolor="rgba(143,171,110,0.10)",
    ))
    fig_prob.add_hline(y=0.5, line=dict(color="#4a6070", width=1, dash="dash"))

    fig_prob.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Probability of Being Above Entry Price", font=dict(size=12, color="#fff")),
        xaxis_title="Trading Day",
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], tickformat=".0%"),
        yaxis_title="Probability",
        height=300,
    )
    st.plotly_chart(fig_prob, use_container_width=True)


# ─────────────────────────────────────────────
# STATS TABLE
# ─────────────────────────────────────────────

st.markdown('<div class="section-head">Full Risk Report</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

table_price = {
    "Metric": ["Mean Price", "Median Price", "Std Dev", "Min", "Max",
               "P5 (Bear)", "P25", "P75", "P95 (Bull)"],
    "Value":  [
        price(stats["mean_price"]),
        price(stats["median_price"]),
        f"${stats['std_price']:,.2f}",
        price(stats["min_price"]),
        price(stats["max_price"]),
        price(stats["p5"]),
        price(stats["p25"]),
        price(stats["p75"]),
        price(stats["p95"]),
    ],
}

table_risk = {
    "Metric": ["Prob. Profit", "Prob. Loss >10%", "Prob. Gain >20%",
               "Mean Return", "Median Return", "VaR (95%)", "CVaR (95%)", "Approx Sharpe"],
    "Value": [
        f"{stats['prob_profit']*100:.1f}%",
        f"{stats['prob_loss_10']*100:.1f}%",
        f"{stats['prob_gain_20']*100:.1f}%",
        pct(stats["mean_return"]),
        pct(stats["median_return"]),
        f"{stats['var_95']*100:.1f}%",
        f"{stats['cvar_95']*100:.1f}%",
        f"{stats['sharpe_approx']:.3f}",
    ],
}

with col_l:
    st.dataframe(
        pd.DataFrame(table_price),
        use_container_width=True, hide_index=True,
    )

with col_r:
    st.dataframe(
        pd.DataFrame(table_risk),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-size:0.6rem;color:#1c2a3a;letter-spacing:0.15em;">'
    'MONTE CARLO SIMULATOR · GEOMETRIC BROWNIAN MOTION · NOT FINANCIAL ADVICE'
    '</div>',
    unsafe_allow_html=True
)
