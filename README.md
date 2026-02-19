# Monte Carlo Stock Price Simulator

A interactive dashboard that simulates thousands of possible futures for a stock price and visualizes the range of outcomes — from the worst-case scenarios to the most optimistic ones.

Built with Python and Streamlit.


## What it actually does

Stock prices don't move in straight lines. They drift up or down over time, but with a lot of randomness layered on top. This tool uses a well-established financial model called **Geometric Brownian Motion** to capture both of those forces — the general direction a stock tends to move (drift), and the day-to-day unpredictability (volatility).

Instead of giving you one predicted price, it runs the simulation thousands of times. Each run is slightly different because of the randomness involved. The result is a full picture of what *could* happen — not just what's most likely.


## What you can control

| Parameter | What it means |
|---|---|
| **Ticker Symbol** | Just a label — doesn't pull live data, purely for display |
| **Current Price** | The price you're starting the simulation from |
| **Annualised Drift (μ)** | The expected yearly return, e.g. `0.10` means 10% per year |
| **Annualised Volatility (σ)** | How wildly the price swings, e.g. `0.25` means 25% annual volatility |
| **Trading Days** | How far into the future to simulate (252 = roughly 1 year) |
| **Simulated Paths** | How many parallel "what if" scenarios to run |
| **Random Seed** | Lock this number to get the same simulation every time you run it |


## What the charts show

**Price Path Simulation**
The main chart. Each faint line is one simulated future. The shaded bands show where most paths end up — tighter in the middle (where outcomes cluster), wider at the edges (the extremes). The solid line is the median outcome.

**Final Price Distribution**
A histogram of where the price ended up across all simulations. Skewed right means more upside potential; a wide spread means high uncertainty.

**Cumulative Return Probability**
Shows the full picture of returns from worst to best. The red zone is loss territory, green is profit. The point where the line crosses 50% is your median return.

**Price Dispersion Over Time**
As you go further into the future, simulated paths spread further apart. This chart captures that widening uncertainty — a reminder that longer horizons mean less certainty, not more.

**Probability of Being Above Entry**
At each point in time, what fraction of simulated paths are still in profit? This drops when volatility is high or drift is negative.


## Risk metrics explained

| Metric | Plain English |
|---|---|
| **P5 / P95** | The 5th and 95th percentile prices — your bear and bull case |
| **Prob. Profit** | Percentage of simulated paths that ended above your entry price |
| **VaR (95%)** | In the worst 5% of outcomes, the return was *at least* this bad |
| **CVaR (95%)** | The *average* return across those worst 5% outcomes |
| **Approx Sharpe** | Return per unit of risk — higher is better |


## Setup

**1. Clone or download the repo**

```bash
git clone https://github.com/YOUR_USERNAME/monte-carlo-simulator.git
cd monte-carlo-simulator
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the dashboard**

```bash
python -m streamlit run dashboard.py
```

The app opens in your browser automatically at `http://localhost:8501`.

---

## Project structure

```
monte-carlo-simulator/
│
├── simulation.py       # The math — GBM engine, stat calculations, percentile bands
├── dashboard.py        # The visuals — Streamlit UI, all charts and controls
├── requirements.txt    # Dependencies
└── README.md
```

The two files are intentionally kept separate. `simulation.py` is pure logic with no UI code in it, so if you ever want to swap out the frontend or run the simulation headlessly, you can import it directly without touching anything else.


## Requirements

- Python 3.10+
- streamlit
- plotly
- numpy
- pandas


## Disclaimer

This is a modelling tool, not a crystal ball. The simulation is only as good as the assumptions you feed it. Real markets are influenced by earnings, macro events, sentiment, and a hundred other things that no random process can fully capture. Use this to build intuition and explore scenarios — not to make financial decisions.


## License

MIT — do whatever you want with it.
