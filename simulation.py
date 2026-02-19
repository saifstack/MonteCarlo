"""
Monte Carlo Stock Price Simulator
----------------------------------
Core simulation logic — kept separate from the UI layer.
Uses Geometric Brownian Motion (GBM), the industry-standard model
for equity price paths.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    ticker: str
    start_price: float
    mu: float          # annualised drift  (e.g. 0.08  →  8 %)
    sigma: float       # annualised volatility (e.g. 0.20 → 20 %)
    days: int          # trading days to simulate
    n_paths: int       # number of Monte Carlo paths
    seed: int | None = 42


@dataclass
class SimulationResult:
    paths: np.ndarray          # shape (days+1, n_paths)
    final_prices: np.ndarray   # shape (n_paths,)
    config: SimulationConfig


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    """
    Simulate price paths via Geometric Brownian Motion.

    Each daily step follows:
        S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt * Z)
    where Z ~ N(0,1).
    """
    rng = np.random.default_rng(cfg.seed)

    dt = 1 / 252  # one trading day in years

    # Daily log-returns for every path at once
    drift = (cfg.mu - 0.5 * cfg.sigma ** 2) * dt
    diffusion = cfg.sigma * np.sqrt(dt)
    shocks = rng.standard_normal((cfg.days, cfg.n_paths))

    log_returns = drift + diffusion * shocks          # (days, n_paths)
    cumulative = np.vstack([
        np.zeros(cfg.n_paths),                        # day 0 — log-return = 0
        np.cumsum(log_returns, axis=0),
    ])                                                # (days+1, n_paths)

    paths = cfg.start_price * np.exp(cumulative)     # (days+1, n_paths)
    final_prices = paths[-1]

    return SimulationResult(paths=paths, final_prices=final_prices, config=cfg)


# ── Summary statistics ───────────────────────────────────────────────────────

def compute_stats(result: SimulationResult) -> dict:
    fp = result.final_prices
    start = result.config.start_price

    returns = (fp - start) / start

    stats = {
        "mean_price":    float(np.mean(fp)),
        "median_price":  float(np.median(fp)),
        "std_price":     float(np.std(fp)),
        "min_price":     float(np.min(fp)),
        "max_price":     float(np.max(fp)),
        "p5":            float(np.percentile(fp, 5)),
        "p25":           float(np.percentile(fp, 25)),
        "p75":           float(np.percentile(fp, 75)),
        "p95":           float(np.percentile(fp, 95)),
        "prob_profit":   float(np.mean(fp > start)),
        "prob_loss_10":  float(np.mean(fp < start * 0.90)),
        "prob_gain_20":  float(np.mean(fp > start * 1.20)),
        "mean_return":   float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "sharpe_approx": float(
            np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        ),
        "var_95":        float(np.percentile(returns, 5)),   # Value-at-Risk
        "cvar_95":       float(np.mean(returns[returns <= np.percentile(returns, 5)])),
    }
    return stats


def percentile_band(result: SimulationResult, low: float, high: float) -> tuple:
    """Return (lower_band, upper_band) arrays across time for the given percentile range."""
    lower = np.percentile(result.paths, low, axis=1)
    upper = np.percentile(result.paths, high, axis=1)
    return lower, upper
