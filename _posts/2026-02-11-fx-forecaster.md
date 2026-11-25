---
layout: post
title: "FX Forecaster: Quantile Prediction with Sentiment-Fused LSTMs"
usemathjax: true
categories: low-level-latency-optimization
---

*February 11, 2026*

---

## The Problem

Forecasting USD/INR in the short term — say, the next 10 minutes — is hard. Rate ticks are noisy, momentum is fleeting, and macro sentiment can flip a move in seconds. A single point prediction is not just overconfident, it is misleading. What a trader actually needs is a *distribution* of outcomes: where is the rate likely to go, and how wide is the uncertainty?

This post walks through the architecture of an FX forecaster that answers exactly that — 10 future horizons, 5 probability quantiles each, fusing raw rate history with structured AI sentiment signals.

---

## Inputs

The model takes two independent inputs at inference time:

**Rate sequence** — the last 96 ticks of 15-minute resampled USD/INR rates (24 hours of price history), shaped `(B, 96)`.

**Sentiment features** — 18 floats derived from an AI analyst's read of the current macro environment (described below), shaped `(B, 18)`.

---

## Sentiment Features

An AI analyst produces a short and long summary of the current FX environment. Nine parameters are scored from that text using the following logic:

```
"positive for INR" / "appreciate"  →  +1.0
"slightly positive"                →  +0.5
"neutral"                          →   0.0
"slightly negative"                →  -0.5
"negative for INR" / "depreciate"  →  -1.0
```

The nine scored parameters are:

| Feature | What it captures |
|---|---|
| `sentiment_fx_trend` | Day's FX trend |
| `sentiment_dxy` | Broad dollar tone (DXY index) |
| `sentiment_vix` | Market risk appetite (VIX) |
| `sentiment_brent` | Brent crude trend |
| `sentiment_fpi` | FPI flows |
| `sentiment_futures` | NSE USD-INR futures positioning |
| `sentiment_fwd_premium` | 1-year forward premium |
| `sentiment_events` | Event tape (data, geopolitics, RBI) |
| `sentiment_analyst` | Analyst consensus expectations |

The `net_sentiment_score` (sum of all nine) plus the 9 individual scores plus the `short_summary` and `long_summary` text embeddings (collapsed to scalar features) form the full 18-dimensional sentiment vector fed to the model.

---

## Architecture

```
x_seq (B, 96)                          x_static (B, 18)
     │                                       │
     ▼                                       ▼
unsqueeze → (B, 96, 1)          ┌─────────────────────┐
     │                           │  Linear(18 → 64)    │
     ▼                           │  ReLU               │
┌──────────────┐                 │  Dropout(0.2)       │
│ 2-layer LSTM │                 │  Linear(64 → 64)    │
│ hidden = 64  │                 │  ReLU               │
└──────┬───────┘                 └──────────┬──────────┘
       │ h[-1] → (B, 64)                    │ (B, 64)
       └──────────────────┬─────────────────┘
                          │ concatenate → (B, 128)
                          ▼
                  ┌───────────────┐
                  │  GATED FUSION │
                  │  gate = σ(W·x)│
                  │  x = x * gate │
                  │  Linear→(B,64)│
                  └───────┬───────┘
                          │ (B, 64)
                          ▼
                  ┌───────────────┐
                  │  OUTPUT HEAD  │
                  │  Linear(64→64)│
                  │  ReLU         │
                  │  Linear(64→50)│
                  └───────┬───────┘
                          │ (B, 50)
                          ▼
                   reshape → (B, 10, 5)
                          │
                          ▼
               ┌─────────────────────┐
               │  QUANTILE ORDERING  │
               │  base = out[:,:,0]  │
               │  increments =       │
               │    softplus(Δ)      │
               │  cumsum →           │
               └──────────┬──────────┘
                          │
                          ▼
                   (B, 10, 5) — predicted rate deltas
```

### LSTM Encoder

The rate sequence is unsqueezed to `(B, 96, 1)` and passed through a 2-layer LSTM with hidden size 64. Only the final hidden state `h[-1]` is used — it summarises the entire 24-hour sequence into a 64-dimensional vector. This captures momentum, mean-reversion tendencies, and intraday patterns from the recent tick history.

### Sentiment MLP

The 18 sentiment features go through a two-layer MLP:

```
Linear(18 → 64) → ReLU → Dropout(0.2) → Linear(64 → 64) → ReLU
```

The dropout at 0.2 adds regularisation on a relatively small input — sentiment data is sparse and can overfit quickly.

### Gated Fusion

The two 64-dimensional vectors are concatenated to form a 128-dim joint representation. A learned gate then decides how much of each signal to trust:

```
gate = σ(W · x_concat)         # sigmoid → values in [0, 1]
x    = x_concat * gate          # element-wise masking
out  = Linear(128 → 64)(x)
```

This is the key design decision. The gate is *per-element*, so it can suppress the LSTM signal on flat-rate days when macro sentiment dominates, or suppress the sentiment signal when the rate is in a strong intraday trend and macro news is stale. The model learns these trade-offs from data — you do not hardcode them.

### Output Head and Quantile Ordering

The fused 64-dim vector passes through one more MLP to produce 50 raw values:

```
Linear(64 → 64) → ReLU → Linear(64 → 50)
reshape → (B, 10, 5)
```

The 10 horizons are **1 min through 10 min** ahead. The 5 quantiles are **p10, p25, p50, p75, p90**.

The raw 50 values are **rate deltas** (change from current rate), not absolute levels.

**The quantile ordering problem.** A naive network can predict p10 > p90, which is nonsensical. The softplus + cumsum trick enforces monotonicity without constraining the model's expressiveness:

```
base            = out[:, :, 0]           # raw p10 value
increments[1:]  = softplus(out[:, :, 1:])  # always ≥ 0
quantiles       = cumsum([base, increments])
```

Since `softplus(x) = log(1 + e^x) > 0` for all x, every increment is strictly positive. The cumulative sum guarantees:

```
p10 ≤ p25 ≤ p50 ≤ p75 ≤ p90
```

always, for every sample, every horizon, without any post-processing.

---

## Loss: Quantile (Pinball) Loss

For a single quantile `q` and a single prediction `ŷ` against target `y`:

$$L(y, \hat{y}, q) = \max\left(q \cdot (y - \hat{y}),\ (q - 1) \cdot (y - \hat{y})\right)$$

This is asymmetric by design. At p10, the model is penalised more heavily for predicting *above* the actual value (it should be conservative). At p90, the opposite. At p50, the loss reduces to half mean-absolute error. The final training loss averages across all 10 horizons and all 5 quantiles.

The result: a model calibrated such that ~10% of actual outcomes fall below the p10 band, ~90% fall below the p90 band. When those empirical coverage rates hold on a held-out set, you have a genuinely probabilistic forecaster.

---

## Why This Design

**Sequence + sentiment separately before fusion** — the two inputs live in very different spaces. Rate ticks are dense, continuous, and auto-correlated. Sentiment scores are sparse, structured, and change on a much slower timescale (minutes to hours). Encoding them separately before fusion lets each encoder specialise.

**Gated fusion over simple concatenation** — a straight concat + linear layer weights all 128 dimensions equally at initialisation. The gate provides an inductive bias: *the model should learn to route*, not just average. In practice this reduces the number of gradient steps needed to learn the right balance.

**Quantile outputs over point + variance** — a Gaussian output head (mean + sigma) assumes symmetric uncertainty. FX rates under news events are not symmetric — the downside on a RBI intervention day is structurally different from the upside. Quantiles capture this without imposing a distributional assumption.

**Rate deltas, not levels** — predicting `Δrate` rather than the absolute rate makes the target stationary. The distribution of 10-minute deltas is roughly consistent across different base rate regimes (84, 85, 86 USD/INR); the distribution of absolute levels is not. This matters for generalisation across time.

---

## Summary

The model combines 24 hours of price history with a structured AI analyst sentiment signal, fuses them through a learned gating mechanism, and outputs a full quantile distribution over 10 future horizons. The softplus + cumsum trick enforces valid quantile ordering without any constrained optimisation. The pinball loss trains the model to be calibrated, not just accurate at the median.

The practical output at inference time: for the next 10 minutes of USD/INR, here is not just where the rate is likely to go, but how wide the uncertainty is — and that width is itself a tradeable signal.
