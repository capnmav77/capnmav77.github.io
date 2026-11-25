---
layout: post
title: "FX Forecaster in C: From 5s to 600ms by Cutting What You Don't Need"
usemathjax: true
categories: low-level-latency-optimization
---

*March 18, 2026*

---

[The previous post]({% post_url 2026-02-11-fx-forecaster %}) described the architecture of the FX forecaster — a sentiment-fused LSTM that outputs a full quantile distribution across 10 horizons. That model runs fine in PyTorch. But when you need it ticking every minute inside a production trading loop, 5 seconds per inference is a wall you have to tear down.

This post is about how rewriting inference in C and stripping every capability the production path does not actually use took that 5 seconds to 600 milliseconds.

---

## What Was Slow

The PyTorch model is expressive: 10 horizons, 5 quantiles, a full gated fusion path, a 2-layer LSTM, quantile ordering via softplus + cumsum. At training time, all of that is load-bearing. At inference time in production, most of it is dead weight.

Profiling the Python path broke down roughly as:

| Step | Time |
|---|---|
| Python → tensor conversion + device transfer | ~800ms |
| LSTM forward (2 layers, full sequence) | ~1.4s |
| Sentiment MLP + gated fusion | ~600ms |
| Output head + softplus + cumsum | ~300ms |
| Tensor → Python extraction + post-processing | ~1.9s |

Five seconds. The model itself is maybe 2 seconds of that. The other 3 are Python overhead, object allocation, and framework bookkeeping that exists entirely to support training — autograd graphs, gradient hooks, dispatch tables, the works. None of it is needed when you are only doing a forward pass.

---

## What Production Actually Needs

The model outputs `(B, 10, 5)` — 10 horizons, 5 quantiles. In production:

- **Horizons used**: 1 min and 5 min. That is 2 of the 10.
- **Quantiles used**: p25, p50, p75. The p10/p90 tails are logged but not acted on in the base strategy.
- **Batch size**: always 1. No batching at inference.
- **Sentiment path**: only 9 of the 18 features are live at runtime; the remaining 9 are text-embedding scalars that require a separate embedding call and are updated on a 15-minute schedule, not per-tick.

The stripped target: a forward pass that produces `(2, 3)` — 2 horizons, 3 quantiles — using 9 live sentiment scalars and the 96-tick rate sequence.

---

## Rewriting in C

The weights are exported from PyTorch as raw float arrays. The C implementation is a straight matrix multiply + activation loop over those arrays — no framework, no dynamic dispatch, no garbage collector.

### LSTM: Single Layer, No Gradient State

The original model used a 2-layer LSTM. The second layer added ~15% accuracy on validation but roughly doubled the LSTM compute. In production, the accuracy delta was below the threshold that changes a signal — it was kept during training for regularisation, not because the second layer's representations were needed at inference.

The C LSTM implements a single-layer forward pass:

```c
// single LSTM cell step
void lstm_step(float *h, float *c,
               const float *x, const float *Wh, const float *Wx,
               const float *b, int hidden) {
    float gates[4 * hidden];
    // input + hidden projections, fused into one pass
    matvec(gates, Wx, x, 4 * hidden, 1);
    matvec_add(gates, Wh, h, 4 * hidden, hidden);
    vec_add(gates, b, 4 * hidden);

    for (int i = 0; i < hidden; i++) {
        float ig = sigmoid(gates[i]);
        float fg = sigmoid(gates[hidden + i]);
        float gg = tanhf(gates[2 * hidden + i]);
        float og = sigmoid(gates[3 * hidden + i]);
        c[i] = fg * c[i] + ig * gg;
        h[i] = og * tanhf(c[i]);
    }
}
```

No autograd, no saved tensors, no layer norm buffers. The 96 steps of the sequence are a tight loop over this function.

### Sentiment MLP: 9 Features, One Layer Folded

The two-layer sentiment MLP (`18→64→64`) becomes a single fused layer for the 9 live features:

The first linear was `Linear(18 → 64)`. With only 9 features active, the remaining 9 input rows of the weight matrix are always multiplied by zero. Those 9 rows are dropped at export time — the exported weight is `Linear(9 → 64)`. The two linear layers are then fused: since `Dropout(0.2)` is identity at inference and there is no skip connection, `Linear(64→64)(ReLU(Linear(9→64)(x)))` collapses to a single matrix after absorbing the ReLU (not analytically, but the two matmuls can be executed back-to-back with no intermediate allocation beyond a 64-float buffer).

### Gated Fusion: No Dynamic Allocation

The gate in PyTorch allocates intermediate tensors for the sigmoid, the elementwise product, and the linear projection. In C, the entire fusion is three in-place operations on a 128-float stack buffer:

```c
float fused[128];
memcpy(fused,       h_seq,    64 * sizeof(float));
memcpy(fused + 64,  h_static, 64 * sizeof(float));

float gate[128];
matvec(gate, W_gate, fused, 128, 128);
for (int i = 0; i < 128; i++) gate[i] = sigmoid(gate[i]);
for (int i = 0; i < 128; i++) fused[i] *= gate[i];

float h_fused[64];
matvec(h_fused, W_proj, fused, 64, 128);
```

128 floats on the stack. No heap, no free.

### Output Head: Only the Horizons and Quantiles You Use

The original output head produces 50 values (`10 horizons × 5 quantiles`). The C version exports only the rows of the final weight matrix corresponding to the 2 active horizons and 3 active quantiles — 6 values out instead of 50. The softplus + cumsum quantile ordering is applied only over those 3 quantiles:

```c
// p25, p50, p75 ordering (indices 1, 2, 3 of the original 5)
float base = out[0];
float p25  = base;
float p50  = p25 + softplus(out[1]);
float p75  = p50 + softplus(out[2]);
```

Three softplus calls instead of four cumsum steps over 5 values — trivial in isolation, but the real saving is that 44 of the 50 output neurons are never computed at all.

---

## Where the Time Actually Went

After the port, the breakdown is:

| Step | Time |
|---|---|
| Rate array copy (96 floats) | ~0.1ms |
| Single-layer LSTM forward (96 steps) | ~180ms |
| Sentiment MLP (9→64→64, fused) | ~40ms |
| Gated fusion + output head | ~15ms |
| Result extraction | ~0.1ms |
| **Total** | **~235ms** |

Wait — that is 235ms, not 600ms. The remaining ~365ms in the production number is the **sentiment fetch**: calling the AI analyst API, parsing the JSON, scoring the 9 parameters. That part cannot be eliminated, but it runs on a 15-minute schedule in a background thread and writes into a shared float array. The inference path reads from that array — it does not block on it.

The 600ms wall-clock figure includes the occasional sentiment refresh amortised over the per-minute ticks. The forward pass itself is under 250ms.

---

## What Was Not Cut

A few things were kept even though they add cost:

**The gated fusion** — removing it and concatenating directly into the output head reduced latency by another ~10ms but degraded live signal quality noticeably. The gate is load-bearing in production, not just at training time.

**Quantile ordering** — the softplus + cumsum adds 3 function calls. The alternative (clamping or sorting) introduces discontinuities in the output that caused downstream signal noise. Kept.

**The full 96-tick LSTM sequence** — truncating to 48 ticks (12h) saved ~90ms but visibly degraded 5-minute horizon accuracy during backtesting. The 24-hour window captures end-of-day mean-reversion patterns that a 12-hour window misses.

---

## The Core Lesson

The PyTorch model is an *exploration vehicle*. It carries all 10 horizons because you need them during development to understand which horizons are predictable. It carries 5 quantiles because you want the full distribution during backtesting. It carries 2 LSTM layers and an 18-feature sentiment path because you are still learning what matters.

By the time you ship, you know. The C port is not a rewrite — it is a *read of what you learned*. Every dropped weight row, every skipped output neuron, every stack-allocated buffer is a decision the training process made for you. The C code just makes it explicit.

Five seconds of Python curiosity. Six hundred milliseconds of C conviction.
