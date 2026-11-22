# E-means

CPU implementation based on [Bessa et al., Algorithm 1 / Appendix H](https://arxiv.org/html/2611.09501v1)
and [Faiss NumpyEmeans 1.16.0](https://github.com/facebookresearch/faiss/tree/v1.16.0/contrib/e_means).

For batch count `n`, sum `s`, mass `N`, center `c` and scheduled step `alpha`:

- `N_new = (1-alpha)*N + alpha*n`
- `c_new = c + (alpha*n/N_new)*(s/n-c)`

Every five batches, low-mass clusters can be revived by splitting live donors
sampled without replacement, proportional to mass. Closing centers are means of
one pass's assignments; those labels need not be nearest to the closing centers.

## Default and reference fits

`EMeans::fit` uses seed 1234 and retains the closing fit with the lowest assigned
sum of squared errors (SSE). Five passes calibrate the batch size and cosine
schedule, but do not force the run length:

- One cluster returns its mean without EMA iterations.
- A repeated partition can finish early if verified against its closing means.
- Otherwise, after the five-pass horizon, stop following two passes without
  0.1% improvement in best SSE. Keep alpha at its schedule floor; cap at 25 passes.

`epochs` reports work executed, even when an earlier fit is returned.
`fit_with_params` uses the exact requested budget and seed for reference checks.
Assigned SSE is measured with direct residuals to avoid cancellation; it does
not guarantee minimum nearest-center error or held-out error.

## Local choices

- Contiguous, scaled f64 input; specialized 2D/3D distances and vectorizable
  accumulation for larger dimensions. Training reads every row, including tails.
- Farthest-first initialization from a random pool of at most `32*k` rows avoids
  a full O(nkd) seeding pass. It improves separated-group coverage over random
  seeds, but can still miss rare groups. Duplicate seeds are removed.
- Revival is disabled when batch size is below k: tiny batches produced noisy
  mass estimates that destroyed valid groups. Otherwise its threshold is 0.1.
- Faiss conventions: initial mass `2^-16`, split perturbation `1e-4`, automatic
  quality batch law through 16 dimensions and throughput law above that.

Memory is O(nd + kd + n); distance work is O(nkd) per pass. Adaptive scoring adds
O(nd), and fixed-point verification adds a nearest-center scan when applicable.
The UI clusters PCA coordinates. Its timing covers the fit, excluding PCA and
rendering. No Python runtime or new dependency is required.

## Verification

```
cargo test --test e-means --test ml --lib
```

[e-means-oracle.json](e-means-oracle.json) stores inputs, settings, expected
centers/labels, revival counts, and reference-source hashes. Generated outside
this repository with faiss-cpu 1.16.0, NumPy 2.5.5, and one Faiss thread.
Tests compare every label exactly and center coordinates within
`2e-6*(1+abs(expected))`, including runs with revival and different budgets.
Separate regressions cover adaptive stopping, means, scale, precision, invalid
input, separated groups, and UI visibility/layout.

To regenerate, scale each input in f64 and reproduce the seeding and row order
from `e-means.rs`; then convert data/seeds to float32. Construct NumpyEmeans with
the stored seed and budget, call `begin`, set the revival threshold to 0 for
`B < k` (otherwise 0.1), and supply the same uniforms through its split RNG.
Feed batches through `accumulate`, `update`, `end_epoch`, and `finalize`.
Scatter final-pass labels into input order and restore the center scale.
Faiss performs the update, donor selection, splitting and finalization math.

## Benchmarks

```
cargo test --release --test clustering_bench -- --ignored --nocapture --test-threads=1
```

The three ignored benchmarks serve different comparisons: `compare_clustering`
compares the public defaults, `compare_default_budget` isolates adaptive versus
fixed-five stopping, and `compare_held_out_quality` measures real-data error
across budgets. Synthetic timings alternate warmed fits across three dataset
seeds; scoring is outside the timer. The SIFT test needs `CLUSTER_BENCH_FVECS`
pointing to `siftsmall_base.fvecs` from this [SIFT-small mirror](https://huggingface.co/datasets/vecdata/siftsmall/blob/main/siftsmall.tar.gz).
Its SHA-256 is `1e90414a3254361aba48a0d58e461f7661fb135dabb3da985290e94de8f60fe0`.
The seed-42 shuffle splits rows 80/10/10 into training/validation/test; PCA is
fitted only on training rows. Validation and test targets are reported separately.

Recorded evidence: [former fixed-five release run](clustering-benchmark.txt),
[adaptive before/after audit](clustering-adaptive-benchmark.txt). These are dated
measurements, not promises of current runtime. The audit found roughly 2x gains
on easy PCA groups, but SIFT 128D rose from 24.4 to 41.0 ms with only 0.032% lower
training error and 0.0155% higher test error. Small noisy inputs can also be
slower. Local CPU benchmarks do not reproduce the paper's GPU headline or
establish universal speed/quality superiority.
