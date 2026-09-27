# Pred metadata and RAM sorting

Each image, audio, and video Pred holds `Option<std::fs::Metadata>`, populated by
`new_simple`. The shared `Pred::metadata()` accessor exposes dates, byte size,
permissions, and file type for sorting, filtering, and future plotting.
Missing/unreadable files have no snapshot. Reloading refreshes the snapshot; external
changes do not silently alter the loaded dataset. Metadata is never saved in prediction
sidecars, including video sidecars.

Sorting uses only RAM. There are no filesystem queries, directory scans, parallel
workers, or cache invalidation machinery in the sorter. Equal dates retain input
order, missing dates stay last in either direction, and source indexes remain intact.

## Reproduce

```sh
cargo test --lib --release benchmark_date_sorting --locked -- --ignored --nocapture
```

The benchmark creates 100,000 distinct temporary files with shuffled modified dates,
loads real Preds, and **deletes every source file before timing**. It runs the production
sorter on 1,000/10,000/100,000 shuffled Preds, with five samples per field/direction.
Setup, loading, deletion, and correctness checks are outside the timed section.
Every case verifies ordering, stable ties, and that indexes form a complete permutation.

## Results — 2026-09-27

Windows, Ryzen 9 9950X3D, optimized release build, one calling thread. Median of five
samples; all times below are milliseconds.

| Files | Modified ascending | Modified descending | Created ascending | Created descending |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.0108 | 0.0105 | 0.0106 | 0.0104 |
| 10,000 | 0.1414 | 0.1299 | 0.1386 | 0.1428 |
| 100,000 | 2.1368 | 2.0196 | 2.2681 | 2.3447 |

Previously, per-sort filesystem reads took about 1,806 ms for 100,000 files;
directory batching reduced this to about 67 ms. That batching code is now removed.

Metadata snapshots occupy 8,800,000 bytes (8.39 MiB) per 100,000 Preds on this build,
excluding paths, predictions, and temporary sorting buffers. Sorting is stable
O(n log n) with O(n) additional space.

File loading still accesses storage once for metadata and separately for prediction
sidecars. The benchmark's file creation and actual Pred loading took 13.68 seconds;
this combines fixture setup with loading and is not a standalone loading measurement.
Disk speed affects loading, not subsequent sorts. Actual low-end CPUs and other
platforms have not been benchmarked; one thread does not simulate a slower CPU.

Two focused correctness tests cover all three constructors, metadata exclusion from
sidecars, refresh on reload (including video sidecar hits), precision, missing dates,
stable ties, and sorting with deleted sources. Tests remain under `tests/`; the library
includes the private sorter tests by path without exposing a production test API.
