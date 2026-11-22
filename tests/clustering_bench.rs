//! Run with: cargo test --release --test clustering_bench -- --ignored --nocapture
//! Synthetic CPU comparison of the public fits, not a reproduction of the paper.
use boquilahub::api::ml::{EMeans, KMeans, Pca};
use std::{hint::black_box, time::Instant};

const CASES: [(usize, usize, usize); 6] = [
    (100, 2, 8),
    (1000, 2, 8),
    (10000, 2, 8),
    (10000, 3, 20),
    (100000, 3, 20),
    (10000, 64, 32),
];

fn timed<T>(fit: impl FnOnce() -> T) -> f64 {
    let start = Instant::now();
    let result = fit();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    black_box(result);
    elapsed
}

// Alternate warmed fits; measure fitting without result destruction or scoring.
fn paired<A, B>(
    mut a: impl FnMut() -> A,
    mut b: impl FnMut() -> B,
    repeats: usize,
) -> [Vec<f64>; 2] {
    let mut times = [Vec::new(), Vec::new()];
    for repeat in 0..repeats {
        for index in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
            times[index].push(if index == 0 {
                timed(&mut a)
            } else {
                timed(&mut b)
            });
        }
    }
    times
}

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn points(n: usize, d: usize, k: usize, separated: bool, state: &mut u64) -> Vec<Vec<f64>> {
    let centers: Vec<Vec<f64>> = (0..k)
        .map(|_| (0..d).map(|_| uniform(state) * 20.0).collect())
        .collect();
    (0..n)
        .map(|_| {
            let group = (uniform(state) * k as f64) as usize;
            (0..d)
                .map(|j| {
                    if separated {
                        centers[group][j] + (uniform(state) - 0.5) * 0.1
                    } else {
                        uniform(state) * 2.0 - 1.0
                    }
                })
                .collect()
        })
        .collect()
}

fn mse(data: &[Vec<f64>], centers: &[Vec<f64>]) -> f64 {
    data.iter()
        .map(|p| {
            centers
                .iter()
                .map(|c| p.iter().zip(c).map(|(a, b)| (a - b).powi(2)).sum::<f64>())
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>()
        / data.len() as f64
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn compare_default(name: &str, data: &[Vec<f64>], k: usize, test: Option<&[Vec<f64>]>) {
    let fixed = EMeans::fit_with_params(data, k, 5, 1234).unwrap();
    let adaptive = EMeans::fit(data, k).unwrap();
    let [mut fixed_times, mut adaptive_times] = paired(
        || EMeans::fit_with_params(black_box(data), k, 5, 1234).unwrap(),
        || EMeans::fit(black_box(data), k).unwrap(),
        9,
    );
    let delta = |rows: &[Vec<f64>]| {
        100.0 * (mse(rows, &adaptive.centers) / mse(rows, &fixed.centers) - 1.0)
    };
    println!(
        "{name},{},{},{k},{:.6},{:.6},{},{:.6},{}",
        data.len(),
        data[0].len(),
        median(&mut fixed_times),
        median(&mut adaptive_times),
        adaptive.epochs,
        delta(data),
        test.map_or(String::new(), |rows| format!("{:.6}", delta(rows)))
    );
}

#[test]
#[ignore = "manual paired default-budget benchmark"]
fn compare_default_budget() {
    assert!(!cfg!(debug_assertions), "run with --release");
    println!(
        "dataset,n,d,k,fixed5_ms,adaptive_ms,adaptive_epochs,train_mse_delta_pct,test_mse_delta_pct"
    );
    for (n, d, k) in CASES {
        for separated in [true, false] {
            for seed in [17, 42, 1234] {
                let data = points(n, d, k, separated, &mut (seed as u64));
                compare_default(
                    &format!("{}-{seed}", if separated { "separated" } else { "uniform" }),
                    &data,
                    k,
                    None,
                );
            }
        }
    }
}

#[test]
#[ignore = "manual release benchmark"]
fn compare_clustering() {
    assert!(!cfg!(debug_assertions), "run with --release");
    println!("shape,n,d,k,e_ms,k_ms,speedup,k_mse,e_mse,relative_mse_pct");
    for (n, d, k) in CASES {
        for separated in [true, false] {
            let [mut e_times, mut k_times] = std::array::from_fn(|_| Vec::new());
            let [mut errors, mut e_losses, mut k_losses] = std::array::from_fn(|_| Vec::new());
            for seed in [17, 42, 1234] {
                let data = points(n, d, k, separated, &mut (seed as u64));
                // Warm both methods, then alternate order to reduce ordering bias.
                let e = EMeans::fit(&data, k).unwrap();
                let km = KMeans::fit(&data, k).unwrap();
                let e_loss = mse(&data, &e.centers);
                let k_loss = mse(&data, &km.centers);
                errors.push(100.0 * (e_loss / k_loss - 1.0));
                e_losses.push(e_loss);
                k_losses.push(k_loss);
                let [e, km] = paired(
                    || EMeans::fit(black_box(&data), black_box(k)).unwrap(),
                    || KMeans::fit(black_box(&data), black_box(k)).unwrap(),
                    5,
                );
                e_times.extend(e);
                k_times.extend(km);
            }
            let e = median(&mut e_times);
            let km = median(&mut k_times);
            println!(
                "{},{n},{d},{k},{e:.6},{km:.6},{:.3},{:.8},{:.8},{:.3}",
                if separated { "separated" } else { "uniform" },
                km / e,
                median(&mut k_losses),
                median(&mut e_losses),
                median(&mut errors)
            );
        }
    }
}

/// CLUSTER_BENCH_FVECS supplies SIFT-small. Train PCA on training rows only;
/// evaluate budgets on validation rows and report test error independently.
#[test]
#[ignore = "manual held-out quality/time benchmark; needs CLUSTER_BENCH_FVECS"]
fn compare_held_out_quality() {
    assert!(!cfg!(debug_assertions), "run with --release");
    let bytes =
        std::fs::read(std::env::var("CLUSTER_BENCH_FVECS").expect("set CLUSTER_BENCH_FVECS"))
            .unwrap();
    let d = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut rows: Vec<Vec<f64>> = bytes
        .chunks_exact(4 * (d + 1))
        .map(|row| {
            assert_eq!(u32::from_le_bytes(row[..4].try_into().unwrap()) as usize, d);
            row[4..]
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()) as f64)
                .collect()
        })
        .collect();
    let mut rng = 42;
    for i in (1..rows.len()).rev() {
        let j = (uniform(&mut rng) * (i + 1) as f64) as usize;
        rows.swap(i, j);
    }
    let split = rows.len() * 8 / 10;
    let validation_end = rows.len() * 9 / 10;
    println!(
        "dataset,dimensions,k,epochs,median_ms,validation_mse,test_mse,test_delta_pct,meets_validation_target"
    );
    for (dimensions, k) in [(2, 8), (3, 20), (d, 32)] {
        let projected = if dimensions < d {
            let pca = Pca::fit(&rows[..split], dimensions).unwrap();
            rows.iter()
                .map(|row| pca.transform(row).unwrap())
                .collect::<Vec<_>>()
        } else {
            rows.clone()
        };
        let train = &projected[..split];
        let val = &projected[split..validation_end];
        let test = &projected[validation_end..];
        // Compare default versus fixed-five with identical data and seed.
        println!(
            "dataset,n,d,k,fixed5_ms,adaptive_ms,adaptive_epochs,train_mse_delta_pct,test_mse_delta_pct"
        );
        compare_default("sift-default", train, k, Some(test));
        let km = KMeans::fit(train, k).unwrap();
        let target = mse(val, &km.centers);
        let test_target = mse(test, &km.centers);
        let mut times = Vec::new();
        for _ in 0..3 {
            times.push(timed(|| KMeans::fit(black_box(train), k).unwrap()));
        }
        println!(
            "sift-kmeans,{dimensions},{k},converged,{:.6},{target:.6},{test_target:.6},0,true",
            median(&mut times)
        );
        for epochs in [1, 2, 3, 5, 10, 25] {
            let [mut times, mut validation, mut testing] = std::array::from_fn(|_| Vec::new());
            for seed in [17, 42, 1234] {
                let start = Instant::now();
                let fit = EMeans::fit_with_params(black_box(train), k, epochs, seed).unwrap();
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                validation.push(mse(val, &fit.centers));
                testing.push(mse(test, &fit.centers));
            }
            let val = median(&mut validation);
            let test = median(&mut testing);
            println!(
                "sift-emeans,{dimensions},{k},{epochs},{:.6},{val:.6},{test:.6},{:.4},{}",
                median(&mut times),
                100.0 * (test / test_target - 1.0),
                val <= target
            );
        }
    }
}
