use boquilahub::api::ml::EMeans;
use serde::Deserialize;

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum()
}

#[derive(Deserialize)]
struct Case {
    name: String,
    data: Vec<Vec<f64>>,
    count: usize,
    centers: Vec<Vec<f64>>,
    labels: Vec<usize>,
    epochs: usize,
    seed: u64,
    splits: usize,
}

#[test]
fn matches_faiss_numpy_emeans() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("assets/e-means-oracle.json")).unwrap();
    let cases: Vec<Case> = serde_json::from_value(oracle["cases"].clone()).unwrap();
    assert!(
        cases.iter().any(|case| case.splits > 0),
        "oracle must exercise revival"
    );
    for case in cases {
        let fit = EMeans::fit_with_params(&case.data, case.count, case.epochs, case.seed).unwrap();
        assert_eq!(fit.epochs, case.epochs);
        assert_eq!(fit.labels, case.labels, "{}", case.name);
        assert_eq!(fit.centers.len(), case.centers.len());
        for (actual, expected) in fit
            .centers
            .iter()
            .flatten()
            .zip(case.centers.iter().flatten())
        {
            assert!(
                (actual - expected).abs() < 2e-6 * (1.0 + expected.abs()),
                "{}: {actual} != {expected}",
                case.name
            );
        }
    }
}

#[test]
fn deterministic_scale_safe_and_closing_centers_are_means() {
    let data: Vec<_> = (0..1000)
        .map(|i| vec![(i as f64 * 0.73).sin(), (i as f64 * 1.17).cos()])
        .collect();
    let fit = EMeans::fit(&data, 8).unwrap();
    assert_eq!(fit.labels, EMeans::fit(&data, 8).unwrap().labels);
    for factor in [1e-200, 1e200] {
        let scaled: Vec<Vec<f64>> = data
            .iter()
            .map(|p| p.iter().map(|x| x * factor).collect())
            .collect();
        let result = EMeans::fit(&scaled, 8).unwrap();
        assert_eq!(fit.labels, result.labels);
        for (a, b) in fit
            .centers
            .iter()
            .flatten()
            .zip(result.centers.iter().flatten())
        {
            assert!((a - b / factor).abs() < 1e-12);
        }
    }
    for (j, center) in fit.centers.iter().enumerate() {
        let rows: Vec<_> = data
            .iter()
            .zip(&fit.labels)
            .filter(|(_, label)| **label == j)
            .collect();
        if rows.is_empty() {
            continue;
        }
        for axis in 0..2 {
            let mean = rows.iter().map(|(p, _)| p[axis]).sum::<f64>() / rows.len() as f64;
            assert!((center[axis] - mean).abs() < 1e-12);
        }
    }
}

#[test]
fn rejects_invalid_input() {
    assert!(EMeans::fit_with_params(&[vec![1.0]], 1, 0, 1234).is_err());
    assert!(EMeans::fit_with_params(&[vec![1.0], vec![2.0]], 1, usize::MAX, 1234).is_err());
    assert!(EMeans::fit(&[vec![1.0]], 0).is_err());
    for data in [
        vec![],
        vec![vec![]],
        vec![vec![1.0], vec![1.0, 2.0]],
        vec![vec![f64::NAN]],
        vec![vec![f64::INFINITY]],
    ] {
        assert!(EMeans::fit(&data, 2).is_err());
    }
}

#[test]
fn sampled_seeds_preserve_separated_groups() {
    for n in [100, 1000, 10000] {
        let data: Vec<_> = (0..n)
            .map(|i| {
                let group = i % 8;
                vec![
                    group as f64 * 10.0 + (i as f64).sin() * 0.01,
                    (i as f64).cos() * 0.01,
                ]
            })
            .collect();
        let fit = EMeans::fit(&data, 8).unwrap();
        assert!(fit.epochs < 5, "easy groups should not need five passes");
        let labels: std::collections::HashSet<_> = fit.labels[..8].iter().collect();
        assert_eq!(labels.len(), 8);
        for i in 0..n {
            assert_eq!(fit.labels[i], fit.labels[i % 8]);
        }
    }
}

#[test]
fn adaptive_fit_retains_five_pass_quality_or_certifies_an_early_fixed_point() {
    for n in [47, 301, 1000] {
        for dimensions in [2, 3, 17] {
            let data: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..dimensions)
                        .map(|j| ((i * 17 + j * 31) as f64 * 0.37).sin())
                        .collect()
                })
                .collect();
            let adaptive = EMeans::fit(&data, 8).unwrap();
            assert!((1..=25).contains(&adaptive.epochs));
            let loss = |fit: &EMeans| {
                data.iter()
                    .zip(&fit.labels)
                    .map(|(point, &label)| distance(point, &fit.centers[label]))
                    .sum::<f64>()
            };
            if adaptive.epochs >= 5 {
                let fixed = EMeans::fit_with_params(&data, 8, 5, 1234).unwrap();
                assert!(loss(&adaptive) <= loss(&fixed) + 1e-10);
            } else {
                for (point, &label) in data.iter().zip(&adaptive.labels) {
                    let assigned = distance(point, &adaptive.centers[label]);
                    assert!(
                        adaptive
                            .centers
                            .iter()
                            .all(|c| assigned <= distance(point, c) + 1e-12)
                    );
                }
            }
        }
    }
}

#[test]
fn one_cluster_needs_no_ema_passes() {
    for data in [vec![vec![2.0, 3.0]; 20], vec![vec![-1e300], vec![1e300]]] {
        let fit = EMeans::fit(&data, 1).unwrap();
        assert_eq!(fit.epochs, 0);
        assert!(fit.labels.iter().all(|&label| label == 0));
        for axis in 0..data[0].len() {
            let expected = data
                .iter()
                .map(|row| row[axis] / data.len() as f64)
                .sum::<f64>();
            assert!((fit.centers[0][axis] - expected).abs() <= 1e-12 * (1.0 + expected.abs()));
        }
    }
}
