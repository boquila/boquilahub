//! Python is only needed to regenerate tests/oracles/ml.json, not to run these tests.

use boquilahub::api::ml::{KMeans, Pca};
use serde::{Deserialize, de::DeserializeOwned};

#[derive(Deserialize)]
struct PcaCase {
    name: String,
    data: Vec<Vec<f64>>,
    count: usize,
    mean: Vec<f64>,
    components: Vec<Vec<f64>>,
    variance: Vec<f64>,
    scores: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct KMeansCase {
    name: String,
    data: Vec<Vec<f64>>,
    count: usize,
    centers: Vec<Vec<f64>>,
    labels: Vec<usize>,
    inertia: f64,
}

fn cases<T: DeserializeOwned>(method: &str) -> Vec<T> {
    let oracle: serde_json::Value = serde_json::from_str(include_str!("oracles/ml.json")).unwrap();
    serde_json::from_value(oracle[method].clone()).unwrap()
}

fn scale(data: &[Vec<f64>]) -> f64 {
    let max = data.iter().flatten().map(|x| x.abs()).fold(0.0, f64::max);
    if max == 0.0 { 1.0 } else { max }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn close(actual: f64, expected: f64, name: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 * (1.0 + expected.abs()),
        "{name}: actual={actual}, expected={expected}",
    );
}

#[test]
fn pca_matches_sklearn_full_svd() {
    for case in cases::<PcaCase>("pca") {
        let fit = Pca::fit(&case.data, case.count).unwrap();
        let factor = scale(&case.data);
        assert_eq!(fit.components.len(), case.components.len(), "{}", case.name);
        for (actual, expected) in fit.mean.iter().zip(&case.mean) {
            close(actual / factor, expected / factor, &case.name);
        }
        // Compare the retained subspace: signs and bases within repeated eigenvalues
        // are arbitrary, so individual component vectors need not be identical.
        for i in 0..fit.mean.len() {
            for j in 0..fit.mean.len() {
                let actual: f64 = fit.components.iter().map(|axis| axis[i] * axis[j]).sum();
                let expected: f64 = case.components.iter().map(|axis| axis[i] * axis[j]).sum();
                close(actual, expected, &case.name);
            }
        }
        let rotation: Vec<Vec<f64>> = fit
            .components
            .iter()
            .map(|actual| {
                case.components
                    .iter()
                    .map(|expected| dot(actual, expected))
                    .collect()
            })
            .collect();
        let scores: Vec<_> = case
            .data
            .iter()
            .map(|row| fit.transform(row).unwrap())
            .collect();
        for (actual, expected) in scores.iter().zip(&case.scores) {
            for (i, value) in actual.iter().enumerate() {
                close(
                    value / factor,
                    dot(&rotation[i], expected) / factor,
                    &case.name,
                );
            }
        }
        for i in 0..fit.components.len() {
            let variance = if scores.len() > 1 {
                scores
                    .iter()
                    .map(|row| (row[i] / factor).powi(2))
                    .sum::<f64>()
                    / (scores.len() - 1) as f64
            } else {
                0.0
            };
            close(variance, case.variance[i], &case.name);
            for j in 0..i {
                close(dot(&fit.components[i], &fit.components[j]), 0.0, &case.name);
            }
        }
    }
}

#[test]
fn kmeans_matches_sklearn_with_the_same_seeds() {
    for case in cases::<KMeansCase>("kmeans") {
        let fit = KMeans::fit(&case.data, case.count).unwrap();
        let factor = scale(&case.data);
        assert_eq!(fit.centers.len(), case.centers.len(), "{}", case.name);
        let distance = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x / factor - y / factor).powi(2))
                .sum::<f64>()
        };
        // Match centers before comparing labels: cluster numbers have no meaning.
        let mapping: Vec<_> = fit
            .centers
            .iter()
            .map(|center| {
                case.centers
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| distance(center, a).total_cmp(&distance(center, b)))
                    .unwrap()
                    .0
            })
            .collect();
        let mut unique = mapping.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            mapping.len(),
            "{}: center mapping is not one-to-one",
            case.name
        );
        for (actual, &index) in fit.centers.iter().zip(&mapping) {
            for (actual, expected) in actual.iter().zip(&case.centers[index]) {
                close(actual / factor, expected / factor, &case.name);
            }
        }
        let actual_labels: Vec<_> = fit.labels.iter().map(|&id| mapping[id]).collect();
        assert_eq!(actual_labels, case.labels, "{}", case.name);
        let inertia: f64 = case
            .data
            .iter()
            .zip(&fit.labels)
            .map(|(point, &id)| distance(point, &fit.centers[id]))
            .sum();
        close(inertia, case.inertia, &case.name);
    }
}
