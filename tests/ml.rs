//! ML regressions and checked-in scikit-learn reference results.

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
    let oracle: serde_json::Value = serde_json::from_str(include_str!("assets/ml-oracle.json")).unwrap();
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

mod k_means_tests {
    use super::KMeans;

    fn assert_fixed_point(points: &[Vec<f64>], fit: &KMeans) {
        let mut sums = vec![vec![0.0; points[0].len()]; fit.centers.len()];
        let mut sizes = vec![0; fit.centers.len()];
        for (point, &label) in points.iter().zip(&fit.labels) {
            let distance = |center: &[f64]| {
                point
                    .iter()
                    .zip(center)
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
            };
            assert!(
                fit.centers
                    .iter()
                    .all(|center| distance(&fit.centers[label]) <= distance(center))
            );
            sizes[label] += 1;
            for (sum, x) in sums[label].iter_mut().zip(point) {
                *sum += x;
            }
        }
        for ((sum, n), center) in sums.iter().zip(sizes).zip(&fit.centers) {
            assert!(n > 0);
            for (sum, x) in sum.iter().zip(center) {
                assert!((sum / n as f64 - x).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn separates_2d_and_3d_groups() {
        let xy = vec![
            vec![-0.1, 0.0],
            vec![0.1, 0.0],
            vec![-0.1, 0.0],
            vec![0.1, 0.0],
        ];
        let fit = KMeans::fit(&xy, 2).unwrap();
        assert_eq!(fit.labels[0], fit.labels[2]);
        assert_eq!(fit.labels[1], fit.labels[3]);
        assert_ne!(fit.labels[0], fit.labels[1]);
        let xyz = vec![
            vec![-0.1, 0.0, -1.0],
            vec![0.1, 0.0, -1.0],
            vec![-0.1, 0.0, 1.0],
            vec![0.1, 0.0, 1.0],
        ];
        let fit = KMeans::fit(&xyz, 2).unwrap();
        assert_eq!(fit.labels[0], fit.labels[1]);
        assert_eq!(fit.labels[2], fit.labels[3]);
        assert_ne!(fit.labels[0], fit.labels[2]);
        assert_fixed_point(&xyz, &fit);
    }

    #[test]
    fn fits_all_points_without_stride_aliasing() {
        let points: Vec<_> = (0..4096).map(|i| vec![(i % 2) as f64, 0.0]).collect();
        let fit = KMeans::fit(&points, 2).unwrap();
        assert_eq!(fit.centers.len(), 2);
        for (i, &label) in fit.labels.iter().enumerate() {
            assert_eq!(label, fit.labels[i % 2]);
        }
        assert_ne!(fit.labels[0], fit.labels[1]);
        assert_fixed_point(&points, &fit);
    }

    #[test]
    fn converges_past_the_old_twenty_iteration_limit() {
        let mut seed = 123456789_u64;
        for trial in 0..=5 {
            let points: Vec<_> = (0..1024 + trial)
                .map(|_| {
                    let mut next = || {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (seed >> 32) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0
                    };
                    vec![next(), next()]
                })
                .collect();
            assert_fixed_point(&points, &KMeans::fit(&points, 2 + trial).unwrap());
        }
    }

    #[test]
    fn scaling_does_not_change_the_partition() {
        let points: Vec<_> = [0.0, 0.4, 0.45, 0.49, 0.51, 0.56, 1.0]
            .iter()
            .map(|&x| vec![x])
            .collect();
        let fit = KMeans::fit(&points, 2).unwrap();
        assert_fixed_point(&points, &fit);
        for scale in [1e-200, 1e-6, 1e200] {
            let scaled: Vec<_> = points.iter().map(|p| vec![p[0] * scale]).collect();
            let scaled_fit = KMeans::fit(&scaled, 2).unwrap();
            assert_eq!(fit.labels, scaled_fit.labels);
            for (a, b) in fit.centers.iter().zip(&scaled_fit.centers) {
                assert!((a[0] - b[0] / scale).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn handles_duplicates_and_more_clusters_than_points() {
        let points = vec![vec![2.0, 3.0]; 4];
        let fit = KMeans::fit(&points, 20).unwrap();
        assert_eq!(fit.centers, vec![vec![2.0, 3.0]]);
        assert_eq!(fit.labels, vec![0; 4]);
        let points = vec![vec![0.0], vec![1.0], vec![1.0], vec![2.0]];
        let fit = KMeans::fit(&points, 20).unwrap();
        assert_eq!(fit.centers.len(), 3);
        assert_fixed_point(&points, &fit);
    }

    #[test]
    fn keeps_distinct_clusters_occupied() {
        let points = vec![vec![0.0], vec![0.0], vec![2.0], vec![4.0]];
        let fit = KMeans::fit(&points, 3).unwrap();
        assert_eq!(fit.centers.len(), 3);
        assert_fixed_point(&points, &fit);
    }

    #[test]
    fn rejects_invalid_input() {
        assert!(KMeans::fit(&[vec![1.0]], 0).is_err());
        for points in [
            vec![],
            vec![vec![]],
            vec![vec![1.0], vec![1.0, 2.0]],
            vec![vec![f64::NAN]],
            vec![vec![f64::INFINITY]],
        ] {
            assert!(KMeans::fit(&points, 2).is_err());
        }
    }
}

mod pca_tests {
    use super::Pca;

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn finds_largest_variance_even_if_first_row_is_on_a_minor_axis() {
        let data = vec![
            vec![0.0, 1.0],
            vec![0.0, -1.0],
            vec![4.0, 0.0],
            vec![-4.0, 0.0],
        ];
        let fit = Pca::fit(&data, 2).unwrap();
        assert!(fit.components[0][0].abs() > 1.0 - 1e-12);
        assert!(fit.components[1][1].abs() > 1.0 - 1e-12);
        let projected: Vec<_> = data.iter().map(|row| fit.transform(row).unwrap()).collect();
        let variance = |i: usize| projected.iter().map(|row| row[i].powi(2)).sum::<f64>();
        assert!((variance(0) - 32.0).abs() < 1e-12);
        assert!((variance(1) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn centered_orthogonal_components_reconstruct_full_rank_data() {
        let data = vec![
            vec![2.0, 1.0, 5.0],
            vec![4.0, 2.0, 0.0],
            vec![-1.0, 0.0, 2.0],
            vec![1.0, -3.0, 3.0],
        ];
        let fit = Pca::fit(&data, 3).unwrap();
        let projected: Vec<_> = data.iter().map(|row| fit.transform(row).unwrap()).collect();
        for i in 0..3 {
            assert!(projected.iter().map(|row| row[i]).sum::<f64>().abs() < 1e-12);
            for j in 0..3 {
                assert!(
                    (dot(&fit.components[i], &fit.components[j]) - if i == j { 1.0 } else { 0.0 })
                        .abs()
                        < 1e-12
                );
            }
        }
        for (row, scores) in data.iter().zip(projected) {
            for j in 0..3 {
                let reconstructed = fit.mean[j]
                    + scores
                        .iter()
                        .zip(&fit.components)
                        .map(|(score, axis)| score * axis[j])
                        .sum::<f64>();
                assert!((row[j] - reconstructed).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn uses_all_rows_instead_of_a_stride_sample() {
        let data: Vec<_> = (0..1024)
            .map(|i| vec![if i % 2 == 0 { 0.0 } else { 2.0 }, 0.0])
            .collect();
        let fit = Pca::fit(&data, 2).unwrap();
        assert_eq!(fit.mean, vec![1.0, 0.0]);
        assert!(fit.components[0][0].abs() > 1.0 - 1e-12);
        assert_eq!(fit.components[1], vec![0.0, 0.0]);
    }

    #[test]
    fn handles_constant_singleton_and_rank_deficient_data() {
        for data in [vec![vec![2.0, 3.0]], vec![vec![2.0, 3.0]; 4]] {
            let fit = Pca::fit(&data, 3).unwrap();
            assert_eq!(fit.components, vec![vec![0.0, 0.0]; 2]);
            assert_eq!(fit.transform(&data[0]).unwrap(), vec![0.0; 2]);
        }
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        let fit = Pca::fit(&data, 3).unwrap();
        assert!((dot(&fit.components[0], &fit.components[0]) - 1.0).abs() < 1e-12);
        assert_eq!(fit.components[1], vec![0.0; 3]);
        assert_eq!(fit.components[2], vec![0.0; 3]);
    }

    #[test]
    fn scaling_and_translation_preserve_principal_directions() {
        let data = vec![
            vec![0.0, 1.0],
            vec![0.0, -1.0],
            vec![4.0, 0.0],
            vec![-4.0, 0.0],
        ];
        let fit = Pca::fit(&data, 2).unwrap();
        for scale in [1e-200, 1e200] {
            let scaled: Vec<_> = data
                .iter()
                .map(|row| row.iter().map(|x| (x + 5.0) * scale).collect())
                .collect();
            let transformed = Pca::fit(&scaled, 2).unwrap();
            for (a, b) in fit.components.iter().zip(&transformed.components) {
                assert!((dot(a, b).abs() - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn rejects_invalid_input_and_projection() {
        assert!(Pca::fit(&[vec![1.0]], 0).is_err());
        for points in [
            vec![],
            vec![vec![]],
            vec![vec![1.0], vec![1.0, 2.0]],
            vec![vec![f64::NAN]],
            vec![vec![f64::INFINITY]],
        ] {
            assert!(Pca::fit(&points, 2).is_err());
        }
        let fit = Pca::fit(&[vec![1.0, 2.0]], 2).unwrap();
        assert!(fit.transform(&[1.0]).is_err());
        assert!(fit.transform(&[f64::NAN, 1.0]).is_err());
    }
}
