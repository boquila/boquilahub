//! Centered PCA using all rows and the right singular vectors of the data matrix.

use anyhow::{Context, Result, ensure};
use nalgebra::{DMatrix, linalg::SVD};

#[derive(Debug)]
pub struct Pca {
    pub mean: Vec<f64>,
    /// Unit directions, ordered by decreasing variance; zero for missing rank.
    pub components: Vec<Vec<f64>>,
}

impl Pca {
    /// Fits up to `count` components without whitening or per-feature scaling.
    /// Rank-deficient data gets zero components for the remaining directions.
    pub fn fit(data: &[Vec<f64>], count: usize) -> Result<Self> {
        ensure!(count > 0, "component count must be positive");
        let dimensions = super::validate(data)?;
        let scale = super::scale(data);
        let mut mean = vec![0.0; dimensions];
        for (i, row) in data.iter().enumerate() {
            for (avg, value) in mean.iter_mut().zip(row) {
                *avg += (value / scale - *avg) / (i + 1) as f64;
            }
        }
        let centered =
            DMatrix::from_fn(data.len(), dimensions, |i, j| data[i][j] / scale - mean[j]);
        let mut components = vec![vec![0.0; dimensions]; count.min(dimensions)];
        if centered.iter().any(|&x| x != 0.0) {
            let svd = SVD::try_new(centered, false, true, f64::EPSILON, 10000)
                .context("PCA singular value decomposition did not converge")?;
            let tolerance =
                svd.singular_values[0] * f64::EPSILON * data.len().max(dimensions) as f64;
            let directions = svd
                .v_t
                .context("PCA did not compute component directions")?;
            for (i, component) in components.iter_mut().enumerate() {
                if i >= svd.singular_values.len() || svd.singular_values[i] <= tolerance {
                    break;
                }
                for (j, value) in component.iter_mut().enumerate() {
                    *value = directions[(i, j)];
                }
                // Fix the otherwise arbitrary sign for reproducible display coordinates.
                let pivot = component
                    .iter()
                    .max_by(|a, b| a.abs().total_cmp(&b.abs()))
                    .unwrap();
                if *pivot < 0.0 {
                    for value in component {
                        *value = -*value;
                    }
                }
            }
        }
        for value in &mut mean {
            *value *= scale;
        }
        Ok(Self { mean, components })
    }

    /// Projects one observation using the fitted mean and component directions.
    pub fn transform(&self, row: &[f64]) -> Result<Vec<f64>> {
        ensure!(
            row.len() == self.mean.len(),
            "observation has the wrong number of features"
        );
        ensure!(row.iter().all(|x| x.is_finite()), "values must be finite");
        let projected: Vec<_> = self
            .components
            .iter()
            .map(|component| {
                row.iter()
                    .zip(&self.mean)
                    .zip(component)
                    .map(|((value, avg), direction)| (value - avg) * direction)
                    .sum::<f64>()
            })
            .collect();
        ensure!(
            projected.iter().all(|x| x.is_finite()),
            "PCA projection overflowed"
        );
        Ok(projected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
