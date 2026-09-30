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
