//! Small, CPU-only methods. Rows are observations; columns are features.

#[path = "k-means.rs"]
pub mod k_means;
pub mod pca;

pub use k_means::KMeans;
pub use pca::Pca;

use anyhow::{Result, ensure};

fn validate(data: &[Vec<f64>]) -> Result<usize> {
    ensure!(!data.is_empty(), "data must contain at least one row");
    let dimensions = data[0].len();
    ensure!(dimensions > 0, "rows must contain at least one feature");
    for row in data {
        ensure!(row.len() == dimensions, "rows must have the same length");
        ensure!(row.iter().all(|x| x.is_finite()), "values must be finite");
    }
    Ok(dimensions)
}

// One common scale preserves Euclidean geometry and avoids squaring huge values.
fn scale(data: &[Vec<f64>]) -> f64 {
    let maximum = data.iter().flatten().map(|x| x.abs()).fold(0.0, f64::max);
    if maximum == 0.0 { 1.0 } else { maximum }
}
