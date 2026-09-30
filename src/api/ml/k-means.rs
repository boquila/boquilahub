//! Deterministic Lloyd k-means using every observation, in any dimension.

use anyhow::{Result, bail, ensure};

#[derive(Debug)]
pub struct KMeans {
    pub centers: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
}

impl KMeans {
    /// Farthest-first seeds, then nearest-center assignment and mean updates.
    /// Stops when labels are unchanged; errors rather than returning an unfinished fit.
    /// Uses at most `count` centers (fewer when there are fewer distinct points).
    /// Like other Lloyd implementations, this finds a local minimum.
    pub fn fit(data: &[Vec<f64>], count: usize) -> Result<Self> {
        ensure!(count > 0, "cluster count must be positive");
        super::validate(data)?;
        let scale = super::scale(data);
        let points: Vec<Vec<f64>> = data
            .iter()
            .map(|row| row.iter().map(|x| x / scale).collect())
            .collect();
        let mut centers = seed_centers(&points, count.min(points.len()));
        let mut labels = vec![usize::MAX; points.len()];
        for _ in 0..1000 {
            let previous = labels.clone();
            for (point, label) in points.iter().zip(&mut labels) {
                *label = nearest(point, &centers, *label);
            }
            centers = update_centers(&points, &mut labels, centers.len());
            if labels == previous {
                for center in &mut centers {
                    for value in center {
                        *value *= scale;
                    }
                }
                return Ok(Self { centers, labels });
            }
        }
        bail!("k-means did not converge after 1000 iterations")
    }
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn nearest(point: &[f64], centers: &[Vec<f64>], previous: usize) -> usize {
    // Keep the previous label on exact ties, so duplicates cannot oscillate.
    let mut best = if previous < centers.len() {
        previous
    } else {
        0
    };
    let mut distance = squared_distance(point, &centers[best]);
    for (index, center) in centers.iter().enumerate() {
        if index == best {
            continue;
        }
        let candidate = squared_distance(point, center);
        if candidate < distance {
            best = index;
            distance = candidate;
        }
    }
    best
}

fn seed_centers(points: &[Vec<f64>], count: usize) -> Vec<Vec<f64>> {
    let mut centers = vec![points[0].clone()];
    let mut distances = vec![f64::INFINITY; points.len()];
    while centers.len() < count {
        let last = centers.last().unwrap();
        for (distance, point) in distances.iter_mut().zip(points) {
            *distance = distance.min(squared_distance(point, last));
        }
        let (index, &distance) = distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        if distance == 0.0 {
            break;
        }
        centers.push(points[index].clone());
    }
    centers
}

fn update_centers(points: &[Vec<f64>], labels: &mut [usize], count: usize) -> Vec<Vec<f64>> {
    let mut sums = vec![vec![0.0; points[0].len()]; count];
    let mut sizes = vec![0usize; count];
    for (point, &label) in points.iter().zip(labels.iter()) {
        sizes[label] += 1;
        for (sum, value) in sums[label].iter_mut().zip(point) {
            *sum += value;
        }
    }
    // Give an empty cluster the worst-fitting point from a cluster with spare points.
    for empty in 0..count {
        if sizes[empty] != 0 {
            continue;
        }
        let error = |index: usize| {
            let label = labels[index];
            points[index]
                .iter()
                .zip(&sums[label])
                .map(|(x, sum)| (x - sum / sizes[label] as f64).powi(2))
                .sum::<f64>()
        };
        let index = (0..points.len())
            .filter(|&i| sizes[labels[i]] > 1)
            .max_by(|&a, &b| error(a).total_cmp(&error(b)))
            .unwrap();
        let donor = labels[index];
        sizes[donor] -= 1;
        sizes[empty] = 1;
        for (sum, value) in sums[donor].iter_mut().zip(&points[index]) {
            *sum -= value;
        }
        sums[empty].clone_from(&points[index]);
        labels[index] = empty;
    }
    sums.into_iter()
        .zip(sizes)
        .map(|(sum, n)| sum.into_iter().map(|x| x / n as f64).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_fixed_point(points: &[Vec<f64>], fit: &KMeans) {
        let mut sums = vec![vec![0.0; points[0].len()]; fit.centers.len()];
        let mut sizes = vec![0; fit.centers.len()];
        for (point, &label) in points.iter().zip(&fit.labels) {
            assert_eq!(nearest(point, &fit.centers, label), label);
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
    fn repairs_empty_clusters() {
        let points = vec![vec![0.0], vec![0.0], vec![2.0], vec![4.0]];
        let mut labels = vec![0, 0, 0, 0];
        let centers = update_centers(&points, &mut labels, 3);
        assert_eq!(centers.len(), 3);
        assert!(centers.iter().flatten().all(|x| x.is_finite()));
        for id in 0..3 {
            assert!(labels.contains(&id));
        }
        let tied = vec![vec![0.0], vec![0.0]];
        assert_eq!(nearest(&[0.0], &tied, 1), 1);
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
