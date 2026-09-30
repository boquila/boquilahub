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
