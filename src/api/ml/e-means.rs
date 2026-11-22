//! CPU e-means. Reference and adaptations: tests/assets/e-means.md.

use super::k_means::seed_centers;
use anyhow::{Result, ensure};

// Five-pass calibration; adaptive stopping up to the cap.
const SCHEDULE_EPOCHS: usize = 5;
const MAX_EPOCHS: usize = 25;
const MIN_RELATIVE_GAIN: f64 = 1e-3;

#[derive(Debug)]
pub struct EMeans {
    pub centers: Vec<Vec<f64>>,
    /// Assignments used for the returned closing means; not a new nearest pass.
    pub labels: Vec<usize>,
    /// Full EMA passes executed (zero for an exact single-cluster mean).
    pub epochs: usize,
}

impl EMeans {
    /// Retain the best fit until assignments stabilize or error improvement stalls.
    pub fn fit(data: &[Vec<f64>], count: usize) -> Result<Self> {
        Self::run(data, count, MAX_EPOCHS, 1234, true)
    }

    /// Fit with an exact pass budget and seed.
    pub fn fit_with_params(
        data: &[Vec<f64>],
        count: usize,
        epochs: usize,
        seed: u64,
    ) -> Result<Self> {
        Self::run(data, count, epochs, seed, false)
    }

    fn run(
        data: &[Vec<f64>],
        count: usize,
        epochs: usize,
        seed: u64,
        adaptive: bool,
    ) -> Result<Self> {
        ensure!(count > 0, "cluster count must be positive");
        ensure!(epochs > 0, "epoch count must be positive");
        let dimensions = super::validate(data)?;
        let scale = super::scale(data);
        let points: Vec<f64> = data.iter().flatten().map(|x| x / scale).collect();
        let mut order: Vec<_> = (0..data.len()).collect();
        let mut rng = Random(seed);
        rng.shuffle(&mut order);
        // Sampled farthest-first seeds avoid a full O(nkd) initialization.
        let pool: Vec<_> = order[..count.saturating_mul(32).min(data.len())]
            .iter()
            .map(|&i| points[i * dimensions..(i + 1) * dimensions].to_vec())
            .collect();
        let mut centers = seed_centers(&pool, count.min(pool.len()));
        drop(pool);
        let k = centers.len();
        let n = data.len();
        if adaptive && k == 1 {
            let mut mean = vec![0.0; dimensions];
            for point in points.chunks_exact(dimensions) {
                for (sum, x) in mean.iter_mut().zip(point) {
                    *sum += x;
                }
            }
            mean.iter_mut().for_each(|x| *x = (*x / n as f64) * scale);
            return Ok(Self {
                centers: vec![mean],
                labels: vec![0; n],
                epochs: 0,
            });
        }
        let horizon = if adaptive { SCHEDULE_EPOCHS } else { epochs };
        let (batch_size, alpha0) = recipe(n, k, dimensions, horizon);
        let total_steps = horizon
            .checked_mul(n.div_ceil(batch_size))
            .ok_or_else(|| anyhow::anyhow!("epoch budget is too large"))?;
        let mut mass = vec![2.0_f64.powi(-16); k];
        let mut sums = vec![vec![0.0; dimensions]; k];
        let mut counts = vec![0usize; k];
        let mut closing_sums = sums.clone();
        let mut closing_counts = counts.clone();
        let mut labels = vec![usize::MAX; n];
        for (i, value) in order.iter_mut().enumerate() {
            *value = i;
        }
        let mut split_rng = Random(seed);
        let mut step = 0;
        let mut best = None;
        let mut best_loss = f64::INFINITY;
        let mut stalled = 0;
        let mut completed = 0;
        for epoch in 0..epochs {
            let closing = adaptive || epoch + 1 == epochs;
            let mut changed = false;
            if adaptive {
                closing_counts.fill(0);
                closing_sums.iter_mut().for_each(|sum| sum.fill(0.0));
            }
            rng.shuffle(&mut order);
            for batch in order.chunks(batch_size) {
                counts.fill(0);
                sums.iter_mut().for_each(|sum| sum.fill(0.0));
                for &i in batch {
                    let point = &points[i * dimensions..(i + 1) * dimensions];
                    let label = nearest(point, &centers);
                    counts[label] += 1;
                    for (sum, x) in sums[label].iter_mut().zip(point) {
                        *sum += x;
                    }
                    if closing {
                        changed |= labels[i] != label;
                        labels[i] = label;
                    }
                }
                let phase =
                    std::f64::consts::PI * step.min(total_steps) as f64 / total_steps as f64;
                let alpha = alpha0 * (0.1 + 0.45 * (1.0 + phase.cos()));
                for j in 0..k {
                    // Faiss retains unoccupied masses on the alpha == 1 step.
                    if alpha != 1.0 {
                        mass[j] *= 1.0 - alpha;
                    }
                    if counts[j] == 0 {
                        continue;
                    }
                    let added = alpha * counts[j] as f64;
                    mass[j] = if alpha == 1.0 { added } else { mass[j] + added };
                    let weight = added / mass[j];
                    for (center, sum) in centers[j].iter_mut().zip(&sums[j]) {
                        let mean = sum / counts[j] as f64;
                        *center = if weight == 1.0 {
                            mean
                        } else {
                            *center + weight * (mean - *center)
                        };
                    }
                    if closing {
                        closing_counts[j] += counts[j];
                        for (total, sum) in closing_sums[j].iter_mut().zip(&sums[j]) {
                            *total += sum;
                        }
                    }
                }
                step += 1;
                // Tiny batches underestimate mass and can destroy valid groups.
                if batch_size >= k && step % 5 == 0 {
                    revive(&mut centers, &mut mass, &mut split_rng);
                }
            }
            completed = epoch + 1;
            if adaptive {
                let mut candidate = centers.clone();
                close_centers(&mut candidate, &closing_sums, &closing_counts);
                let loss: f64 = points
                    .chunks_exact(dimensions)
                    .zip(&labels)
                    .map(|(point, &label)| {
                        point
                            .iter()
                            .zip(&candidate[label])
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                    })
                    .sum();
                // Certify assignments against closing means before stopping early.
                let fixed = !changed
                    && points
                        .chunks_exact(dimensions)
                        .zip(&labels)
                        .all(|(point, &label)| nearest(point, &candidate) == label);
                let certified_best = fixed && loss <= best_loss;
                if best_loss - loss > best_loss * MIN_RELATIVE_GAIN || !best_loss.is_finite() {
                    stalled = 0;
                } else {
                    stalled += 1;
                }
                if loss < best_loss || certified_best {
                    best_loss = loss;
                    best = Some((candidate, labels.clone()));
                }
                if certified_best || (completed >= horizon && stalled >= 2) {
                    break;
                }
            }
        }
        if adaptive {
            (centers, labels) = best.unwrap();
        } else {
            close_centers(&mut centers, &closing_sums, &closing_counts);
        }
        centers.iter_mut().flatten().for_each(|x| *x *= scale);
        Ok(Self {
            centers,
            labels,
            epochs: completed,
        })
    }
}

fn close_centers(centers: &mut [Vec<f64>], sums: &[Vec<f64>], counts: &[usize]) {
    for ((center, sum), &count) in centers.iter_mut().zip(sums).zip(counts) {
        if count > 0 {
            for (value, sum) in center.iter_mut().zip(sum) {
                *value = sum / count as f64;
            }
        }
    }
}

fn recipe(n: usize, k: usize, dimensions: usize, epochs: usize) -> (usize, f64) {
    let quality = dimensions <= 16;
    let coefficient = if quality { 0.00439175 } else { 0.035134 };
    let round = |x: f64| 2.0_f64.powf(x.max(1.0).log2().round()) as usize;
    let raw = coefficient
        * (k as f64).powf(0.134749)
        * (n as f64).powf(0.876473)
        * (epochs as f64).powf(0.813344);
    let cap = if quality {
        n / 2
    } else {
        round(n as f64 / 2.0)
    };
    let batch = round(raw).min(cap).clamp(1, n);
    let alpha = (0.8244
        * (k as f64).powf(-0.4617)
        * (batch as f64).powf(0.9003)
        * (n as f64).powf(-0.4498)
        * (epochs as f64).powf(-0.1057))
    .clamp(1e-4, 1.0);
    (batch, alpha)
}

fn nearest(point: &[f64], centers: &[Vec<f64>]) -> usize {
    let mut best = 0;
    let mut minimum = f64::INFINITY;
    for (j, center) in centers.iter().enumerate() {
        let distance = match point {
            [x, y] => (x - center[0]).powi(2) + (y - center[1]).powi(2),
            [x, y, z] => {
                (x - center[0]).powi(2) + (y - center[1]).powi(2) + (z - center[2]).powi(2)
            }
            _ => squared_distance(point, center),
        };
        if distance < minimum {
            minimum = distance;
            best = j;
        }
    }
    best
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn shuffle(&mut self, order: &mut [usize]) {
        for i in (1..order.len()).rev() {
            order.swap(i, ((self.next() >> 32) % (i as u64 + 1)) as usize);
        }
    }

    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64).max(f64::MIN_POSITIVE)
    }
}

// Faiss: Gumbel keys sample donors without replacement, weighted by mass.
fn revive(centers: &mut [Vec<f64>], mass: &mut [f64], rng: &mut Random) {
    let threshold = 0.1 * mass.iter().sum::<f64>() / mass.len() as f64;
    let unused: Vec<_> = (0..mass.len()).filter(|&j| mass[j] < threshold).collect();
    if unused.is_empty() {
        return;
    }
    let mut donors: Vec<_> = mass
        .iter()
        .enumerate()
        .filter_map(|(j, &m)| {
            let uniform = rng.uniform();
            (m >= threshold && m > 0.0).then(|| (j, m.ln() - (-uniform.ln()).ln()))
        })
        .collect();
    donors.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
    for (&target, &(source, _)) in unused.iter().zip(&donors) {
        mass[source] *= 0.5;
        mass[target] = mass[source];
        for axis in 0..centers[source].len() {
            let value = centers[source][axis];
            let offset = if axis % 2 == 0 { 1e-4 } else { -1e-4 };
            centers[source][axis] = value * (1.0 - offset);
            centers[target][axis] = value * (1.0 + offset);
        }
        if centers[source] == centers[target] {
            centers[target][0] = centers[target][0].next_up();
        }
    }
}

#[inline]
fn squared_distance(point: &[f64], center: &[f64]) -> f64 {
    // Independent lanes allow SIMD while preserving f64 precision.
    let mut lanes = [0.0; 4];
    let mut p = point.chunks_exact(4);
    let mut c = center.chunks_exact(4);
    for (p, c) in p.by_ref().zip(c.by_ref()) {
        for axis in 0..4 {
            lanes[axis] += (p[axis] - c[axis]).powi(2);
        }
    }
    lanes.iter().sum::<f64>()
        + p.remainder()
            .iter()
            .zip(c.remainder())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
}
