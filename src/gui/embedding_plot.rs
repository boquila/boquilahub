use super::Gui;
use crate::api::abstractions::{AIOutputs, Embedding, PredImg, XYXY};
use crate::localization::{Key, translate};
use egui_plot::{Plot, PlotPoint, Points};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

pub(super) struct EmbeddingPlot {
    pub open: bool,
    points: Vec<PlotPoint>,
    cloud_3d: super::embedding_plot_3d::Cloud3D,
    view_3d: bool,
    indexes: Vec<usize>,
    boxes: Vec<Option<XYXY>>,
    labels: Vec<String>,
    objects: bool,
    model: String,
    color_clusters: bool,
    cluster_count: usize,
    computed_cluster_count: usize,
    clusters: Vec<Vec<PlotPoint>>,
    cluster_ids: Vec<usize>,
    hovered: Option<usize>,
    preview_point: Option<usize>,
    selected_point: Option<usize>,
    highlight_selection: bool,
    thumbnails: VecDeque<(usize, Option<egui::TextureHandle>)>,
    pending: Vec<usize>,
    requests: Sender<(usize, PathBuf, Option<XYXY>)>,
    results: Receiver<(usize, Option<egui::ColorImage>)>,
}

impl EmbeddingPlot {
    pub fn new(files: &[PredImg]) -> Self {
        let has_images = files.iter().any(|file| matches!(&file.aioutput, Some(AIOutputs::Embed(embedding)) if valid_embedding(embedding)));
        Self::for_kind(files, !has_images)
    }

    fn for_kind(files: &[PredImg], objects: bool) -> Self {
        let mut entries: Vec<(usize, Option<XYXY>, String, &Embedding)> = Vec::new();
        for (index, file) in files.iter().enumerate() {
            match file.aioutput.as_ref() {
                Some(AIOutputs::Embed(embedding)) if !objects => {
                    entries.push((index, None, String::new(), embedding));
                }
                Some(AIOutputs::ObjectDetection(boxes)) if objects => {
                    for bbox in boxes {
                        if let Some(embedding) = &bbox.embedding {
                            entries.push((index, Some(bbox.xyxy), bbox.label.clone(), embedding));
                        }
                    }
                }
                Some(AIOutputs::Segmentation(segments)) if objects => {
                    for segment in segments {
                        if let Some(embedding) = &segment.bbox.embedding {
                            entries.push((
                                index,
                                Some(segment.bbox.xyxy),
                                segment.bbox.label.clone(),
                                embedding,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        entries.retain(|(_, _, _, embedding)| valid_embedding(embedding));
        let (model, indexes, boxes, labels, points, points_3d) =
            if let Some((_, _, _, first)) = entries.first() {
                let model = first.model.clone();
                let dimensions = first.values.len();
                let entries: Vec<_> = entries
                    .into_iter()
                    .filter(|(_, _, _, emb)| emb.model == model && emb.values.len() == dimensions)
                    .collect();
                let indexes = entries.iter().map(|(index, _, _, _)| *index).collect();
                let boxes = entries.iter().map(|(_, bbox, _, _)| *bbox).collect();
                let labels = entries
                    .iter()
                    .map(|(_, _, label, _)| label.clone())
                    .collect();
                let embeddings: Vec<_> = entries
                    .iter()
                    .map(|(index, _, _, emb)| (*index, *emb))
                    .collect();
                let (points, points_3d) = project(&embeddings);
                (model, indexes, boxes, labels, points, points_3d)
            } else {
                (
                    String::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            };
        let (requests, incoming) = mpsc::channel::<(usize, PathBuf, Option<XYXY>)>();
        let (outgoing, results) = mpsc::channel();
        std::thread::spawn(move || {
            for (index, path, bbox) in incoming {
                let thumbnail = image::open(path).ok().and_then(|image| {
                    let image = if let Some(bbox) = bbox {
                        let x1 = (bbox.x1.max(0.0) as u32).min(image.width());
                        let y1 = (bbox.y1.max(0.0) as u32).min(image.height());
                        let x2 = (bbox.x2.max(0.0) as u32).min(image.width());
                        let y2 = (bbox.y2.max(0.0) as u32).min(image.height());
                        if x2 <= x1 || y2 <= y1 {
                            None
                        } else {
                            Some(image.crop_imm(x1, y1, x2 - x1, y2 - y1))
                        }
                    } else {
                        Some(image)
                    }?;
                    let image = image.thumbnail(320, 240).to_rgba8();
                    Some(egui::ColorImage::from_rgba_unmultiplied(
                        [image.width() as usize, image.height() as usize],
                        image.as_raw(),
                    ))
                });
                if outgoing.send((index, thumbnail)).is_err() {
                    break;
                }
            }
        });
        Self {
            open: true,
            points,
            cloud_3d: super::embedding_plot_3d::Cloud3D::new(points_3d),
            view_3d: false,
            indexes,
            boxes,
            labels,
            objects,
            model,
            color_clusters: false,
            cluster_count: 8,
            computed_cluster_count: 0,
            clusters: Vec::new(),
            cluster_ids: Vec::new(),
            hovered: None,
            preview_point: None,
            selected_point: None,
            highlight_selection: true,
            thumbnails: VecDeque::new(),
            pending: Vec::new(),
            requests,
            results,
        }
    }

    fn preview(&mut self, ctx: &egui::Context, point_index: usize, path: &Path) {
        self.preview_point = Some(point_index);
        while let Ok((index, pixels)) = self.results.try_recv() {
            self.pending.retain(|pending| *pending != index);
            let texture = pixels.map(|pixels| {
                ctx.load_texture("embedding_preview", pixels, egui::TextureOptions::LINEAR)
            });
            self.thumbnails.push_back((index, texture));
            if self.thumbnails.len() > 16 {
                self.thumbnails.pop_front();
            }
        }
        if !self
            .thumbnails
            .iter()
            .any(|(index, _)| *index == point_index)
            && !self.pending.contains(&point_index)
            && self.pending.len() < 2
            && self
                .requests
                .send((point_index, path.to_path_buf(), self.boxes[point_index]))
                .is_ok()
        {
            self.pending.push(point_index);
        }
        if !self.pending.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }

    fn recluster(&mut self) {
        self.cluster_ids = k_means_3d(self.cloud_3d.positions(), self.cluster_count);
        self.clusters = vec![Vec::new(); self.cluster_ids.iter().max().map_or(0, |id| id + 1)];
        for (&point, &cluster) in self.points.iter().zip(&self.cluster_ids) {
            self.clusters[cluster].push(point);
        }
        self.cloud_3d
            .set_clusters(&self.cluster_ids, self.clusters.len());
        self.computed_cluster_count = self.cluster_count;
    }
}

fn valid_embedding(embedding: &Embedding) -> bool {
    !embedding.values.is_empty() && embedding.values.iter().all(|value| value.is_finite())
}

// Fit on a bounded sample; assign every displayed point to the nearest centre.
fn k_means_3d(points: &[[f32; 3]], count: usize) -> Vec<usize> {
    if points.is_empty() {
        return Vec::new();
    }
    let stride = points.len().div_ceil(2048).max(1);
    let sample: Vec<_> = points.iter().step_by(stride).copied().collect();
    let distance = |a: [f32; 3], b: [f32; 3]| {
        a.into_iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
    };
    let mut centers = vec![sample[0]];
    while centers.len() < count.min(sample.len()) {
        let next = sample
            .iter()
            .copied()
            .max_by(|a, b| {
                let nearest = |point: [f32; 3]| {
                    centers
                        .iter()
                        .map(|&center| distance(point, center))
                        .fold(f32::INFINITY, f32::min)
                };
                nearest(*a).total_cmp(&nearest(*b))
            })
            .unwrap();
        if centers
            .iter()
            .any(|&center| distance(next, center) <= 1e-20)
        {
            break;
        }
        centers.push(next);
    }
    let nearest = |point: [f32; 3], centers: &[[f32; 3]]| {
        centers
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| distance(point, **a).total_cmp(&distance(point, **b)))
            .unwrap()
            .0
    };
    for _ in 0..20 {
        let mut sums = vec![([0.0_f32; 3], 0usize); centers.len()];
        for &point in &sample {
            let (sum, n) = &mut sums[nearest(point, &centers)];
            for (total, value) in sum.iter_mut().zip(point) {
                *total += value;
            }
            *n += 1;
        }
        let mut movement = 0.0_f32;
        for (center, (sum, n)) in centers.iter_mut().zip(sums) {
            if n > 0 {
                let next = sum.map(|value| value / n as f32);
                movement = movement.max(distance(*center, next));
                *center = next;
            }
        }
        if movement < 1e-8 {
            break;
        }
    }
    points
        .iter()
        .map(|&point| nearest(point, &centers))
        .collect()
}

pub(super) fn cluster_color(index: usize, count: usize) -> egui::Color32 {
    egui::ecolor::Hsva::new((index as f32 / count as f32 + 0.04) % 1.0, 0.78, 0.85, 1.0).into()
}

// Three principal components, fitted on at most 512 evenly spaced images.
// Projection of the full collection stays linear in image count and dimensions.
fn project(data: &[(usize, &Embedding)]) -> (Vec<PlotPoint>, Vec<[f32; 3]>) {
    let dimensions = data[0].1.values.len();
    let stride = data.len().div_ceil(512).max(1);
    let sample: Vec<_> = data.iter().step_by(stride).map(|(_, emb)| *emb).collect();
    let mut mean = vec![0.0; dimensions];
    for emb in &sample {
        for (avg, value) in mean.iter_mut().zip(&emb.values) {
            *avg += value.to_f32() as f64;
        }
    }
    for avg in &mut mean {
        *avg /= sample.len() as f64;
    }

    let x = axis(&sample, &mean, &[]);
    let y = axis(&sample, &mean, &[&x]);
    let z = axis(&sample, &mean, &[&x, &y]);
    let projected: Vec<_> = data
        .iter()
        .map(|(_, emb)| {
            let projection = |axis: &[f64]| {
                emb.values
                    .iter()
                    .zip(&mean)
                    .zip(axis)
                    .map(|((value, avg), direction)| (value.to_f32() as f64 - avg) * direction)
                    .sum::<f64>()
            };
            [projection(&x), projection(&y), projection(&z)]
        })
        .collect();
    let points = projected
        .iter()
        .map(|p| PlotPoint::new(p[0], p[1]))
        .collect();
    let points_3d = projected
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    (points, points_3d)
}

fn axis(sample: &[&Embedding], mean: &[f64], previous: &[&[f64]]) -> Vec<f64> {
    let mut axis = sample
        .iter()
        .map(|emb| {
            emb.values
                .iter()
                .zip(mean)
                .map(|(value, avg)| value.to_f32() as f64 - avg)
                .collect::<Vec<_>>()
        })
        .find(|candidate| {
            let residual: f64 = previous
                .iter()
                .map(|prev| dot(candidate, prev).powi(2))
                .sum();
            dot(candidate, candidate) - residual > 1e-12
        })
        .unwrap_or_else(|| vec![0.0; mean.len()]);
    for prev in previous {
        let component = dot(&axis, prev);
        for (value, prior) in axis.iter_mut().zip(prev.iter()) {
            *value -= component * prior;
        }
    }
    if !normalize(&mut axis) {
        return axis;
    }
    for _ in 0..24 {
        let mut next = vec![0.0; mean.len()];
        for emb in sample {
            let score: f64 = emb
                .values
                .iter()
                .zip(mean)
                .zip(&axis)
                .map(|((value, avg), direction)| (value.to_f32() as f64 - avg) * direction)
                .sum();
            for ((sum, value), avg) in next.iter_mut().zip(&emb.values).zip(mean) {
                *sum += (value.to_f32() as f64 - avg) * score;
            }
        }
        for prev in previous {
            let component = dot(&next, prev);
            for (value, prior) in next.iter_mut().zip(prev.iter()) {
                *value -= component * prior;
            }
        }
        if !normalize(&mut next) {
            break;
        }
        axis = next;
    }
    axis
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(values: &mut [f64]) -> bool {
    let norm = dot(values, values).sqrt();
    if norm <= 1e-12 {
        return false;
    }
    for value in values {
        *value /= norm;
    }
    true
}

impl Gui {
    pub(super) fn show_embedding_plot(&mut self, ui: &egui::Ui) {
        let cloud_renderer = self.cloud_renderer.clone();
        let Some(plot) = self.embedding_plot.as_mut() else {
            return;
        };
        if !plot.open {
            return;
        }
        let files = &self.selected_imgs;
        let current = self.image_texture_n;
        let lang = &self.lang;
        let mut open = true;
        let mut selected = None;
        let mut next_kind = None;
        let has_images = files.iter().any(|file| matches!(&file.aioutput, Some(AIOutputs::Embed(embedding)) if valid_embedding(embedding)));
        let has_objects = files.iter().any(|file| match file.aioutput.as_ref() {
            Some(AIOutputs::ObjectDetection(boxes)) => boxes
                .iter()
                .any(|bbox| bbox.embedding.as_ref().is_some_and(valid_embedding)),
            Some(AIOutputs::Segmentation(segments)) => segments
                .iter()
                .any(|seg| seg.bbox.embedding.as_ref().is_some_and(valid_embedding)),
            _ => false,
        });
        egui::Window::new(translate(Key::embedding_space, lang))
            .open(&mut open)
            .default_size([860.0, 500.0])
            .min_width(400.0)
            .min_height(360.0)
            .show(ui.ctx(), |ui| {
                if has_images && has_objects {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(!plot.objects, translate(Key::image, lang))
                            .clicked()
                            && plot.objects
                        {
                            next_kind = Some(false);
                        }
                        if ui
                            .selectable_label(plot.objects, translate(Key::detections, lang))
                            .clicked()
                            && !plot.objects
                        {
                            next_kind = Some(true);
                        }
                    });
                }
                if plot.points.is_empty() {
                    ui.label(translate(
                        if plot.objects {
                            Key::no_object_embeddings
                        } else {
                            Key::no_embeddings
                        },
                        lang,
                    ));
                    return;
                }
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut plot.view_3d, false, "2D");
                    ui.selectable_value(&mut plot.view_3d, true, "3D");
                });
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut plot.color_clusters, "K-means").changed()
                        && plot.color_clusters
                        && plot.clusters.is_empty()
                    {
                        plot.recluster();
                    }
                    if plot.color_clusters {
                        let slider = ui.add(
                            egui::Slider::new(&mut plot.cluster_count, 2..=20)
                                .text(translate(Key::groups, lang)),
                        );
                        if !slider.is_pointer_button_down_on()
                            && plot.cluster_count != plot.computed_cluster_count
                        {
                            plot.recluster();
                        }
                    }
                });
                if plot.objects {
                    ui.label(format!("{} · {}", plot.points.len(), plot.model));
                } else {
                    ui.label(format!(
                        "{} / {} · {}",
                        plot.points.len(),
                        files.len(),
                        plot.model
                    ));
                }
                let id = egui::Id::new((
                    "image_embeddings",
                    &files[0].file_path,
                    files.len(),
                    plot.points.len(),
                    &plot.model,
                    plot.objects,
                ));
                let current_point = if plot.highlight_selection {
                    plot.selected_point
                        .filter(|&point| plot.indexes[point] + 1 == current)
                        .or_else(|| plot.indexes.iter().position(|&index| index + 1 == current))
                } else {
                    None
                };
                let inspector_width = (ui.available_width() * 0.30).clamp(160.0, 240.0);
                let plot_width = (ui.available_width() - inspector_width - 12.0).max(200.0);
                let plot_height = ui.available_height().max(300.0);
                let accent = ui.visuals().selection.stroke.color;
                ui.horizontal(|ui| {
                    let interaction = if plot.view_3d {
                        plot.cloud_3d.show(
                            ui,
                            egui::vec2(plot_width, plot_height),
                            &cloud_renderer,
                            current_point,
                            plot.color_clusters,
                        )
                    } else {
                        let result = Plot::new(id)
                            .width(plot_width)
                            .height(plot_height)
                            .data_aspect(1.0)
                            .show_axes([false, false])
                            .show_grid([false, false])
                            .show(ui, |plot_ui| {
                                if plot.color_clusters {
                                    for (index, points) in plot.clusters.iter().enumerate() {
                                        if !points.is_empty() {
                                            plot_ui.points(
                                                Points::new(
                                                    format!("Cluster {}", index + 1),
                                                    points.as_slice(),
                                                )
                                                .radius(5.0)
                                                .color(cluster_color(index, plot.clusters.len()))
                                                .allow_hover(false),
                                            );
                                        }
                                    }
                                } else {
                                    plot_ui.points(
                                        Points::new("Images", plot.points.as_slice())
                                            .radius(5.0)
                                            .allow_hover(false),
                                    );
                                }
                                if let Some(point_index) = current_point {
                                    let point = plot.points[point_index];
                                    plot_ui.points(
                                        Points::new("Selected", [point.x, point.y])
                                            .radius(8.0)
                                            .color(accent)
                                            .allow_hover(false),
                                    );
                                }
                                if let Some(point_index) = plot.hovered {
                                    let point = plot.points[point_index];
                                    plot_ui.points(
                                        Points::new("Hovered", [point.x, point.y])
                                            .radius(10.0)
                                            .filled(false)
                                            .color(accent)
                                            .allow_hover(false),
                                    );
                                }
                            });
                        let hovered = result.response.hover_pos().and_then(|pointer| {
                            let mut nearest = None;
                            let mut distance = 12.0_f32.powi(2);
                            for (index, point) in plot.points.iter().enumerate() {
                                let candidate = result
                                    .transform
                                    .position_from_point(point)
                                    .distance_sq(pointer);
                                if candidate < distance {
                                    nearest = Some(index);
                                    distance = candidate;
                                }
                            }
                            nearest
                        });
                        if hovered.is_some() {
                            result
                                .response
                                .clone()
                                .on_hover_cursor(egui::CursorIcon::PointingHand);
                        }
                        super::embedding_plot_3d::Interaction {
                            hovered,
                            clicked: result.response.clicked().then_some(hovered),
                        }
                    };
                    if plot.hovered != interaction.hovered {
                        plot.hovered = interaction.hovered;
                        ui.ctx().request_repaint();
                    }
                    if let Some(clicked) = interaction.clicked {
                        if let Some(point_index) = clicked {
                            selected = Some(point_index);
                        } else {
                            plot.selected_point = None;
                            plot.highlight_selection = false;
                            ui.ctx().request_repaint();
                        }
                    }

                    let shown = interaction
                        .hovered
                        .or(plot.preview_point)
                        .or(current_point)
                        .unwrap_or(0);
                    let file_index = plot.indexes[shown];
                    let file = &files[file_index];
                    plot.preview(ui.ctx(), shown, &file.file_path);
                    ui.vertical(|ui| {
                        ui.set_width(inspector_width);
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(inspector_width, (inspector_width * 0.8).min(190.0)),
                            egui::Sense::hover(),
                        );
                        ui.painter()
                            .rect_filled(rect, 6.0, ui.visuals().faint_bg_color);
                        if let Some(texture) = plot
                            .thumbnails
                            .iter()
                            .find(|(index, _)| *index == shown)
                            .and_then(|(_, texture)| texture.as_ref())
                        {
                            let size = texture.size_vec2();
                            let scale = (rect.width() / size.x).min(rect.height() / size.y);
                            let image_rect =
                                egui::Rect::from_center_size(rect.center(), size * scale);
                            egui::Image::new(texture)
                                .corner_radius(6.0)
                                .paint_at(ui, image_rect);
                        } else if plot.pending.contains(&shown) {
                            ui.put(
                                egui::Rect::from_center_size(rect.center(), egui::vec2(20.0, 20.0)),
                                egui::Spinner::new(),
                            );
                        }
                        ui.strong(
                            file.file_path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("?"),
                        )
                        .on_hover_text(file.file_path.display().to_string());
                        if plot.objects {
                            ui.label(&plot.labels[shown]);
                        }
                        if plot.color_clusters {
                            let cluster = plot.cluster_ids[shown];
                            ui.colored_label(
                                cluster_color(cluster, plot.clusters.len()),
                                format!("{} {}", translate(Key::cluster, lang), cluster + 1),
                            );
                        }
                        ui.label(
                            egui::RichText::new(translate(Key::plot_help, lang))
                                .weak()
                                .small(),
                        );
                        if plot.view_3d {
                            ui.label(
                                egui::RichText::new(translate(Key::plot_3d_help, lang))
                                    .weak()
                                    .small(),
                            );
                        }
                    });
                });
            });
        plot.open = open;
        if let Some(objects) = next_kind {
            let mut replacement = EmbeddingPlot::for_kind(files, objects);
            replacement.color_clusters = plot.color_clusters;
            replacement.cluster_count = plot.cluster_count;
            replacement.highlight_selection = plot.highlight_selection;
            replacement.view_3d = plot.view_3d;
            if replacement.color_clusters && !replacement.points.is_empty() {
                replacement.recluster();
            }
            self.embedding_plot = Some(replacement);
            return;
        }
        if let Some(point_index) = selected {
            plot.selected_point = Some(point_index);
            plot.highlight_selection = true;
            ui.ctx().request_repaint();
            let index = plot.indexes[point_index];
            if index + 1 != current {
                self.image_texture_n = index + 1;
                self.image_view.reset();
                self.paint(ui, index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::abstractions::XYXYc;

    #[test]
    fn k_means_colors_separated_groups_together() {
        let points = [
            [0.0, 0.0, -5.0],
            [0.1, 0.0, -5.1],
            [0.0, 0.1, -4.9],
            [0.0, 0.0, 5.0],
            [0.1, 0.0, 5.1],
            [0.0, 0.1, 4.9],
        ];
        let groups = k_means_3d(&points, 2);
        assert_eq!(groups[0], groups[1]);
        assert_eq!(groups[1], groups[2]);
        assert_eq!(groups[3], groups[4]);
        assert_eq!(groups[4], groups[5]);
        assert_ne!(groups[0], groups[3]);
    }

    #[test]
    fn sample_images_group_by_similarity() {
        let files: Vec<_> = (1..=4)
            .map(|n| PredImg::new_simple(format!("tests/assets/embed/{n}.jpg").into()))
            .collect();
        let plot = EmbeddingPlot::new(&files);
        assert_eq!(plot.points.len(), 4);
        assert_eq!(plot.model, "dinov3-vitb16");
        for point in &plot.points {
            assert!(point.x.is_finite() && point.y.is_finite());
        }
        let distance = |a: usize, b: usize| {
            let (a, b) = (plot.points[a], plot.points[b]);
            (a.x - b.x).hypot(a.y - b.y)
        };
        // Images 1/4 show deer; 2/3 show elephants.
        assert!(distance(0, 3) < distance(0, 1));
        assert!(distance(1, 2) < distance(1, 0));
    }

    #[test]
    fn detection_embeddings_plot_each_crop_and_read_legacy_boxes() {
        let mut file = PredImg::new_simple("missing-plot-test-image.jpg".into());
        let mut first = XYXYc::new(XYXY::new(1.0, 2.0, 11.0, 12.0, 0.9, 0), "deer".into());
        first.embedding = Some(Embedding::from_raw(&[1.0, 0.0, 0.0], "encoder".into()));
        let mut second = XYXYc::new(XYXY::new(20.0, 21.0, 30.0, 31.0, 0.8, 1), "elk".into());
        second.embedding = Some(Embedding::from_raw(&[0.0, 1.0, 0.0], "encoder".into()));
        file.aioutput = Some(AIOutputs::ObjectDetection(vec![first, second]));
        let plot = EmbeddingPlot::new(&[file]);
        assert!(plot.objects);
        assert_eq!(plot.points.len(), 2);
        assert_eq!(plot.indexes, [0, 0]);
        assert_eq!(plot.labels, ["deer", "elk"]);
        assert_eq!(plot.boxes[1].unwrap().x1, 20.0);

        let legacy = r#"{"xyxy":{"x1":0.0,"y1":0.0,"x2":1.0,"y2":1.0,"prob":0.5,"class_id":0},"label":"old","extra_cls":null}"#;
        let box_from_old_sidecar: XYXYc = serde_json::from_str(legacy).unwrap();
        assert!(box_from_old_sidecar.embedding.is_none());
    }

    #[test]
    fn three_dimensional_projection_keeps_independent_direction() {
        let embeddings: Vec<_> = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
        .into_iter()
        .map(|values| Embedding::from_raw(&values, "test".into()))
        .collect();
        let data: Vec<_> = embeddings.iter().enumerate().collect();
        let (points_2d, points_3d) = project(&data);
        assert_eq!(points_2d.len(), points_3d.len());
        assert!(points_3d.iter().any(|point| point[2].abs() > 0.9));
        for (two, three) in points_2d.iter().zip(points_3d) {
            assert!((two.x - three[0] as f64).abs() < 1e-5);
            assert!((two.y - three[1] as f64).abs() < 1e-5);
        }
    }
}
