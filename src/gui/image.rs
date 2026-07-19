use super::{imgbuf_to_texture, Gui, Mode, OpenDialog};
use crate::api::abstractions::*;
use crate::api::bq::process_imgbuf;
use crate::api::export;
use crate::api::render::*;
use crate::api::rest::Payload;
use crate::localization::*;
use std::fs;

const MIN_PREVIEW_H: f32 = 240.0;

impl Gui {
    // ---------- texture loading ----------

    pub(super) fn paint(&mut self, ui: &egui::Ui, i: usize) {
        let Some(predimg) = self.selected_imgs.get(i) else { return; };
        let Ok(loaded) = image::open(&predimg.file_path) else { return; };
        let img = loaded.into_rgba8();
        self.img_state.texture = imgbuf_to_texture(&img, ui);
        self.mask_textures = build_mask_textures(predimg, ui);
    }

    // ---------- analysis lifecycle ----------
    pub(super) fn start_single_img_analysis(&mut self, target: usize) {
        if target >= self.selected_imgs.len() || self.img_state.is_processing {
            return;
        }
        self.selected_imgs[target].reset();

        let (tx, mut cancel_rx) = self.img_state.start();
        let predimg = self.selected_imgs[target].clone();

        let rest_client = self.rest_client.clone();
        let is_remote = !self.ep_selected.is_local();
        tokio::spawn(async move {
            if cancel_rx.try_recv().is_ok() {
                return;
            }
            let result = if is_remote {
                let buffer = fs::read(&predimg.file_path).unwrap();
                match rest_client.as_ref().unwrap().detect(Payload::RawImageBytes(buffer)).await {
                    Ok(result) => Some(result),
                    Err(_) => None,
                }
            } else {
                let img = image::open(&predimg.file_path).unwrap().into_rgb8();
                match tokio::task::spawn_blocking(move || process_imgbuf(&img)).await {
                    Ok(Ok(result)) => Some(result),
                    _ => None,
                }
            };

            let _ = tx.send((target, result));
        });
    }

    pub(super) fn start_img_analysis(&mut self) {
        if self.process_all_imgs {
            self.selected_imgs
                .iter_mut()
                .for_each(|pred_img| pred_img.reset());
        }
        let (tx, mut cancel_rx) = self.img_state.start();
        let copy_predigms = self.selected_imgs.clone();

        let rest_client = self.rest_client.clone();
        let is_remote = !self.ep_selected.is_local();
        tokio::spawn(async move {
            for (i, predimg) in copy_predigms.iter().enumerate() {
                if predimg.wasprocessed {
                    continue;
                }
                if cancel_rx.try_recv().is_ok() {
                    break;
                }
                let result = if is_remote {
                    let buffer = fs::read(&predimg.file_path).unwrap();
                    match rest_client.as_ref().unwrap().detect(Payload::RawImageBytes(buffer)).await {
                        Ok(result) => Some(result),
                        Err(_) => None,
                    }
                } else {
                    let img = image::open(&predimg.file_path).unwrap().into_rgb8();
                    match tokio::task::spawn_blocking(move || process_imgbuf(&img)).await {
                        Ok(Ok(result)) => Some(result),
                        _ => None,
                    }
                };

                if tx.send((i, result)).is_err() {
                    break;
                }
            }
        });
    }

    pub(super) fn img_handle_results(&mut self, ui: &egui::Ui) {
        let (updates, closed) = self.img_state.drain();
        for (i, result) in updates {
            match result {
                Some(aio) => {
                    self.selected_imgs[i].aioutput = Some(aio);
                    self.selected_imgs[i].wasprocessed = true;
                    if i == self.image_texture_n - 1 {
                        self.paint(ui, i);
                    }
                }
                None => self.push_toast(super::Message::Error),
            }
        }
        if closed {
            self.img_state.finish();
        }
        self.img_state.progress_bar = self.selected_imgs.get_progress();
        ui.request_repaint();
    }

    // ---------- left-panel widget (Analyze / Export) ----------

    pub(super) fn img_analysis_widget(&mut self, ui: &mut egui::Ui) {
        if self.selected_imgs.is_empty() || !self.can_run_image_ai() {
            return;
        }

        ui.vertical_centered(|ui| {
            ui.heading(self.t(Key::image));
            ui.heading(self.t(Key::analysis));
        });
        ui.separator();

        ui.vertical_centered(|ui| {
            if ui
                .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::analyze)))
                .clicked()
                && !self.img_state.is_processing
            {
                if self.selected_imgs.get_progress() == 0.0 {
                    self.start_img_analysis();
                } else {
                    self.dialog = OpenDialog::ProcessAll;
                }
            }

            if self.img_state.is_processing
                && ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::cancel)))
                    .clicked()
            {
                self.img_state.cancel();
            }
        });

        ui.add(
            egui::ProgressBar::new(self.img_state.progress_bar)
                .show_percentage()
                .animate(self.img_state.is_processing),
        );

        ui.add_space(8.0);

        if self.mode == Mode::Image {
            ui.vertical_centered(|ui| {
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::export)))
                    .clicked()
                {
                    self.dialog = OpenDialog::Export;
                }
            });
        }

        self.process_all_dialog(ui, |gui, process_all| gui.process_all_imgs = process_all, Gui::start_img_analysis);
    }

    pub fn img_export_dialog(&mut self, ui: &mut egui::Ui) {
        egui::Window::new(self.t(Key::export))
            .collapsible(false)
            .resizable(false)
            .show(ui, |ui| {
                if ui.button(self.t(Key::export_predictions)).clicked() {
                    for file in self.selected_imgs.clone() {
                        tokio::spawn(async move {
                            let _ = file.write_predictions();
                        });
                    }
                    let msg = self.t(Key::saved_next_to_originals).to_string();
                    self.push_toast(super::Message::ok(msg));
                    self.dialog = OpenDialog::None;
                }

                if ui
                    .button(self.t(Key::export_imgs_with_predictions))
                    .clicked()
                {
                    for file in self.selected_imgs.clone() {
                        let has_predictions = file
                            .aioutput
                            .as_ref()
                            .map(|a| !a.is_empty())
                            .unwrap_or(false);
                        if file.wasprocessed && has_predictions {
                            tokio::spawn(async move {
                                let _ = file.save();
                            });
                        }
                    }
                    self.process_done_at(format!("{}/", export::EXPORT_DIR));
                    self.dialog = OpenDialog::None;
                }

                if ui.button(self.t(Key::cancel)).clicked() {
                    self.dialog = OpenDialog::None;
                }
            });
    }

    // ---------- main image viewer ----------

    pub(super) fn ui_image(&mut self, ui: &mut egui::Ui) {
        if self.selected_imgs.is_empty() {
            return;
        }
        let n = self.selected_imgs.len();
        if self.image_texture_n < 1 {
            self.image_texture_n = 1;
        }
        if self.image_texture_n > n {
            self.image_texture_n = n;
        }

        self.draw_image_header(ui, n);

        let i = self.image_texture_n - 1;
        let has_spatial_output = matches!(
            self.selected_imgs[i].aioutput.as_ref(),
            Some(AIOutputs::ObjectDetection(b)) if !b.is_empty()
        ) || matches!(
            self.selected_imgs[i].aioutput.as_ref(),
            Some(AIOutputs::Segmentation(s)) if !s.is_empty()
        );

        let avail = ui.available_size_before_wrap();
        let preview_w = avail.x.max(200.0);
        let echo_strip_h: f32 = if has_spatial_output { 30.0 } else { 0.0 };
        let echo_strip_gap: f32 = if has_spatial_output { 6.0 } else { 0.0 };
        let preview_h = (avail.y - 8.0 - echo_strip_h - echo_strip_gap).max(MIN_PREVIEW_H);

        ui.vertical(|ui| {
            ui.set_max_width(preview_w);
            let echo = self.draw_image_preview(ui, preview_w, preview_h);
            if has_spatial_output {
                ui.add_space(echo_strip_gap);
                draw_echo_strip(ui, echo.as_ref(), echo_strip_h, preview_w);
            }
        });

        self.img_handle_results(ui);
    }

    fn draw_image_header(&mut self, ui: &mut egui::Ui, n: usize) {
        let mut new_index = self.image_texture_n;
        let can_analyze = self.can_run_image_ai();
        let mut analyze_this = false;

        ui.horizontal_wrapped(|ui| {
            super::nav_prev_next(
                ui,
                &mut new_index,
                n,
                self.t(Key::prev),
                self.t(Key::next),
            );
            let predimg = &self.selected_imgs[new_index - 1];
            let name = predimg
                .file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(self.t(Key::unknown_file));
            super::nav_filename(ui, name, new_index, n);

            if !predimg.wasprocessed {
                ui.separator();
                ui.label(
                    egui::RichText::new(self.t(Key::not_analysed))
                        .weak()
                        .small(),
                );
            }

            if can_analyze {
                ui.separator();
                let resp = ui.add_enabled(
                    !self.img_state.is_processing,
                    egui::Button::new(self.t(Key::analyze)),
                );
                if resp.clicked() {
                    analyze_this = true;
                }
            }
        });

        super::nav_slider(ui, &mut new_index, n);

        if new_index != self.image_texture_n {
            self.image_texture_n = new_index;
            self.paint(ui, new_index - 1);
        }

        if analyze_this {
            self.start_single_img_analysis(new_index - 1);
        }
    }

    fn draw_image_preview(
        &mut self,
        ui: &mut egui::Ui,
        max_w: f32,
        max_h: f32,
    ) -> Option<HoverEcho> {
        let Some(tex) = self.img_state.texture.clone() else {
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(max_w, max_h.min(MIN_PREVIEW_H)),
                egui::Sense::hover(),
            );
            let bg = if ui.visuals().dark_mode {
                egui::Color32::from_rgb(28, 32, 36)
            } else {
                egui::Color32::from_rgb(228, 230, 234)
            };
            ui.painter().rect_filled(rect, 8.0, bg);
            return None;
        };

        let tex_size = tex.size_vec2();
        if tex_size.x < 1.0 || tex_size.y < 1.0 {
            return None;
        }
        let scale = (max_w / tex_size.x).min(max_h / tex_size.y).min(1.0);
        let disp_w = (tex_size.x * scale).max(1.0);
        let disp_h = (tex_size.y * scale).max(1.0);

        ui.vertical_centered(|ui| -> Option<HoverEcho> {
            let img_resp = ui.add(
                egui::Image::new(&tex)
                    .max_size(egui::vec2(disp_w, disp_h))
                    .corner_radius(8.0),
            );

            let i = self.image_texture_n - 1;
            let predimg = &self.selected_imgs[i];

            if predimg.wasprocessed {
                match predimg.aioutput.as_ref() {
                    Some(aio) if !aio.is_empty() => draw_image_overlay(
                        ui,
                        &img_resp,
                        aio,
                        tex_size,
                        &self.mask_textures,
                        &self.lang,
                    ),
                    Some(_) => {
                        draw_empty_predictions_chip(ui, &img_resp, &self.lang);
                        None
                    }
                    None => None,
                }
            } else {
                None
            }
        })
        .inner
    }

}

// ---------- free helpers ----------

#[derive(Clone)]
struct HoverEcho {
    class_id: u32,
    label: String,
    prob: f32,
    refined: Option<(String, f32)>,
    width_px: i32,
    height_px: i32,
    is_segment: bool,
}

impl HoverEcho {
    fn from_bbox(b: &XYXYc, is_segment: bool) -> Self {
        let refined = b
            .extra_cls
            .as_ref()
            .and_then(|c| c.top())
            .map(|p| (p.label.clone(), p.prob));
        Self {
            class_id: bbox_color_id(b),
            label: b.label.clone(),
            prob: b.xyxy.prob,
            refined,
            width_px: (b.xyxy.x2 - b.xyxy.x1).max(0.0).round() as i32,
            height_px: (b.xyxy.y2 - b.xyxy.y1).max(0.0).round() as i32,
            is_segment,
        }
    }
}

fn draw_echo_strip(
    ui: &mut egui::Ui,
    echo: Option<&HoverEcho>,
    strip_h: f32,
    width: f32,
) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(width, strip_h),
        egui::Sense::hover(),
    );
    let Some(e) = echo else {
        // Reserve the space silently — info pops in when the user hovers,
        // otherwise the strip is invisible.
        return;
    };

    let painter = ui.painter_at(rect);
    let dark = ui.visuals().dark_mode;
    let bg = if dark {
        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 10)
    } else {
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 8)
    };
    painter.rect_filled(rect, 6.0, bg);

    let pad = 10.0;
    let cy = rect.center().y;
    let font_main = egui::FontId::proportional(13.0);
    let font_small = egui::FontId::proportional(11.5);

    let main_color = if dark {
        egui::Color32::from_gray(245)
    } else {
        egui::Color32::from_gray(20)
    };
    let weak_color = if dark {
        egui::Color32::from_gray(170)
    } else {
        egui::Color32::from_gray(110)
    };

    let color = super::class_color32(e.class_id);

    let chip = egui::Rect::from_center_size(
        egui::pos2(rect.left() + pad + 6.0, cy),
        egui::vec2(12.0, 12.0),
    );
    painter.rect_filled(chip, 2.5, color);

    let mut x = chip.right() + 10.0;

    let main_text = format!("{}  {:.0}%", e.label, e.prob * 100.0);
    let g_main = painter.layout_no_wrap(
        main_text,
        font_main.clone(),
        egui::Color32::PLACEHOLDER,
    );
    painter.galley(
        egui::pos2(x, cy - g_main.size().y / 2.0),
        g_main.clone(),
        main_color,
    );
    x += g_main.size().x;

    if let Some((r_label, r_prob)) = &e.refined {
        let sep = painter.layout_no_wrap(
            "  →  ".into(),
            font_main.clone(),
            egui::Color32::PLACEHOLDER,
        );
        painter.galley(
            egui::pos2(x, cy - sep.size().y / 2.0),
            sep.clone(),
            weak_color,
        );
        x += sep.size().x;

        let r_text = format!("{}  {:.0}%", r_label, r_prob * 100.0);
        let g_ref = painter.layout_no_wrap(
            r_text,
            font_main.clone(),
            egui::Color32::PLACEHOLDER,
        );
        painter.galley(
            egui::pos2(x, cy - g_ref.size().y / 2.0),
            g_ref.clone(),
            main_color,
        );
        x += g_ref.size().x;
    }

    let tail_text = if e.is_segment {
        format!("  ·  {} × {} px  ·  segment", e.width_px, e.height_px)
    } else {
        format!("  ·  {} × {} px", e.width_px, e.height_px)
    };
    let g_tail = painter.layout_no_wrap(
        tail_text,
        font_small,
        egui::Color32::PLACEHOLDER,
    );
    if x + g_tail.size().x < rect.right() - pad {
        painter.galley(
            egui::pos2(x, cy - g_tail.size().y / 2.0),
            g_tail,
            weak_color,
        );
    }
}

fn bbox_screen_rect(
    b: &XYXYc,
    origin: egui::Pos2,
    scale: egui::Vec2,
) -> egui::Rect {
    egui::Rect::from_min_max(
        egui::pos2(
            origin.x + b.xyxy.x1 * scale.x,
            origin.y + b.xyxy.y1 * scale.y,
        ),
        egui::pos2(
            origin.x + b.xyxy.x2 * scale.x,
            origin.y + b.xyxy.y2 * scale.y,
        ),
    )
}

fn draw_image_overlay(
    ui: &egui::Ui,
    img_resp: &egui::Response,
    aio: &AIOutputs,
    original_size: egui::Vec2,
    mask_textures: &[egui::TextureHandle],
    lang: &Lang,
) -> Option<HoverEcho> {
    let rect = img_resp.rect;
    if rect.width() < 1.0 || rect.height() < 1.0 {
        return None;
    }
    let scale = egui::vec2(
        rect.width() / original_size.x.max(1.0),
        rect.height() / original_size.y.max(1.0),
    );
    let hover_pos = img_resp.hover_pos();
    let painter = ui.painter_at(rect);

    match aio {
        AIOutputs::ObjectDetection(bboxes) => {
            let hovered = pick_hovered(bboxes, hover_pos, rect.min, scale, |b| b);
            for (idx, b) in bboxes.iter().enumerate() {
                draw_box_with_label(
                    &painter,
                    rect,
                    bbox_screen_rect(b, rect.min, scale),
                    b,
                    Some(idx) == hovered,
                );
            }
            if let Some(idx) = hovered {
                img_resp.clone().on_hover_ui_at_pointer(|ui| {
                    bbox_tooltip_ui(ui, &bboxes[idx], lang);
                });
            }
            hovered.map(|idx| HoverEcho::from_bbox(&bboxes[idx], false))
        }
        AIOutputs::Segmentation(segs) => {
            let hovered = pick_hovered(segs, hover_pos, rect.min, scale, |s| &s.bbox);
            let uv = egui::Rect::from_min_max(
                egui::pos2(0.0, 0.0),
                egui::pos2(1.0, 1.0),
            );
            for (idx, s) in segs.iter().enumerate() {
                let bbox_rect = bbox_screen_rect(&s.bbox, rect.min, scale);

                // Paint the alpha mask (40 % class-colour over masked pixels)
                // so the segmented shape is visible, not just the bbox.
                // A second pass on hover brightens the masked area.
                if let Some(tex) = mask_textures.get(idx) {
                    painter.image(tex.id(), bbox_rect, uv, egui::Color32::WHITE);
                    if Some(idx) == hovered {
                        painter.image(tex.id(), bbox_rect, uv, egui::Color32::WHITE);
                    }
                }

                draw_box_with_label(
                    &painter,
                    rect,
                    bbox_rect,
                    &s.bbox,
                    Some(idx) == hovered,
                );
            }
            if let Some(idx) = hovered {
                img_resp.clone().on_hover_ui_at_pointer(|ui| {
                    seg_tooltip_ui(ui, &segs[idx], lang);
                });
            }
            hovered.map(|idx| HoverEcho::from_bbox(&segs[idx].bbox, true))
        }
        AIOutputs::Classification(probs) => {
            draw_classification_ribbon(&painter, rect, probs);
            img_resp.clone().on_hover_ui_at_pointer(|ui| {
                classification_tooltip_ui(ui, probs, lang);
            });
            None
        }
        AIOutputs::AudioClassification(_) => None,
        AIOutputs::Embed(emb) => {
            let chip = format!("{} · {}", translate(Key::embedding, lang), emb.model);
            draw_corner_chip(&painter, rect, &chip, 12.0, 22.0, 10.0);
            None
        }
    }
}

/// Small rounded label chip anchored to `rect`'s top-left corner.
fn draw_corner_chip(
    painter: &egui::Painter,
    rect: egui::Rect,
    text: &str,
    font_size: f32,
    chip_h: f32,
    corner_radius: f32,
) {
    let galley = painter.layout_no_wrap(
        text.into(),
        egui::FontId::proportional(font_size),
        egui::Color32::WHITE,
    );
    let pad = 8.0;
    let chip_w = galley.size().x + 14.0;
    let bg = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180);
    let chip = egui::Rect::from_min_size(
        egui::pos2(rect.min.x + pad, rect.min.y + pad),
        egui::vec2(chip_w, chip_h),
    );
    painter.rect_filled(chip, corner_radius, bg);
    painter.galley(
        chip.min + egui::vec2(7.0, (chip_h - galley.size().y) / 2.0),
        galley,
        egui::Color32::WHITE,
    );
}

/// Smallest-area item under the cursor — smallest wins so a small box nested
/// inside a bigger one is still selectable.
fn pick_hovered<T>(
    items: &[T],
    hover_pos: Option<egui::Pos2>,
    origin: egui::Pos2,
    scale: egui::Vec2,
    bbox_of: impl Fn(&T) -> &XYXYc,
) -> Option<usize> {
    let hover_point = hover_pos?;
    let mut best: Option<(usize, f32)> = None;
    for (idx, item) in items.iter().enumerate() {
        let item_rect = bbox_screen_rect(bbox_of(item), origin, scale);
        if item_rect.contains(hover_point) {
            let area = item_rect.width() * item_rect.height();
            if best.map_or(true, |(_, best_area)| area < best_area) {
                best = Some((idx, area));
            }
        }
    }
    best.map(|(idx, _)| idx)
}

fn bbox_color_id(b: &XYXYc) -> u32 {
    b.extra_cls
        .as_ref()
        .and_then(|c| c.top())
        .map(|p| p.class_id)
        .unwrap_or(b.xyxy.class_id)
}

fn draw_box_with_label(
    painter: &egui::Painter,
    clip: egui::Rect,
    r: egui::Rect,
    b: &XYXYc,
    hovered: bool,
) {
    let color = super::class_color32(bbox_color_id(b));

    let stroke_w = if hovered { 3.0 } else { 1.8 };
    painter.rect_stroke(
        r,
        2.0,
        egui::Stroke::new(stroke_w, color),
        egui::StrokeKind::Inside,
    );

    // Mirror render::str_label — two lines when extra_cls refined this
    // detection so the refined class+confidence are always visible on the
    // image (not just in a hover tooltip).
    let text = label_text(b);
    let font = egui::FontId::proportional(12.0);
    let galley = painter.layout_no_wrap(text, font, egui::Color32::WHITE);
    let label_w = galley.size().x + 8.0;
    let label_h = galley.size().y + 4.0;

    if r.width() >= label_w * 0.6 {
        let y = if r.min.y - label_h >= clip.min.y {
            r.min.y - label_h
        } else {
            r.min.y
        };
        let bg = egui::Rect::from_min_size(
            egui::pos2(r.min.x, y),
            egui::vec2(label_w, label_h),
        );
        painter.rect_filled(bg, 2.0, color);
        painter.galley(
            bg.min + egui::vec2(4.0, 2.0),
            galley,
            egui::Color32::WHITE,
        );
    }
}

fn label_text(b: &XYXYc) -> String {
    let base = format!("{}  {:.0}%", b.label, b.xyxy.prob * 100.0);
    match b.extra_cls.as_ref().and_then(|c| c.top()) {
        Some(p) => format!("{}\n{}  {:.0}%", base, p.label, p.prob * 100.0),
        None => base,
    }
}

/// Upload one RGBA texture per segmentation mask. Each pixel is the segment's
/// class colour at 40% alpha where the mask bit is set, fully transparent
/// otherwise — matches `render::draw_seg_from_imgbuf`'s blend factor so the
/// on-screen look mirrors the burned-in export.
pub(super) fn build_mask_textures(
    predimg: &PredImg,
    ui: &egui::Ui,
) -> Vec<egui::TextureHandle> {
    let Some(AIOutputs::Segmentation(segs)) = predimg.aioutput.as_ref() else {
        return Vec::new();
    };
    const MASK_ALPHA: u8 = 102; // ≈ 0.4 × 255
    segs.iter()
        .enumerate()
        .map(|(idx, s)| {
            let w = s.mask.width;
            let h = s.mask.height;
            let c = class_color(bbox_color_id(&s.bbox));
            let mut pixels = Vec::with_capacity(w * h * 4);
            for bit in s.mask.data.iter() {
                if *bit {
                    pixels.extend_from_slice(&[c[0], c[1], c[2], MASK_ALPHA]);
                } else {
                    pixels.extend_from_slice(&[0, 0, 0, 0]);
                }
            }
            let color_img =
                egui::ColorImage::from_rgba_unmultiplied([w, h], &pixels);
            ui.ctx().load_texture(
                format!("seg_mask_{idx}"),
                color_img,
                egui::TextureOptions::NEAREST,
            )
        })
        .collect()
}

fn draw_classification_ribbon(
    painter: &egui::Painter,
    rect: egui::Rect,
    probs: &[Prob],
) {
    if probs.is_empty() {
        return;
    }
    let mut ranked: Vec<&Prob> = probs.iter().collect();
    ranked.sort_by(|a, b| {
        b.prob
            .partial_cmp(&a.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let pad = 8.0;
    let chip_h = 24.0;
    let font = egui::FontId::proportional(13.0);
    let mut x = rect.min.x + pad;
    let y = rect.min.y + pad;
    let max_x = rect.max.x - pad;

    for p in ranked.iter().take(3) {
        let c = class_color(p.class_id);
        let bg = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], 230);
        let text = format!("{}  {:.0}%", p.label, p.prob * 100.0);
        let galley = painter.layout_no_wrap(text, font.clone(), egui::Color32::WHITE);
        let w = galley.size().x + 14.0;
        if x + w > max_x {
            break;
        }
        let chip = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, chip_h));
        painter.rect_filled(chip, 12.0, bg);
        painter.galley(
            chip.min + egui::vec2(7.0, (chip_h - galley.size().y) / 2.0),
            galley,
            egui::Color32::WHITE,
        );
        x += w + 6.0;
    }
}

fn draw_empty_predictions_chip(ui: &egui::Ui, img_resp: &egui::Response, lang: &Lang) {
    let painter = ui.painter_at(img_resp.rect);
    let text = translate(Key::no_predictions, lang);
    draw_corner_chip(&painter, img_resp.rect, text, 13.0, 24.0, 12.0);
}

fn bbox_tooltip_ui(ui: &mut egui::Ui, detection: &XYXYc, lang: &Lang) {
    let color = super::class_color32(bbox_color_id(detection));
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("■").color(color).monospace());
        ui.label(egui::RichText::new(&detection.label).strong());
    });
    ui.label(format!(
        "{:.0}{}",
        detection.xyxy.prob * 100.0,
        translate(Key::confidence_pct, lang)
    ));
    let width_px = (detection.xyxy.x2 - detection.xyxy.x1).max(0.0).round() as i32;
    let height_px = (detection.xyxy.y2 - detection.xyxy.y1).max(0.0).round() as i32;
    ui.label(
        egui::RichText::new(format!("{} × {} px", width_px, height_px))
            .weak()
            .small(),
    );
    refined_extras_ui(ui, detection.extra_cls.as_ref(), lang);
}

fn seg_tooltip_ui(ui: &mut egui::Ui, segment: &SEGc, lang: &Lang) {
    let color = super::class_color32(bbox_color_id(&segment.bbox));
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("■").color(color).monospace());
        ui.label(egui::RichText::new(&segment.bbox.label).strong());
        ui.label(
            egui::RichText::new(format!("· {}", translate(Key::segment, lang)))
                .weak()
                .small(),
        );
    });
    ui.label(format!(
        "{:.0}{}",
        segment.bbox.xyxy.prob * 100.0,
        translate(Key::confidence_pct, lang)
    ));
    let width_px = (segment.bbox.xyxy.x2 - segment.bbox.xyxy.x1).max(0.0).round() as i32;
    let height_px = (segment.bbox.xyxy.y2 - segment.bbox.xyxy.y1).max(0.0).round() as i32;
    ui.label(
        egui::RichText::new(format!("bbox {} × {} px", width_px, height_px))
            .weak()
            .small(),
    );
    ui.label(
        egui::RichText::new(format!("mask {} × {}", segment.mask.width, segment.mask.height))
            .weak()
            .small(),
    );
    refined_extras_ui(ui, segment.bbox.extra_cls.as_ref(), lang);
}

/// The "Refined: ..." block shown under a bbox/segment tooltip when
/// `extra_cls` (a secondary classifier's output) is present.
fn refined_extras_ui(ui: &mut egui::Ui, extra_cls: Option<&Vec<Prob>>, lang: &Lang) {
    let Some(extras) = extra_cls else { return; };
    if extras.is_empty() {
        return;
    }
    ui.separator();
    ui.label(egui::RichText::new(translate(Key::refined, lang)).strong());
    let mut sorted: Vec<&Prob> = extras.iter().collect();
    sorted.sort_by(|left, right| {
        right.prob
            .partial_cmp(&left.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for prediction in sorted.iter().take(4) {
        super::tooltip_row(ui, prediction.class_id, &prediction.label, prediction.prob);
    }
}

fn classification_tooltip_ui(ui: &mut egui::Ui, probs: &[Prob], lang: &Lang) {
    if probs.is_empty() {
        ui.label(egui::RichText::new(translate(Key::no_predictions, lang)).weak());
        return;
    }
    ui.label(egui::RichText::new(translate(Key::classification, lang)).strong());
    let mut sorted: Vec<&Prob> = probs.iter().collect();
    sorted.sort_by(|left, right| {
        right.prob
            .partial_cmp(&left.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for prediction in sorted.iter().take(6) {
        super::tooltip_row(ui, prediction.class_id, &prediction.label, prediction.prob);
    }
    if probs.len() > 6 {
        ui.label(
            egui::RichText::new(super::and_more(probs.len() - 6, lang)).weak(),
        );
    }
}
