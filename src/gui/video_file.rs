use super::{imgbuf_to_texture, Gui};
use crate::api::abstractions::*;
use crate::api::bq::process_imgbuf;
use crate::api::render::*;
use crate::api::rest::{detect_remotely, rgb_image_to_jpeg_buffer};
use crate::api::video_file;
use crate::localization::*;
use image::{ImageBuffer, Rgb};
use std::sync::Arc;
use std::time::Instant;

const THUMB_W: u32 = 480;

/// One analysed frame, posted from the worker task back to the UI thread.
pub(super) struct AnalysisFrame {
    pub frame_idx: u64,
    pub aioutput: AIOutputs,
    pub thumbnail_jpeg: Vec<u8>,
}

pub(super) struct ExportProgress {
    pub current: u64,
    pub total: u64,
    pub done: bool,
    pub ok: bool,
}

impl Gui {
    // ---------- analysis ----------

    pub(super) fn start_video_analysis(&mut self) {
        // Re-analysis: wipe cache and re-open the decoder.
        let needs_fresh_decoder = self
            .video_pred
            .as_ref()
            .map(|p| p.wasprocessed)
            .unwrap_or(false)
            || self
                .video_file_processor
                .lock()
                .ok()
                .map(|g| g.is_none())
                .unwrap_or(true);

        if let Some(pv) = self.video_pred.as_mut() {
            if pv.wasprocessed {
                pv.reset();
                self.video_thumbnails.clear();
                self.video_playhead_frame = Some(0);
                self.video_last_displayed_frame = None;
            }
            pv.set_step(self.video_step_frame as u32);
        }

        if needs_fresh_decoder {
            if let Some(pv) = self.video_pred.as_ref() {
                if let Some(path) = pv.file_path.to_str() {
                    let fresh = video_file::VideofileProcessor::new(path);
                    *self.video_file_processor.lock().unwrap() = Some(fresh);
                }
            }
        }

        self.video_state.is_processing = true;
        self.video_state.progress_bar = self
            .video_pred
            .as_ref()
            .map(|p| p.get_progress())
            .unwrap_or(0.0);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AnalysisFrame>();
        self.video_processing_receiver = Some(rx);

        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        self.video_state.cancel_sender = Some(cancel_tx);

        let processor = Arc::clone(&self.video_file_processor);
        let api_endpoint = self.get_endpoint();
        let is_remote = !self.ep_selected.is_local();
        let step = self.video_step_frame.max(1) as u64;

        tokio::spawn(async move {
            loop {
                if cancel_rx.try_recv().is_ok() {
                    break;
                }
                let next = {
                    let mut guard = processor.lock().unwrap();
                    guard.as_mut().and_then(|p| p.next())
                };
                let (frame_idx, mut img) = match next {
                    Some(item) => item,
                    None => break,
                };
                if frame_idx % step != 0 {
                    continue;
                }
                let aioutput = if is_remote {
                    let buffer = rgb_image_to_jpeg_buffer(&img, 95);
                    match detect_remotely(api_endpoint.as_ref().unwrap(), buffer).await {
                        Ok(r) => r,
                        Err(_) => break,
                    }
                } else {
                    match tokio::task::spawn_blocking(move || {
                        let result = process_imgbuf(&img);
                        (img, result)
                    })
                    .await
                    {
                        Ok((returned, result)) => {
                            img = returned;
                            result
                        }
                        Err(_) => break,
                    }
                };
                let thumb = thumbnail_with_overlay(&img, &aioutput);
                let jpeg = rgb_image_to_jpeg_buffer(&thumb, 80);
                if tx
                    .send(AnalysisFrame {
                        frame_idx,
                        aioutput,
                        thumbnail_jpeg: jpeg,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
    }

    pub(super) fn cancel_video_processing(&mut self) {
        self.video_state.cancel();
        self.video_processing_receiver = None;
    }

    pub(super) fn video_handle_results(&mut self, ui: &egui::Ui) {
        let mut messages: Vec<AnalysisFrame> = Vec::new();
        let mut channel_closed = false;
        if let Some(rx) = &mut self.video_processing_receiver {
            loop {
                match rx.try_recv() {
                    Ok(msg) => messages.push(msg),
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        channel_closed = true;
                        break;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                }
            }
        }
        if messages.is_empty() && !channel_closed {
            return;
        }

        let mut latest_idx: Option<u64> = None;
        for msg in messages {
            if let Some(pv) = self.video_pred.as_mut() {
                pv.record(msg.frame_idx, msg.aioutput);
            }
            self.video_thumbnails
                .insert(msg.frame_idx, msg.thumbnail_jpeg);
            latest_idx = Some(msg.frame_idx);
        }
        if let Some(idx) = latest_idx {
            self.video_playhead_frame = Some(idx);
            self.refresh_video_texture(ui, idx);
        }
        if let Some(pv) = self.video_pred.as_ref() {
            self.video_state.progress_bar = pv.get_progress();
        }
        if channel_closed {
            self.video_state.progress_bar = 1.0;
            self.video_state.is_processing = false;
            self.video_processing_receiver = None;
            if let Some(pv) = self.video_pred.as_mut() {
                pv.wasprocessed = true;
            }
            self.process_done();
        }
        ui.request_repaint();
    }

    // ---------- playback ----------

    pub(super) fn start_video_playback(&mut self, start_frame: u64) {
        let Some(pv) = self.video_pred.as_ref() else { return; };
        if pv.n_frames == 0 {
            return;
        }
        self.video_play_start = Some(Instant::now());
        self.video_play_start_frame = start_frame.min(pv.n_frames.saturating_sub(1));
        self.video_playhead_frame = Some(self.video_play_start_frame);
        self.video_playing = true;
        self.video_last_displayed_frame = None;
    }

    pub(super) fn stop_video_playback(&mut self) {
        self.video_playing = false;
        self.video_play_start = None;
    }

    fn nearest_thumbnail_frame(&self, target: u64) -> Option<u64> {
        if self.video_thumbnails.is_empty() {
            return None;
        }
        let pv = self.video_pred.as_ref()?;
        let step = pv.step.max(1) as u64;
        let mut candidate = (target / step) * step;
        loop {
            if self.video_thumbnails.contains_key(&candidate) {
                return Some(candidate);
            }
            if candidate == 0 {
                break;
            }
            candidate = candidate.saturating_sub(step);
        }
        let mut candidate = ((target / step) + 1) * step;
        while candidate < pv.n_frames {
            if self.video_thumbnails.contains_key(&candidate) {
                return Some(candidate);
            }
            candidate = candidate.saturating_add(step);
        }
        None
    }

    fn refresh_video_texture(&mut self, ui: &egui::Ui, target_frame: u64) {
        let Some(src_frame) = self.nearest_thumbnail_frame(target_frame) else { return; };
        if self.video_last_displayed_frame == Some(src_frame) {
            return;
        }
        let Some(jpeg) = self.video_thumbnails.get(&src_frame) else { return; };
        let Ok(dynimg) = image::load_from_memory(jpeg) else { return; };
        let rgba = dynimg.to_rgba8();
        self.video_state.texture = imgbuf_to_texture(&rgba, ui);
        self.video_last_displayed_frame = Some(src_frame);
    }

    // ---------- export ----------

    pub(super) fn export_video_predictions_json(&mut self) {
        let Some(pv) = self.video_pred.as_ref() else { return; };
        let pv_clone = pv.clone();
        tokio::spawn(async move {
            let _ = pv_clone.write_pred_video_to_file().await;
        });
        self.process_done();
    }

    pub(super) fn export_video_annotated(&mut self) {
        let Some(pv) = self.video_pred.as_ref() else { return; };
        let output_path = video_file::get_output_path(pv.file_path.to_str().unwrap_or(""));
        let pv_clone = pv.clone();
        let (tx, rx) = std::sync::mpsc::channel::<ExportProgress>();
        self.video_export_receiver = Some(rx);
        self.video_state.progress_bar = 0.0;

        tokio::spawn(async move {
            let tx_progress = tx.clone();
            let total = pv_clone.n_frames;
            let result = tokio::task::spawn_blocking(move || {
                video_file::export_video_with_predictions(
                    &pv_clone,
                    &output_path,
                    |i, n| {
                        let _ = tx_progress.send(ExportProgress {
                            current: i,
                            total: n,
                            done: false,
                            ok: true,
                        });
                    },
                )
            })
            .await;
            let ok = matches!(result, Ok(Ok(_)));
            let _ = tx.send(ExportProgress {
                current: total,
                total,
                done: true,
                ok,
            });
        });
    }

    pub(super) fn video_handle_export(&mut self, ui: &egui::Ui) {
        let Some(rx) = self.video_export_receiver.as_ref() else { return; };
        let mut done = false;
        let mut ok = true;
        while let Ok(msg) = rx.try_recv() {
            if msg.total > 0 {
                self.video_state.progress_bar = msg.current as f32 / msg.total as f32;
            }
            if msg.done {
                done = true;
                ok = msg.ok;
            }
        }
        if done {
            self.video_export_receiver = None;
            self.video_state.progress_bar = 0.0;
            if ok {
                self.process_done();
            } else {
                self.process_error();
            }
        }
        ui.request_repaint();
    }

    // ---------- left-panel widget (Analyze / Export buttons) ----------

    pub(super) fn video_analysis_widget(&mut self, ui: &mut egui::Ui) {
        let has_video = self.video_pred.is_some();
        let is_image_capable_ep = self.is_image_model() || !self.ep_selected.is_local();
        if !has_video || !is_image_capable_ep {
            return;
        }

        ui.vertical_centered(|ui| {
            ui.heading(self.t(Key::video_file));
            ui.heading(self.t(Key::analysis));
        });
        ui.separator();

        if let Some(pv) = self.video_pred.as_ref() {
            if let Some(name) = pv.file_path.file_name().and_then(|n| n.to_str()) {
                ui.label(name);
            }
        }

        ui.vertical_centered(|ui| {
            ui.add_enabled_ui(!self.video_state.is_processing, |ui| {
                if ui.button("⚙").clicked() {
                    self.show_config.video = !self.show_config.video;
                }
                if self.show_config.video {
                    ui.label(self.t(Key::freq));
                    ui.style_mut().spacing.slider_width = 125.0;
                    ui.add(egui::Slider::new(&mut self.video_step_frame, 1..=90));
                    ui.add_space(8.0);
                }
            });

            if !self.video_state.is_processing {
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::analyze)))
                    .clicked()
                {
                    self.start_video_analysis();
                }
            } else if ui
                .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::cancel)))
                .clicked()
            {
                self.cancel_video_processing();
            }
        });

        ui.add(
            egui::ProgressBar::new(self.video_state.progress_bar)
                .show_percentage()
                .animate(self.video_state.is_processing || self.video_export_receiver.is_some()),
        );
        ui.add_space(8.0);

        let analysed_anything = self
            .video_pred
            .as_ref()
            .map(|pv| pv.processed_count() > 0)
            .unwrap_or(false);

        if analysed_anything && !self.video_state.is_processing {
            ui.vertical_centered(|ui| {
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::export)))
                    .clicked()
                {
                    self.dialog = super::OpenDialog::Export;
                }
            });
        }

        self.video_export_dialog(ui);
    }

    fn video_export_dialog(&mut self, ui: &egui::Ui) {
        if self.dialog != super::OpenDialog::Export || self.mode != super::Mode::Video {
            return;
        }
        let mut close = false;
        egui::Window::new(self.t(Key::export))
            .collapsible(false)
            .resizable(false)
            .show(ui, |ui| {
                if ui.button(self.t(Key::export_predictions)).clicked() {
                    self.export_video_predictions_json();
                    close = true;
                }
                if ui
                    .button(self.t(Key::export_video_with_predictions))
                    .clicked()
                {
                    self.export_video_annotated();
                    close = true;
                }
                if ui.button(self.t(Key::cancel)).clicked() {
                    close = true;
                }
            });
        if close {
            self.dialog = super::OpenDialog::None;
        }
    }

    // ---------- main video-player UI ----------

    pub(super) fn ui_video(&mut self, ui: &mut egui::Ui) {
        let Some(n_frames) = self.video_pred.as_ref().map(|p| p.n_frames) else { return; };
        if n_frames == 0 {
            return;
        }
        let fps = self.video_pred.as_ref().map(|p| p.fps).unwrap_or(30.0).max(0.1);
        let last_frame = n_frames.saturating_sub(1);

        // Advance the playback clock at fps rate. Texture refresh is throttled
        // to step transitions so we don't burn CPU re-decoding the same JPEG.
        if self.video_playing {
            if let Some(start) = self.video_play_start {
                let elapsed = start.elapsed().as_secs_f64();
                let frame_now = self.video_play_start_frame as f64 + elapsed * fps;
                let idx = (frame_now as u64).min(last_frame);
                self.video_playhead_frame = Some(idx);
                if idx >= last_frame {
                    self.stop_video_playback();
                }
                self.refresh_video_texture(ui, idx);
            }
            ui.ctx().request_repaint();
        }

        // Big frame preview
        let avail = ui.available_size_before_wrap();
        let controls_h: f32 = 28.0;
        let seek_h: f32 = 14.0;
        let strip_h: f32 = 6.0;
        let gap: f32 = 6.0;
        let bottom_chrome = controls_h + seek_h + strip_h + gap * 2.0;
        let preview_h = (avail.y - bottom_chrome - 12.0).max(200.0);

        ui.vertical_centered(|ui| {
            if let Some(tex) = self.video_state.texture.clone() {
                ui.add(
                    egui::Image::new(&tex)
                        .max_height(preview_h)
                        .corner_radius(8.0),
                );
            } else {
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(avail.x * 0.6, preview_h),
                    egui::Sense::hover(),
                );
                let p = ui.painter_at(rect);
                let bg = if ui.visuals().dark_mode {
                    egui::Color32::from_rgb(28, 32, 36)
                } else {
                    egui::Color32::from_rgb(228, 230, 234)
                };
                p.rect_filled(rect, 8.0, bg);
                p.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Run analysis to populate the timeline",
                    egui::FontId::proportional(14.0),
                    ui.visuals().weak_text_color(),
                );
            }
        });

        ui.add_space(gap);

        // Controls row: play / step / time
        let playhead = self.video_playhead_frame.unwrap_or(0).min(last_frame);
        ui.horizontal(|ui| {
            if self.video_playing {
                if ui.button("⏸").on_hover_text("Pause").clicked() {
                    self.stop_video_playback();
                }
            } else if ui.button("▶").on_hover_text("Play").clicked() {
                let start = if playhead >= last_frame { 0 } else { playhead };
                self.start_video_playback(start);
            }
            if ui
                .button("⏮")
                .on_hover_text("Previous analysed frame")
                .clicked()
            {
                if let Some(prev) = prev_analysed_frame(self.video_pred.as_ref(), playhead) {
                    self.stop_video_playback();
                    self.video_playhead_frame = Some(prev);
                    self.refresh_video_texture(ui, prev);
                }
            }
            if ui
                .button("⏭")
                .on_hover_text("Next analysed frame")
                .clicked()
            {
                if let Some(next) = next_analysed_frame(self.video_pred.as_ref(), playhead) {
                    self.stop_video_playback();
                    self.video_playhead_frame = Some(next);
                    self.refresh_video_texture(ui, next);
                }
            }

            ui.add_space(8.0);
            ui.label(format_time_pair(playhead as f64 / fps, last_frame as f64 / fps));
            ui.add_space(8.0);
            if let Some(label) = current_top_label(self.video_pred.as_ref(), playhead) {
                ui.label(egui::RichText::new(label).strong());
            }
        });

        ui.add_space(gap);

        // Seek bar — thin, classic player style
        let new_playhead = self.draw_seek_bar(ui, n_frames, seek_h, strip_h, playhead);
        if let Some(idx) = new_playhead {
            let was_playing = self.video_playing;
            self.stop_video_playback();
            self.video_playhead_frame = Some(idx);
            self.refresh_video_texture(ui, idx);
            if was_playing {
                self.start_video_playback(idx);
            }
        }

        self.video_handle_results(ui);
        self.video_handle_export(ui);
    }

    /// Returns Some(frame_index) when the user clicks or drags on the bar.
    fn draw_seek_bar(
        &self,
        ui: &mut egui::Ui,
        n_frames: u64,
        seek_h: f32,
        strip_h: f32,
        playhead: u64,
    ) -> Option<u64> {
        let avail_w = ui.available_width();
        let total_h = seek_h + strip_h + 2.0;
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(avail_w, total_h),
            egui::Sense::click_and_drag(),
        );
        let p = ui.painter_at(rect);
        let dark = ui.visuals().dark_mode;

        let bar_top = rect.top() + 1.0;
        let bar_bottom = bar_top + seek_h;
        let strip_top = bar_bottom + 2.0;
        let strip_bottom = strip_top + strip_h;

        let bar_rect = egui::Rect::from_min_max(
            egui::pos2(rect.left(), bar_top),
            egui::pos2(rect.right(), bar_bottom),
        );
        let strip_rect = egui::Rect::from_min_max(
            egui::pos2(rect.left(), strip_top),
            egui::pos2(rect.right(), strip_bottom),
        );

        // Background of seek bar
        let bar_bg = if dark {
            egui::Color32::from_gray(55)
        } else {
            egui::Color32::from_gray(210)
        };
        p.rect_filled(bar_rect, 3.0, bar_bg);

        // Played portion
        let t_played = playhead as f32 / n_frames.saturating_sub(1).max(1) as f32;
        let played_x = bar_rect.left() + bar_rect.width() * t_played;
        let played_rect = egui::Rect::from_min_max(
            bar_rect.left_top(),
            egui::pos2(played_x, bar_rect.bottom()),
        );
        let played_color = if dark {
            egui::Color32::from_rgb(110, 180, 255)
        } else {
            egui::Color32::from_rgb(70, 120, 220)
        };
        p.rect_filled(played_rect, 3.0, played_color);

        // Predictions strip below — one column per pixel.
        if let Some(pv) = self.video_pred.as_ref() {
            let strip_bg = if dark {
                egui::Color32::from_gray(45)
            } else {
                egui::Color32::from_gray(225)
            };
            p.rect_filled(strip_rect, 1.0, strip_bg);

            let pixels = strip_rect.width().max(1.0) as i32;
            for col in 0..pixels {
                let t = col as f64 / pixels as f64;
                let frame = (t * n_frames.saturating_sub(1).max(1) as f64) as u64;
                if let Some(aio) = pv.prediction_at(frame) {
                    if let Some((cid, _, _)) = aio.dominant_prob() {
                        let c = class_color(cid);
                        let color = egui::Color32::from_rgb(c[0], c[1], c[2]);
                        let x = strip_rect.left() + col as f32;
                        p.line_segment(
                            [egui::pos2(x, strip_top), egui::pos2(x, strip_bottom)],
                            egui::Stroke::new(1.0, color),
                        );
                    }
                }
            }
        }

        // Playhead handle
        let handle_color = if dark {
            egui::Color32::WHITE
        } else {
            egui::Color32::from_rgb(20, 20, 20)
        };
        p.circle_filled(
            egui::pos2(played_x, bar_rect.center().y),
            (seek_h * 0.5).min(8.0),
            handle_color,
        );

        // Hover tooltip with prediction details at the hovered time.
        if let Some(hover_pos) = response.hover_pos() {
            let t = ((hover_pos.x - bar_rect.left()) / bar_rect.width()).clamp(0.0, 1.0);
            let frame = (t as f64 * n_frames.saturating_sub(1).max(1) as f64) as u64;
            let fps = self.video_pred.as_ref().map(|p| p.fps).unwrap_or(30.0).max(0.1);
            let secs = frame as f64 / fps;

            // Hover marker line.
            let hover_color = if dark {
                egui::Color32::from_white_alpha(120)
            } else {
                egui::Color32::from_black_alpha(80)
            };
            p.line_segment(
                [egui::pos2(hover_pos.x, bar_top), egui::pos2(hover_pos.x, bar_bottom)],
                egui::Stroke::new(1.0, hover_color),
            );

            let tooltip_text = hover_tooltip(self.video_pred.as_ref(), frame, secs);
            response.clone().on_hover_ui_at_pointer(|ui| {
                ui.label(tooltip_text);
            });
        }

        // Click or drag → seek.
        if response.clicked() || response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let t = ((pos.x - bar_rect.left()) / bar_rect.width()).clamp(0.0, 1.0);
                let frame = (t as f64 * n_frames.saturating_sub(1).max(1) as f64) as u64;
                return Some(frame);
            }
        }
        None
    }
}

// ---------- helpers (no Gui borrow) ----------

fn thumbnail_with_overlay(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    aioutput: &AIOutputs,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut small = if img.width() <= THUMB_W {
        img.clone()
    } else {
        let h = (img.height() as f32 * THUMB_W as f32 / img.width() as f32) as u32;
        image::imageops::resize(img, THUMB_W, h.max(1), image::imageops::FilterType::Triangle)
    };
    let sx = small.width() as f32 / img.width() as f32;
    let sy = small.height() as f32 / img.height() as f32;
    let scaled = scale_aioutput(aioutput, sx, sy);
    if !scaled.is_empty() {
        draw_aioutput(&mut small, &scaled);
    }
    small
}

fn scale_aioutput(aio: &AIOutputs, sx: f32, sy: f32) -> AIOutputs {
    match aio {
        AIOutputs::ObjectDetection(bboxes) => {
            let scaled: Vec<XYXYc> = bboxes
                .iter()
                .map(|b| {
                    let mut nb = b.clone();
                    nb.xyxy.x1 *= sx;
                    nb.xyxy.x2 *= sx;
                    nb.xyxy.y1 *= sy;
                    nb.xyxy.y2 *= sy;
                    nb
                })
                .collect();
            AIOutputs::ObjectDetection(scaled)
        }
        AIOutputs::Segmentation(segs) => {
            let scaled: Vec<SEGc> = segs
                .iter()
                .map(|s| {
                    let mut ns = s.clone();
                    ns.bbox.xyxy.x1 *= sx;
                    ns.bbox.xyxy.x2 *= sx;
                    ns.bbox.xyxy.y1 *= sy;
                    ns.bbox.xyxy.y2 *= sy;
                    ns
                })
                .collect();
            AIOutputs::Segmentation(scaled)
        }
        AIOutputs::Classification(_) | AIOutputs::AudioClassification(_) => aio.clone(),
    }
}

fn format_time(secs: f64) -> String {
    let mins = (secs / 60.0).floor() as i64;
    let s = secs - mins as f64 * 60.0;
    if mins > 0 {
        format!("{}:{:05.2}", mins, s)
    } else {
        format!("0:{:05.2}", s)
    }
}

fn format_time_pair(now: f64, total: f64) -> String {
    format!("{} / {}", format_time(now), format_time(total))
}

fn prev_analysed_frame(pv: Option<&PredVideo>, current: u64) -> Option<u64> {
    let pv = pv?;
    if current == 0 {
        return None;
    }
    pv.last_processed_at_or_before(current.saturating_sub(1))
}

fn next_analysed_frame(pv: Option<&PredVideo>, current: u64) -> Option<u64> {
    let pv = pv?;
    let step = pv.step.max(1) as u64;
    let mut candidate = ((current / step) + 1) * step;
    while candidate < pv.n_frames {
        if pv.frames.get(candidate as usize).and_then(|s| s.as_ref()).is_some() {
            return Some(candidate);
        }
        candidate = candidate.saturating_add(step);
    }
    None
}

fn current_top_label(pv: Option<&PredVideo>, frame: u64) -> Option<String> {
    let pv = pv?;
    let aio = pv.prediction_at(frame)?;
    let (_, label, prob) = aio.dominant_prob()?;
    Some(format!("{}  ·  {:.0}%", label, prob * 100.0))
}

fn hover_tooltip(pv: Option<&PredVideo>, frame: u64, secs: f64) -> String {
    let mut lines = vec![format!("{}  ·  frame {}", format_time(secs), frame)];
    let Some(pv) = pv else { return lines.join("\n"); };
    let Some(aio) = pv.prediction_at(frame) else { return lines.join("\n"); };

    let mut shown: Vec<String> = Vec::new();
    match aio {
        AIOutputs::ObjectDetection(bboxes) => {
            let mut bs: Vec<&XYXYc> = bboxes.iter().collect();
            bs.sort_by(|a, b| {
                b.xyxy.prob.partial_cmp(&a.xyxy.prob).unwrap_or(std::cmp::Ordering::Equal)
            });
            for b in bs.iter().take(5) {
                shown.push(format!("  {} — {:.0}%", b.label, b.xyxy.prob * 100.0));
            }
        }
        AIOutputs::Segmentation(segs) => {
            let mut bs: Vec<&SEGc> = segs.iter().collect();
            bs.sort_by(|a, b| {
                b.bbox
                    .xyxy
                    .prob
                    .partial_cmp(&a.bbox.xyxy.prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for s in bs.iter().take(5) {
                shown.push(format!("  {} — {:.0}%", s.bbox.label, s.bbox.xyxy.prob * 100.0));
            }
        }
        AIOutputs::Classification(probs) => {
            let mut ps: Vec<&Prob> = probs.iter().collect();
            ps.sort_by(|a, b| {
                b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal)
            });
            for p in ps.iter().take(5) {
                shown.push(format!("  {} — {:.0}%", p.label, p.prob * 100.0));
            }
        }
        AIOutputs::AudioClassification(_) => {}
    }
    if !shown.is_empty() {
        lines.push(String::new());
        lines.push("AI predictions:".into());
        lines.extend(shown);
    }
    lines.join("\n")
}
