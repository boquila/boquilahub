use super::{imgbuf_to_texture, Gui};
use crate::api::abstractions::*;
use crate::api::bq::process_imgbuf;
use crate::api::export;
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

    /// Decode the source video and regenerate thumbnails from already-cached
    /// predictions (no model calls). Triggered automatically after picking a
    /// video whose sidecar `_predictions.json` was loaded by `PredVideo::new_simple`
    /// so playback works right away instead of being gated behind a re-Analyse.
    pub(super) fn start_video_thumbnail_regen(&mut self) {
        let Some(pv) = self.video_pred.as_ref() else { return; };
        if pv.processed_count() == 0 {
            return;
        }
        let Some(path_str) = pv.file_path.to_str().map(|s| s.to_string()) else { return; };
        let cached_frames: Vec<Option<AIOutputs>> = pv.frames.clone();

        self.video_state.is_processing = true;
        self.video_state.progress_bar = 0.0;
        self.video_thumbnails.clear();
        self.video_last_displayed_frame = None;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AnalysisFrame>();
        self.video_processing_receiver = Some(rx);

        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        self.video_state.cancel_sender = Some(cancel_tx);

        let fresh = video_file::VideofileProcessor::new(&path_str);
        *self.video_file_processor.lock().unwrap() = Some(fresh);
        let processor = Arc::clone(&self.video_file_processor);

        tokio::spawn(async move {
            loop {
                if cancel_rx.try_recv().is_ok() {
                    break;
                }
                let next = {
                    let mut guard = processor.lock().unwrap();
                    guard.as_mut().and_then(|p| p.next())
                };
                let (frame_idx, img) = match next {
                    Some(item) => item,
                    None => break,
                };
                let Some(Some(aio)) = cached_frames.get(frame_idx as usize) else { continue; };
                let thumb = thumbnail_with_overlay(&img, aio);
                let jpeg = rgb_image_to_jpeg_buffer(&thumb, 80);
                if tx
                    .send(AnalysisFrame {
                        frame_idx,
                        aioutput: aio.clone(),
                        thumbnail_jpeg: jpeg,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
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
        let display = pv
            .predictions_file_path()
            .ok()
            .and_then(|p| p.to_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "predictions.json".to_string());
        let pv_clone = pv.clone();
        tokio::spawn(async move {
            let _ = pv_clone.write_pred_video_to_file().await;
        });
        self.process_done_at(display);
    }

    pub(super) fn export_video_annotated(&mut self) {
        let Some(pv) = self.video_pred.as_ref() else { return; };
        let output_path = export::prepare_export_video(&pv.file_path);
        self.video_export_path = output_path.to_str().map(|s| s.to_string());
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
                if let Some(path) = self.video_export_path.take() {
                    self.process_done_at(path);
                } else {
                    self.process_done();
                }
            } else {
                self.process_error();
                self.video_export_path = None;
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

    fn draw_video_header(&self, ui: &mut egui::Ui) {
        let Some(pv) = self.video_pred.as_ref() else { return; };
        let name = pv
            .file_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("(unknown)");
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(name).strong());
            if pv.processed_count() > 0 {
                ui.separator();
                let total = (0..pv.n_frames).step_by(pv.step.max(1) as usize).count();
                ui.label(
                    egui::RichText::new(format!(
                        "{} / {} frames analysed",
                        pv.processed_count(),
                        total.max(1),
                    ))
                    .weak()
                    .small(),
                );
            } else {
                ui.separator();
                ui.label(
                    egui::RichText::new("not analysed")
                        .weak()
                        .small(),
                );
            }
        });
        ui.add_space(4.0);
    }

    pub(super) fn ui_video(&mut self, ui: &mut egui::Ui) {
        let Some(n_frames) = self.video_pred.as_ref().map(|p| p.n_frames) else { return; };
        if n_frames == 0 {
            return;
        }
        self.draw_video_header(ui);
        let fps = self.video_pred.as_ref().map(|p| p.fps).unwrap_or(30.0).max(0.1);
        let last_frame = n_frames.saturating_sub(1);
        // Playback only makes sense once we have at least one analysed frame
        // (that's where the displayable thumbnails come from). Until then we
        // show the first frame statically — no scrub bar, no play button —
        // so the user isn't lured into clicking controls that can't respond.
        let has_playable_content = !self.video_thumbnails.is_empty();
        // The user can only scrub / play through what's been analysed so far.
        // While analysis is running, this grows; once it's complete, this is
        // the last frame.
        let scrub_limit = self
            .video_pred
            .as_ref()
            .and_then(|p| p.max_processed_frame())
            .unwrap_or(0);

        if self.video_playing {
            if let Some(start) = self.video_play_start {
                let elapsed = start.elapsed().as_secs_f64();
                let frame_now = self.video_play_start_frame as f64 + elapsed * fps;
                let idx = (frame_now as u64).min(scrub_limit);
                self.video_playhead_frame = Some(idx);
                if idx >= scrub_limit {
                    self.stop_video_playback();
                }
                self.refresh_video_texture(ui, idx);
            }
            ui.ctx().request_repaint();
        }

        let avail = ui.available_size_before_wrap();
        let controls_h: f32 = 28.0;
        let seek_h: f32 = 14.0;
        let strip_h: f32 = 6.0;
        let gap: f32 = 6.0;
        let bottom_chrome = if has_playable_content {
            controls_h + seek_h + strip_h + gap * 2.0
        } else {
            24.0
        };
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
            }
        });

        ui.add_space(gap);

        if !has_playable_content {
            ui.vertical_centered(|ui| {
                let hint = if self.video_state.is_processing {
                    "Analysing…"
                } else {
                    "Analyse the video to enable playback"
                };
                ui.label(egui::RichText::new(hint).color(ui.visuals().weak_text_color()));
            });
            self.video_handle_results(ui);
            self.video_handle_export(ui);
            return;
        }

        // While analysis (or an export) is running, lock the player so the user
        // can't scrub / play / step into half-baked state.
        let controls_active = !self.video_state.is_processing
            && self.video_export_receiver.is_none();
        if !controls_active && self.video_playing {
            self.stop_video_playback();
        }

        let playhead = self.video_playhead_frame.unwrap_or(0).min(scrub_limit);
        let mut new_playhead: Option<u64> = None;

        ui.add_enabled_ui(controls_active, |ui| {
            ui.horizontal(|ui| {
                if self.video_playing {
                    if ui.button("⏸").on_hover_text("Pause").clicked() {
                        self.stop_video_playback();
                    }
                } else if ui.button("▶").on_hover_text("Play").clicked() {
                    let start = if playhead >= scrub_limit { 0 } else { playhead };
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
                        if next <= scrub_limit {
                            self.stop_video_playback();
                            self.video_playhead_frame = Some(next);
                            self.refresh_video_texture(ui, next);
                        }
                    }
                }

                ui.add_space(8.0);
                let suffix = if self.video_state.is_processing {
                    "  ·  analysing…"
                } else {
                    ""
                };
                ui.label(format!(
                    "{}  ·  analysed up to {}{}",
                    format_time_pair(playhead as f64 / fps, last_frame as f64 / fps),
                    format_time(scrub_limit as f64 / fps),
                    suffix,
                ));
                ui.add_space(8.0);
                if let Some(label) = current_top_label(self.video_pred.as_ref(), playhead) {
                    ui.label(egui::RichText::new(label).strong());
                }
            });

            ui.add_space(gap);

            new_playhead =
                self.draw_seek_bar(ui, n_frames, scrub_limit, seek_h, strip_h, playhead);
        });

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
    /// `scrub_limit` is the highest frame the user is allowed to seek to —
    /// anything past it is greyed out and ignored.
    fn draw_seek_bar(
        &self,
        ui: &mut egui::Ui,
        n_frames: u64,
        scrub_limit: u64,
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

        let max_idx = n_frames.saturating_sub(1).max(1) as f32;
        let t_limit = (scrub_limit as f32 / max_idx).clamp(0.0, 1.0);
        let limit_x = bar_rect.left() + bar_rect.width() * t_limit;

        // Available (analysed) bar background.
        let bar_bg = if dark {
            egui::Color32::from_gray(55)
        } else {
            egui::Color32::from_gray(210)
        };
        p.rect_filled(bar_rect, 3.0, bar_bg);

        // Locked tail (unanalysed) — clearly visually distinct so it reads
        // as "you can't go here".
        if limit_x < bar_rect.right() {
            let locked_rect = egui::Rect::from_min_max(
                egui::pos2(limit_x, bar_top),
                egui::pos2(bar_rect.right(), bar_bottom),
            );
            let locked_color = if dark {
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 130)
            } else {
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 70)
            };
            p.rect_filled(locked_rect, 3.0, locked_color);
        }

        // Played portion within the available range.
        let t_played = playhead as f32 / max_idx;
        let played_x = (bar_rect.left() + bar_rect.width() * t_played).min(limit_x);
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

        // Predictions strip below — dominant-class colour per pixel column
        // inside the analysed range; locked tail stays dim. Built as merged
        // rectangles (one per contiguous same-class run) so a 1000px-wide bar
        // costs ~tens of draw calls per repaint instead of 1000 line_segments
        // + 1000 walk-back lookups.
        if let Some(pv) = self.video_pred.as_ref() {
            let strip_bg = if dark {
                egui::Color32::from_gray(45)
            } else {
                egui::Color32::from_gray(225)
            };
            p.rect_filled(strip_rect, 1.0, strip_bg);

            let n_cols = strip_rect.width().max(1.0) as usize;
            for (col_start, col_end, class_id) in
                build_strip_segments(pv, scrub_limit, n_frames, n_cols)
            {
                let c = class_color(class_id);
                let x0 = strip_rect.left() + col_start as f32;
                let x1 = strip_rect.left() + (col_end + 1) as f32;
                p.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(x0, strip_top),
                        egui::pos2(x1, strip_bottom),
                    ),
                    0.0,
                    egui::Color32::from_rgb(c[0], c[1], c[2]),
                );
            }
        }

        // Limit marker — a vertical line where the user's freedom ends.
        if limit_x < bar_rect.right() - 1.0 {
            let limit_color = if dark {
                egui::Color32::from_rgb(220, 220, 220)
            } else {
                egui::Color32::from_rgb(80, 80, 80)
            };
            p.line_segment(
                [egui::pos2(limit_x, bar_top - 1.0), egui::pos2(limit_x, strip_bottom + 1.0)],
                egui::Stroke::new(1.0, limit_color),
            );
        }

        // Playhead handle.
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

        let enabled = ui.is_enabled();

        // Hover tooltip with prediction details — only when interactive AND
        // pointed at the analysed range. Hover line spans bar + strip so the
        // eye can follow vertical → class colour underneath.
        if enabled {
            if let Some(hover_pos) = response.hover_pos() {
                if hover_pos.x <= limit_x {
                    let t =
                        ((hover_pos.x - bar_rect.left()) / bar_rect.width()).clamp(0.0, 1.0);
                    let frame = ((t as f64 * n_frames.saturating_sub(1).max(1) as f64) as u64)
                        .min(scrub_limit);
                    let fps = self.video_pred.as_ref().map(|p| p.fps).unwrap_or(30.0).max(0.1);
                    let secs = frame as f64 / fps;

                    let hover_color = if dark {
                        egui::Color32::from_white_alpha(120)
                    } else {
                        egui::Color32::from_black_alpha(80)
                    };
                    p.line_segment(
                        [
                            egui::pos2(hover_pos.x, bar_top),
                            egui::pos2(hover_pos.x, strip_bottom),
                        ],
                        egui::Stroke::new(1.0, hover_color),
                    );

                    let pv = self.video_pred.as_ref();
                    response.clone().on_hover_ui_at_pointer(|ui| {
                        tooltip_ui(ui, pv, frame, secs);
                    });
                }
            }
        } else {
            // Dim the whole bar so it visually reads as "locked".
            let dim = if dark {
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 110)
            } else {
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 130)
            };
            p.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(rect.left(), bar_top - 1.0),
                    egui::pos2(rect.right(), strip_bottom + 1.0),
                ),
                3.0,
                dim,
            );
        }

        // Click or drag → seek, clamped to the analysed range. `Response`
        // already suppresses click/drag while the ui is disabled.
        if response.clicked() || response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let t = ((pos.x - bar_rect.left()) / bar_rect.width()).clamp(0.0, 1.0);
                let frame = ((t as f64 * n_frames.saturating_sub(1).max(1) as f64) as u64)
                    .min(scrub_limit);
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

/// One contiguous run of same-dominant-class pixel columns inside the analysed
/// range. `(col_start, col_end_inclusive, class_id)`.
fn build_strip_segments(
    pv: &PredVideo,
    scrub_limit: u64,
    n_frames: u64,
    n_cols: usize,
) -> Vec<(usize, usize, u32)> {
    if n_frames == 0 || n_cols == 0 {
        return Vec::new();
    }
    let max_idx = n_frames.saturating_sub(1).max(1) as f64;
    let limit_col = ((scrub_limit as f64 / max_idx) * n_cols as f64) as usize;
    let upper = n_cols.min(limit_col + 1);

    let mut segments: Vec<(usize, usize, u32)> = Vec::new();
    let mut current: Option<(usize, u32)> = None;
    // Dedup repeated lookups: adjacent pixels usually map to the same frame
    // (when n_cols > n_frames), or to the same nearest analysed frame (when
    // step > 1). Either way we avoid most of the prediction_at walk-backs.
    let mut last_frame: Option<u64> = None;
    let mut last_class: Option<u32> = None;

    for col in 0..upper {
        let t = col as f64 / n_cols as f64;
        let frame = (t * max_idx) as u64;
        let class = if last_frame == Some(frame) {
            last_class
        } else {
            last_frame = Some(frame);
            let c = pv
                .prediction_at(frame)
                .and_then(|aio| aio.dominant_prob().map(|(cid, _, _)| cid));
            last_class = c;
            c
        };
        match (current, class) {
            (Some((_, cur)), Some(c)) if cur == c => continue,
            (Some((start, cur)), Some(c)) => {
                segments.push((start, col - 1, cur));
                current = Some((col, c));
            }
            (Some((start, cur)), None) => {
                segments.push((start, col - 1, cur));
                current = None;
            }
            (None, Some(c)) => current = Some((col, c)),
            (None, None) => {}
        }
    }
    if let Some((start, cur)) = current {
        segments.push((start, upper.saturating_sub(1), cur));
    }
    segments
}

/// Build the hover tooltip body in-place. Mirrors `audio.rs` style:
/// strong header (time + frame), weak/small "nearest analysed" hint when the
/// cursor sits between analysed frames, then class-colour-coded rows for the
/// top predictions of whichever `AIOutputs` variant the model produced.
fn tooltip_ui(ui: &mut egui::Ui, pv: Option<&PredVideo>, frame: u64, secs: f64) {
    ui.label(
        egui::RichText::new(format!("{}  ·  frame {}", format_time(secs), frame)).strong(),
    );
    let Some(pv) = pv else { return; };
    let Some(nearest) = pv.last_processed_at_or_before(frame) else {
        ui.label(egui::RichText::new("(not analysed)").weak());
        return;
    };
    if nearest != frame {
        ui.label(
            egui::RichText::new(format!("nearest analysed: frame {}", nearest))
                .weak()
                .small(),
        );
    }
    let Some(aio) = pv.frames.get(nearest as usize).and_then(|f| f.as_ref()) else { return; };

    match aio {
        AIOutputs::ObjectDetection(bboxes) => {
            if bboxes.is_empty() {
                ui.label(egui::RichText::new("(no detections)").weak());
                return;
            }
            ui.separator();
            let n = bboxes.len();
            let header = if n == 1 { "1 detection".into() } else { format!("{} detections", n) };
            ui.label(egui::RichText::new(header).strong());
            let mut bs: Vec<&XYXYc> = bboxes.iter().collect();
            bs.sort_by(|a, b| {
                b.xyxy.prob.partial_cmp(&a.xyxy.prob).unwrap_or(std::cmp::Ordering::Equal)
            });
            for b in bs.iter().take(6) {
                tooltip_row(ui, b.xyxy.class_id, &b.label, b.xyxy.prob);
            }
            if n > 6 {
                ui.label(egui::RichText::new(format!("…and {} more", n - 6)).weak());
            }
        }
        AIOutputs::Segmentation(segs) => {
            if segs.is_empty() {
                ui.label(egui::RichText::new("(no detections)").weak());
                return;
            }
            ui.separator();
            let n = segs.len();
            let header = if n == 1 { "1 segment".into() } else { format!("{} segments", n) };
            ui.label(egui::RichText::new(header).strong());
            let mut ss: Vec<&SEGc> = segs.iter().collect();
            ss.sort_by(|a, b| {
                b.bbox
                    .xyxy
                    .prob
                    .partial_cmp(&a.bbox.xyxy.prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for s in ss.iter().take(6) {
                tooltip_row(ui, s.bbox.xyxy.class_id, &s.bbox.label, s.bbox.xyxy.prob);
            }
            if n > 6 {
                ui.label(egui::RichText::new(format!("…and {} more", n - 6)).weak());
            }
        }
        AIOutputs::Classification(probs) => {
            if probs.is_empty() {
                ui.label(egui::RichText::new("(no classification)").weak());
                return;
            }
            ui.separator();
            ui.label(egui::RichText::new("Classification").strong());
            let mut ps: Vec<&Prob> = probs.iter().collect();
            ps.sort_by(|a, b| {
                b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal)
            });
            for p in ps.iter().take(6) {
                tooltip_row(ui, p.class_id, &p.label, p.prob);
            }
            if probs.len() > 6 {
                ui.label(
                    egui::RichText::new(format!("…and {} more", probs.len() - 6)).weak(),
                );
            }
        }
        AIOutputs::AudioClassification(_) => {}
    }
}

fn tooltip_row(ui: &mut egui::Ui, class_id: u32, label: &str, prob: f32) {
    let c = class_color(class_id);
    let color = egui::Color32::from_rgb(c[0], c[1], c[2]);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("■").color(color).monospace());
        ui.label(format!("{} — {:.0}%", label, prob * 100.0));
    });
}
