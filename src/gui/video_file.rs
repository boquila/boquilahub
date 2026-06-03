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

const THUMB_W: u32 = 1280;

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
        // If a raw preview decode is running, tear it down — its decoder is
        // already partway through the file, and reusing it here would skip
        // every frame the preview already consumed.
        if !self.video_state.is_processing && self.video_state.is_active() {
            self.video_state.cancel();
            *self.video_file_processor.lock().unwrap() = None;
        }

        let step_u32 = self.video_step_frame as u32;
        let needs_fresh_decoder;
        let path_for_decoder: Option<String>;
        {
            let already_processed = self
                .current_video()
                .map(|p| p.wasprocessed)
                .unwrap_or(false);
            let decoder_missing = self
                .video_file_processor
                .lock()
                .ok()
                .map(|g| g.is_none())
                .unwrap_or(true);
            needs_fresh_decoder = already_processed || decoder_missing;
            path_for_decoder = self
                .current_video()
                .and_then(|p| p.file_path.to_str().map(str::to_owned));
        }

        let mut wiped = false;
        if let Some(pv) = self.current_video_mut() {
            if pv.wasprocessed {
                pv.reset();
                wiped = true;
            }
            pv.set_step(step_u32);
        }
        if wiped {
            self.video_thumbnails.clear();
            self.video_playhead_frame = Some(0);
            self.video_last_displayed_frame = None;
        }

        if needs_fresh_decoder {
            if let Some(path) = path_for_decoder {
                let fresh = video_file::VideofileProcessor::new(&path);
                *self.video_file_processor.lock().unwrap() = Some(fresh);
            }
        }

        self.video_state.progress_bar = self
            .current_video()
            .map(|p| p.frame_progress())
            .unwrap_or(0.0);

        let (tx, mut cancel_rx) = self.video_state.start();

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
    }

    /// Decode-only worker: streams frames at the current step, builds thumbnails
    /// (with overlays from any predictions already loaded — e.g. a sidecar JSON),
    /// and pushes them through the same channel `start_video_analysis` uses.
    /// No AI work, no `is_processing` flag — so playback controls stay live and
    /// the user can play / scrub a freshly-loaded video without selecting a model.
    pub(super) fn start_video_preview(&mut self) {
        if self.video_state.is_active() {
            return;
        }
        let step = self.video_step_frame.max(1) as u64;
        let step_u32 = self.video_step_frame as u32;
        let (path_str, predictions) = {
            let Some(pv) = self.current_video() else { return; };
            if pv.n_frames == 0 {
                return;
            }
            // Cheap done-check: if the last step-aligned frame is cached, the
            // decoder reached EOF on a prior run. Avoids an O(n_frames/step)
            // sweep on every UI tick once decoding is complete.
            let last_keyframe = (pv.n_frames.saturating_sub(1) / step) * step;
            if self.video_thumbnails.contains_key(&last_keyframe) {
                return;
            }
            let Some(path_str) = pv.file_path.to_str().map(str::to_owned) else { return; };
            (path_str, pv.frames.clone())
        };

        if let Some(pv) = self.current_video_mut() {
            pv.set_step(step_u32);
        }

        // Always open a fresh decoder — a prior analysis may have left the
        // processor exhausted, and seeking the existing one isn't supported.
        *self.video_file_processor.lock().unwrap() =
            Some(video_file::VideofileProcessor::new(&path_str));

        let tx = self.video_state.start_streaming();

        let processor = Arc::clone(&self.video_file_processor);
        let empty_overlay = AIOutputs::ObjectDetection(Vec::new());

        tokio::spawn(async move {
            loop {
                let next = {
                    let mut guard = processor.lock().unwrap();
                    guard.as_mut().and_then(|p| p.next())
                };
                let (frame_idx, img) = match next {
                    Some(item) => item,
                    None => break,
                };
                if frame_idx % step != 0 {
                    continue;
                }
                let aio = predictions
                    .get(frame_idx as usize)
                    .and_then(|p| p.clone())
                    .unwrap_or_else(|| empty_overlay.clone());
                let thumb = thumbnail_with_overlay(&img, &aio);
                let jpeg = rgb_image_to_jpeg_buffer(&thumb, 80);
                if tx
                    .send(AnalysisFrame {
                        frame_idx,
                        aioutput: aio,
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
        let (messages, closed) = self.video_state.drain();
        if messages.is_empty() && !closed {
            return;
        }

        let is_analysis = self.video_state.is_processing;

        let mut latest_idx: Option<u64> = None;
        for msg in messages {
            if is_analysis {
                if let Some(pv) = self.current_video_mut() {
                    pv.record(msg.frame_idx, msg.aioutput);
                }
            }
            self.video_thumbnails
                .insert(msg.frame_idx, msg.thumbnail_jpeg);
            latest_idx = Some(msg.frame_idx);
        }
        if let Some(idx) = latest_idx {
            if !self.video_playing {
                // Surface the newest frame when nothing else is driving the
                // texture: during analysis this is the "watch it progress"
                // affordance; during idle preview decode it just keeps the
                // displayed frame near the playhead as thumbnails fill in.
                if is_analysis {
                    self.video_playhead_frame = Some(idx);
                }
                let target = self.video_playhead_frame.unwrap_or(idx);
                self.refresh_video_texture(ui, target);
            }
        }
        if is_analysis {
            if let Some(pv) = self.current_video() {
                self.video_state.progress_bar = pv.frame_progress();
            }
        }
        if closed {
            if is_analysis {
                self.video_state.progress_bar = 1.0;
                if let Some(pv) = self.current_video_mut() {
                    pv.wasprocessed = true;
                }
                self.process_done();
            } else {
                // Preview ran to EOF — drop the exhausted decoder so a later
                // Analyse / replay rebuilds one rather than reusing a dead one.
                *self.video_file_processor.lock().unwrap() = None;
            }
            self.video_state.finish();
        }
        ui.request_repaint();
    }

    // ---------- playback ----------

    pub(super) fn start_video_playback(&mut self, start_frame: u64) {
        let n_frames = match self.current_video() {
            Some(pv) if pv.n_frames > 0 => pv.n_frames,
            _ => return,
        };
        self.video_play_start = Some(Instant::now());
        self.video_play_start_frame = start_frame.min(n_frames.saturating_sub(1));
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
        let pv = self.current_video()?;
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
        let Some(pv) = self.current_video() else { return; };
        let display = pv
            .predictions_file_path()
            .ok()
            .and_then(|p| p.to_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "predictions.json".to_string());
        let pv_clone = pv.clone();
        tokio::spawn(async move {
            let _ = pv_clone.write_predictions();
        });
        self.process_done_at(display);
    }

    pub(super) fn export_video_annotated(&mut self) {
        let (pv_clone, output_path) = match self.current_video() {
            Some(pv) => (pv.clone(), export::prepare_export_video(&pv.file_path)),
            None => return,
        };
        self.video_export_path = output_path.to_str().map(|s| s.to_string());
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
        if self.selected_videos.is_empty() || !self.can_run_image_ai() {
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
            .current_video()
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

    fn draw_video_header(&mut self, ui: &mut egui::Ui, n: usize) {
        let mut new_index = self.video_texture_n;

        ui.horizontal_wrapped(|ui| {
            super::nav_prev_next(
                ui,
                &mut new_index,
                n,
                self.t(Key::prev),
                self.t(Key::next),
            );
            let pv = &self.selected_videos[new_index - 1];
            let name = pv
                .file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(self.t(Key::unknown_file));
            super::nav_filename(ui, name, new_index, n);
            if pv.processed_count() > 0 {
                ui.separator();
                let total = (0..pv.n_frames).step_by(pv.step.max(1) as usize).count();
                ui.label(
                    egui::RichText::new(format!(
                        "{} / {} {}",
                        pv.processed_count(),
                        total.max(1),
                        self.t(Key::frames_analysed),
                    ))
                    .weak()
                    .small(),
                );
            } else {
                ui.separator();
                ui.label(
                    egui::RichText::new(self.t(Key::not_analysed))
                        .weak()
                        .small(),
                );
            }
        });

        super::nav_slider(ui, &mut new_index, n);

        if new_index != self.video_texture_n {
            self.video_texture_n = new_index;
            self.load_current_video(ui);
        }

        ui.add_space(4.0);
    }

    pub(super) fn ui_video(&mut self, ui: &mut egui::Ui) {
        if self.selected_videos.is_empty() {
            return;
        }
        let n = self.selected_videos.len();
        if self.video_texture_n < 1 {
            self.video_texture_n = 1;
        }
        if self.video_texture_n > n {
            self.video_texture_n = n;
        }

        // The user picked a video — start decoding thumbnails so they can play
        // / scrub immediately. Idempotent: no-op if a worker is already running
        // or every step-aligned frame is already cached.
        self.start_video_preview();

        self.draw_video_header(ui, n);

        let Some(n_frames) = self.current_video().map(|p| p.n_frames) else { return; };
        if n_frames == 0 {
            return;
        }
        let fps = self.current_video().map(|p| p.fps).unwrap_or(30.0).max(0.1);
        let last_frame = n_frames.saturating_sub(1);

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

        let avail = ui.available_size_before_wrap();
        let controls_h: f32 = 28.0;
        let seek_h: f32 = 14.0;
        let strip_h: f32 = 6.0;
        let gap: f32 = 6.0;
        let bottom_chrome = controls_h + seek_h + strip_h + gap * 2.0;
        let preview_h = (avail.y - bottom_chrome - 12.0).max(200.0);

        ui.vertical_centered(|ui| {
            if let Some(tex) = self.video_state.texture.clone() {
                // Fit the available area while keeping aspect ratio. `max_height`
                // alone never upscales past the cached thumbnail's native size,
                // so a 1280×720 cache on a 2K screen would look tiny without
                // this explicit scale calc.
                let s = tex.size_vec2();
                let scale =
                    (avail.x / s.x.max(1.0)).min(preview_h / s.y.max(1.0));
                let disp =
                    egui::vec2((s.x * scale).max(1.0), (s.y * scale).max(1.0));
                ui.add(
                    egui::Image::new(&tex)
                        .fit_to_exact_size(disp)
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

        // Lock the player only for destructive ops — analysis touches
        // predictions, export writes a file. Raw preview decode doesn't lock.
        let controls_active = !self.video_state.is_processing
            && self.video_export_receiver.is_none();
        if !controls_active && self.video_playing {
            self.stop_video_playback();
        }

        let playhead = self.video_playhead_frame.unwrap_or(0).min(last_frame);
        let mut new_playhead: Option<u64> = None;

        ui.add_enabled_ui(controls_active, |ui| {
            ui.horizontal(|ui| {
                if self.video_playing {
                    if ui.button("⏸").on_hover_text(self.t(Key::pause)).clicked() {
                        self.stop_video_playback();
                    }
                } else if ui.button("▶").on_hover_text(self.t(Key::play)).clicked() {
                    let start = if playhead >= last_frame { 0 } else { playhead };
                    self.start_video_playback(start);
                }
                if ui
                    .button("⏮")
                    .on_hover_text(self.t(Key::prev))
                    .clicked()
                {
                    if let Some(prev) = prev_analysed_frame(self.current_video(), playhead) {
                        self.stop_video_playback();
                        self.video_playhead_frame = Some(prev);
                        self.refresh_video_texture(ui, prev);
                    }
                }
                if ui
                    .button("⏭")
                    .on_hover_text(self.t(Key::next))
                    .clicked()
                {
                    if let Some(next) = next_analysed_frame(self.current_video(), playhead) {
                        self.stop_video_playback();
                        self.video_playhead_frame = Some(next);
                        self.refresh_video_texture(ui, next);
                    }
                }

                ui.add_space(8.0);
                let suffix: String = if self.video_state.is_processing {
                    format!("  ·  {}", self.t(Key::analysing))
                } else {
                    String::new()
                };
                ui.label(format!(
                    "{}{}",
                    format_time_pair(playhead as f64 / fps, last_frame as f64 / fps),
                    suffix,
                ));
            });

            ui.add_space(gap);

            new_playhead = self.draw_seek_bar(ui, n_frames, seek_h, strip_h, playhead);
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

        let max_idx = n_frames.saturating_sub(1).max(1) as f32;

        // Bar background.
        let bar_bg = if dark {
            egui::Color32::from_gray(55)
        } else {
            egui::Color32::from_gray(210)
        };
        p.rect_filled(bar_rect, 3.0, bar_bg);

        // Played portion.
        let t_played = playhead as f32 / max_idx;
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

        // Predictions strip — dominant-class colour per pixel column, capped
        // at the last analysed frame so unanalysed columns stay plain.
        if let Some(pv) = self.current_video() {
            let strip_bg = if dark {
                egui::Color32::from_gray(45)
            } else {
                egui::Color32::from_gray(225)
            };
            p.rect_filled(strip_rect, 1.0, strip_bg);

            let strip_limit = pv.max_processed_frame().unwrap_or(0);
            let n_cols = strip_rect.width().max(1.0) as usize;
            for (col_start, col_end, class_id) in
                build_strip_segments(pv, strip_limit, n_frames, n_cols)
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

        if ui.is_enabled() {
            if let Some(hover_pos) = response.hover_pos() {
                let t = ((hover_pos.x - bar_rect.left()) / bar_rect.width()).clamp(0.0, 1.0);
                let frame = (t as f64 * n_frames.saturating_sub(1).max(1) as f64) as u64;
                let fps = self.current_video().map(|p| p.fps).unwrap_or(30.0).max(0.1);
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

                let pv = self.current_video();
                let lang = &self.lang;
                response.clone().on_hover_ui_at_pointer(|ui| {
                    tooltip_ui(ui, pv, frame, secs, lang);
                });
            }
        } else {
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
        AIOutputs::PointDetection(points) => {
            let scaled: Vec<XYc> = points
                .iter()
                .map(|p| {
                    let mut np = p.clone();
                    np.xy.x *= sx;
                    np.xy.y *= sy;
                    np
                })
                .collect();
            AIOutputs::PointDetection(scaled)
        }
        AIOutputs::Classification(_)
        | AIOutputs::AudioClassification(_)
        | AIOutputs::Embed(_) => aio.clone(),
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

fn tooltip_ui(
    ui: &mut egui::Ui,
    pv: Option<&PredVideo>,
    frame: u64,
    secs: f64,
    lang: &Lang,
) {
    ui.label(
        egui::RichText::new(format!(
            "{}  ·  {} {}",
            format_time(secs),
            translate(Key::frame_label, lang),
            frame
        ))
        .strong(),
    );
    let Some(pv) = pv else { return; };
    let Some(nearest) = pv.last_processed_at_or_before(frame) else {
        ui.label(egui::RichText::new(translate(Key::not_analysed_parens, lang)).weak());
        return;
    };
    let Some(aio) = pv.frames.get(nearest as usize).and_then(|f| f.as_ref()) else { return; };

    match aio {
        AIOutputs::ObjectDetection(bboxes) => {
            if bboxes.is_empty() {
                ui.label(egui::RichText::new(translate(Key::no_predictions, lang)).weak());
                return;
            }
            ui.separator();
            let n = bboxes.len();
            let noun = if n == 1 { translate(Key::detection, lang) } else { translate(Key::detections, lang) };
            ui.label(egui::RichText::new(format!("{} {}", n, noun)).strong());
            let mut bs: Vec<&XYXYc> = bboxes.iter().collect();
            bs.sort_by(|a, b| {
                b.xyxy.prob.partial_cmp(&a.xyxy.prob).unwrap_or(std::cmp::Ordering::Equal)
            });
            for b in bs.iter().take(6) {
                tooltip_row(ui, b.xyxy.class_id, &b.label, b.xyxy.prob);
            }
            if n > 6 {
                ui.label(egui::RichText::new(and_more(n - 6, lang)).weak());
            }
        }
        AIOutputs::Segmentation(segs) => {
            if segs.is_empty() {
                ui.label(egui::RichText::new(translate(Key::no_predictions, lang)).weak());
                return;
            }
            ui.separator();
            let n = segs.len();
            let noun = if n == 1 { translate(Key::segment, lang) } else { translate(Key::segments, lang) };
            ui.label(egui::RichText::new(format!("{} {}", n, noun)).strong());
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
                ui.label(egui::RichText::new(and_more(n - 6, lang)).weak());
            }
        }
        AIOutputs::Classification(probs) => {
            if probs.is_empty() {
                ui.label(egui::RichText::new(translate(Key::no_predictions, lang)).weak());
                return;
            }
            ui.separator();
            ui.label(egui::RichText::new(translate(Key::classification, lang)).strong());
            let mut ps: Vec<&Prob> = probs.iter().collect();
            ps.sort_by(|a, b| {
                b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal)
            });
            for p in ps.iter().take(6) {
                tooltip_row(ui, p.class_id, &p.label, p.prob);
            }
            if probs.len() > 6 {
                ui.label(
                    egui::RichText::new(and_more(probs.len() - 6, lang)).weak(),
                );
            }
        }
        AIOutputs::PointDetection(points) => {
            if points.is_empty() {
                ui.label(egui::RichText::new(translate(Key::no_predictions, lang)).weak());
                return;
            }
            ui.separator();
            let n = points.len();
            let noun = if n == 1 { translate(Key::detection, lang) } else { translate(Key::detections, lang) };
            ui.label(egui::RichText::new(format!("{} {}", n, noun)).strong());
            let mut pts: Vec<&XYc> = points.iter().collect();
            pts.sort_by(|a, b| {
                b.xy.prob.partial_cmp(&a.xy.prob).unwrap_or(std::cmp::Ordering::Equal)
            });
            for p in pts.iter().take(6) {
                tooltip_row(ui, p.xy.class_id, &p.label, p.xy.prob);
            }
            if n > 6 {
                ui.label(egui::RichText::new(and_more(n - 6, lang)).weak());
            }
        }
        AIOutputs::AudioClassification(_) => {}
        AIOutputs::Embed(_) => {}
    }
}

fn and_more(n: usize, lang: &Lang) -> String {
    translate(Key::and_more_fmt, lang).replace("{}", &n.to_string())
}

fn tooltip_row(ui: &mut egui::Ui, class_id: u32, label: &str, prob: f32) {
    let c = class_color(class_id);
    let color = egui::Color32::from_rgb(c[0], c[1], c[2]);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("■").color(color).monospace());
        ui.label(format!("{} — {:.0}%", label, prob * 100.0));
    });
}
