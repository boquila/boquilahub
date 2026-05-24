use super::{imgbuf_to_texture, Gui, Mode, OpenDialog};
use crate::api::abstractions::*;
use crate::api::bq::process_imgbuf;
use crate::api::render::*;
use crate::api::rest::{detect_remotely, rgb_image_to_jpeg_buffer};
use crate::api::stream;
use crate::localization::*;
use image::{ImageBuffer, Rgb};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

const FEED_THUMB_W: u32 = 720;
const FEED_EXPORT_DIR: &str = "export/feed";

pub const FEED_BUFFER_MIN_SECS: u32 = 5;
pub const FEED_BUFFER_MAX_SECS: u32 = 60;
pub const FEED_BUFFER_DEFAULT_SECS: u32 = 30;

/// One analysed feed frame, posted from the worker task back to the UI thread
/// and stored in the ring buffer for scrubbing / export.
pub(super) struct FeedFrame {
    pub frame_idx: u64,
    pub captured_at: SystemTime,
    pub elapsed: Duration,
    pub aioutput: AIOutputs,
    pub jpeg: Vec<u8>,
}

impl Gui {
    // ---------- url dialog ----------

    pub(super) fn feed_input_dialog(&mut self, ui: &egui::Ui) {
        egui::Window::new(self.t(Key::input_url))
            .collapsible(false)
            .resizable(false)
            .show(ui, |ui| {
                ui.text_edit_singleline(&mut self.temp_str);
                ui.horizontal(|ui| {
                    if ui.button(self.t(Key::ok)).clicked() {
                        let url = self.temp_str.clone();
                        match stream::Feed::new(&url) {
                            Ok(mut feed) => match feed.next() {
                                Some(frame) => {
                                    // Tear down any in-flight worker so its
                                    // frames don't pollute the new session's
                                    // buffer.
                                    if self.feed_state.is_processing {
                                        self.cancel_feed_analysis();
                                    }
                                    let rgba = image::DynamicImage::ImageRgb8(frame).to_rgba8();
                                    self.feed_state.texture = imgbuf_to_texture(&rgba, ui);
                                    self.feed_url = Some(url);
                                    self.reset_feed_session();
                                    self.mode = Mode::Feed;
                                    if self.video_state.is_processing {
                                        self.cancel_video_processing();
                                    }
                                }
                                None => self.process_error(),
                            },
                            Err(_) => self.process_error(),
                        }
                        self.dialog = OpenDialog::None;
                    }
                    ui.add_space(8.0);
                    if ui.button(self.t(Key::cancel)).clicked() {
                        self.dialog = OpenDialog::None;
                        self.feed_url = None;
                    }
                });
            });
    }

    fn reset_feed_session(&mut self) {
        self.feed_buffer.clear();
        self.feed_started_at = None;
        self.feed_playhead_frame = None;
        self.feed_last_displayed_frame = None;
    }

    // ---------- analysis lifecycle ----------

    pub(super) fn start_feed_analysis(&mut self) {
        let Some(url) = self.feed_url.clone() else { return; };

        // Resume picking up where the buffer left off — frame_idx and elapsed
        // stay monotonic so the seek bar / strip don't jump on each ▶.
        let resume_elapsed = self
            .feed_buffer
            .back()
            .map(|f| f.elapsed + Duration::from_millis(1))
            .unwrap_or(Duration::ZERO);
        let resume_frame = self
            .feed_buffer
            .back()
            .map(|f| f.frame_idx)
            .unwrap_or(0);

        let mut feed = match stream::Feed::new(&url) {
            Ok(f) => f,
            Err(_) => {
                self.process_error();
                return;
            }
        };

        let started_at = Instant::now()
            .checked_sub(resume_elapsed)
            .unwrap_or_else(Instant::now);
        self.feed_started_at = Some(started_at);
        self.feed_state.is_processing = true;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<FeedFrame>();
        self.feed_processing_receiver = Some(rx);
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        self.feed_state.cancel_sender = Some(cancel_tx);

        let api_endpoint = self.get_endpoint();
        let is_remote = !self.ep_selected.is_local();
        let step = self.feed_step_frame.max(1);

        tokio::spawn(async move {
            let mut frame_counter: u64 = resume_frame;
            loop {
                if cancel_rx.try_recv().is_ok() {
                    break;
                }
                let Some(mut img) = feed.next() else { break; };
                frame_counter += 1;
                if (frame_counter as usize) % step != 0 {
                    continue;
                }

                let aioutput: AIOutputs = if is_remote {
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
                let elapsed = started_at.elapsed();
                let captured_at = SystemTime::now();
                let frame = FeedFrame {
                    frame_idx: frame_counter,
                    captured_at,
                    elapsed,
                    aioutput,
                    jpeg,
                };
                if tx.send(frame).is_err() {
                    break;
                }
            }
        });
    }

    pub(super) fn cancel_feed_analysis(&mut self) {
        self.feed_state.cancel();
        self.feed_processing_receiver = None;
    }

    pub(super) fn feed_handle_results(&mut self, ui: &egui::Ui) {
        let mut incoming: Vec<FeedFrame> = Vec::new();
        let mut channel_closed = false;
        if let Some(rx) = &mut self.feed_processing_receiver {
            loop {
                match rx.try_recv() {
                    Ok(msg) => incoming.push(msg),
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        channel_closed = true;
                        break;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                }
            }
        }

        if !incoming.is_empty() {
            let save_each = self.save_img_from_feed;
            let mut latest_idx: Option<u64> = None;
            for msg in incoming {
                if save_each && !msg.aioutput.is_empty() {
                    save_frame_jpeg(&msg.jpeg, msg.captured_at);
                }
                latest_idx = Some(msg.frame_idx);
                self.feed_buffer.push_back(msg);
            }
            self.evict_old_feed_frames();
            // Don't strand the playhead on an evicted frame — snap to the start
            // of the remaining buffer and re-paint immediately so the viewer
            // doesn't show a frame that's no longer in the cache.
            let mut snapped = false;
            if let (Some(ph), Some(oldest)) = (
                self.feed_playhead_frame,
                self.feed_buffer.front().map(|f| f.frame_idx),
            ) {
                if ph < oldest {
                    self.feed_playhead_frame = Some(oldest);
                    self.feed_last_displayed_frame = None;
                    snapped = true;
                }
            }
            // Live mode (playhead None): track the latest frame.
            if self.feed_playhead_frame.is_none() {
                if let Some(idx) = latest_idx {
                    self.refresh_feed_texture(ui, idx);
                }
            } else if snapped {
                if let Some(idx) = self.feed_playhead_frame {
                    self.refresh_feed_texture(ui, idx);
                }
            }
        }

        if channel_closed {
            self.feed_state.is_processing = false;
            self.feed_processing_receiver = None;
        }

        if self.feed_state.is_processing {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(50));
        }
    }

    fn evict_old_feed_frames(&mut self) {
        let max_secs = self
            .feed_buffer_max_secs
            .clamp(FEED_BUFFER_MIN_SECS, FEED_BUFFER_MAX_SECS)
            as f64;
        loop {
            let (front_elapsed, back_elapsed) = match (
                self.feed_buffer.front().map(|f| f.elapsed),
                self.feed_buffer.back().map(|f| f.elapsed),
            ) {
                (Some(o), Some(n)) => (o, n),
                _ => return,
            };
            if back_elapsed.saturating_sub(front_elapsed).as_secs_f64() <= max_secs {
                return;
            }
            self.feed_buffer.pop_front();
        }
    }

    // ---------- texture / lookup helpers ----------

    fn nearest_feed_frame_idx(&self, target: u64) -> Option<u64> {
        let mut best: Option<(u64, u64)> = None;
        for f in &self.feed_buffer {
            let d = if f.frame_idx > target {
                f.frame_idx - target
            } else {
                target - f.frame_idx
            };
            if best.map_or(true, |(_, bd)| d < bd) {
                best = Some((f.frame_idx, d));
            }
        }
        best.map(|(i, _)| i)
    }

    fn feed_frame_by_idx(&self, idx: u64) -> Option<&FeedFrame> {
        self.feed_buffer.iter().find(|f| f.frame_idx == idx)
    }

    fn refresh_feed_texture(&mut self, ui: &egui::Ui, target_frame: u64) {
        let Some(src_idx) = self.nearest_feed_frame_idx(target_frame) else { return; };
        if self.feed_last_displayed_frame == Some(src_idx) {
            return;
        }
        let Some(jpeg) = self.feed_frame_by_idx(src_idx).map(|f| f.jpeg.clone()) else { return; };
        let Ok(dynimg) = image::load_from_memory(&jpeg) else { return; };
        let rgba = dynimg.to_rgba8();
        self.feed_state.texture = imgbuf_to_texture(&rgba, ui);
        self.feed_last_displayed_frame = Some(src_idx);
    }

    // ---------- left-panel widget ----------

    pub(super) fn feed_analysis_widget(&mut self, ui: &mut egui::Ui) {
        if self.feed_url.is_none() || !(self.is_image_model() || !self.ep_selected.is_local()) {
            return;
        }

        ui.vertical_centered(|ui| {
            ui.heading(self.t(Key::camera_feed));
            ui.heading(self.t(Key::analysis));
        });
        ui.separator();

        ui.vertical_centered(|ui| {
            if ui.button("⚙").clicked() {
                self.show_config.feed = !self.show_config.feed;
            }
            if self.show_config.feed {
                ui.add_enabled_ui(!self.feed_state.is_processing, |ui| {
                    ui.label(self.t(Key::freq));
                    ui.style_mut().spacing.slider_width = 120.0;
                    ui.add(egui::Slider::new(&mut self.feed_step_frame, 1..=90));
                });
                ui.add_space(6.0);
                ui.label("Cache (s)");
                ui.style_mut().spacing.slider_width = 120.0;
                ui.add(egui::Slider::new(
                    &mut self.feed_buffer_max_secs,
                    FEED_BUFFER_MIN_SECS..=FEED_BUFFER_MAX_SECS,
                ));
                self.evict_old_feed_frames();
                ui.add_space(6.0);
                ui.label(self.t(Key::export_obs));
                ui.checkbox(&mut self.save_img_from_feed, "");
                ui.add_space(8.0);
            }

            if !self.feed_state.is_processing {
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new("▶"))
                    .clicked()
                {
                    self.start_feed_analysis();
                }
            } else if ui
                .add_sized([85.0, 40.0], egui::Button::new("⏸"))
                .clicked()
            {
                self.cancel_feed_analysis();
            }
        });

        ui.add_space(8.0);

        if !self.feed_buffer.is_empty() {
            ui.vertical_centered(|ui| {
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::export)))
                    .clicked()
                {
                    self.dialog = OpenDialog::Export;
                }
            });
        }

        self.feed_export_dialog(ui);
    }

    fn feed_export_dialog(&mut self, ui: &egui::Ui) {
        if self.dialog != OpenDialog::Export || self.mode != Mode::Feed {
            return;
        }
        let mut close = false;
        let mut export_json = false;
        let mut export_png = false;
        egui::Window::new(self.t(Key::export))
            .collapsible(false)
            .resizable(false)
            .show(ui, |ui| {
                if ui.button("Export current frame (.json)").clicked() {
                    export_json = true;
                    close = true;
                }
                if ui.button("Export current frame (.png)").clicked() {
                    export_png = true;
                    close = true;
                }
                if ui.button(self.t(Key::cancel)).clicked() {
                    close = true;
                }
            });
        if close {
            self.dialog = OpenDialog::None;
        }
        if export_json {
            match self.export_current_feed_frame_json() {
                Some(path) => self.process_done_at(path),
                None => self.process_error(),
            }
        }
        if export_png {
            match self.export_current_feed_frame_png() {
                Some(path) => self.process_done_at(path),
                None => self.process_error(),
            }
        }
    }

    fn current_feed_frame(&self) -> Option<&FeedFrame> {
        let target = self
            .feed_playhead_frame
            .or_else(|| self.feed_buffer.back().map(|f| f.frame_idx))?;
        let idx = self.nearest_feed_frame_idx(target)?;
        self.feed_frame_by_idx(idx)
    }

    fn export_current_feed_frame_json(&self) -> Option<String> {
        let frame = self.current_feed_frame()?;
        std::fs::create_dir_all(FEED_EXPORT_DIR).ok()?;
        let stamp = local_stamp(frame.captured_at);
        let path = format!(
            "{}/feed_{}_frame_{}.json",
            FEED_EXPORT_DIR, stamp, frame.frame_idx
        );
        let body = serde_json::to_string_pretty(&frame.aioutput).ok()?;
        std::fs::write(&path, body).ok()?;
        Some(path)
    }

    fn export_current_feed_frame_png(&self) -> Option<String> {
        let frame = self.current_feed_frame()?;
        let dynimg = image::load_from_memory(&frame.jpeg).ok()?;
        std::fs::create_dir_all(FEED_EXPORT_DIR).ok()?;
        let stamp = local_stamp(frame.captured_at);
        let path = format!(
            "{}/feed_{}_frame_{}.png",
            FEED_EXPORT_DIR, stamp, frame.frame_idx
        );
        dynimg.save(&path).ok()?;
        Some(path)
    }

    // ---------- main feed viewer ----------

    pub(super) fn ui_feed(&mut self, ui: &mut egui::Ui) {
        self.draw_feed_header(ui);

        let avail = ui.available_size_before_wrap();
        let has_buffer = !self.feed_buffer.is_empty();
        let controls_h: f32 = 28.0;
        let seek_h: f32 = 14.0;
        let strip_h: f32 = 6.0;
        let gap: f32 = 6.0;
        let bottom_chrome = if has_buffer {
            controls_h + seek_h + strip_h + gap * 2.0
        } else {
            24.0
        };
        let preview_h = (avail.y - bottom_chrome - 12.0).max(200.0);

        ui.vertical_centered(|ui| {
            if let Some(tex) = self.feed_state.texture.clone() {
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

        if !has_buffer {
            ui.vertical_centered(|ui| {
                let hint = if self.feed_state.is_processing {
                    "Waiting for the first analysed frame…"
                } else {
                    "Press ▶ to start analysing this feed"
                };
                ui.label(egui::RichText::new(hint).color(ui.visuals().weak_text_color()));
            });
            self.feed_handle_results(ui);
            return;
        }

        let oldest_idx = self.feed_buffer.front().unwrap().frame_idx;
        let newest_idx = self.feed_buffer.back().unwrap().frame_idx;
        let oldest_elapsed = self.feed_buffer.front().unwrap().elapsed;
        let newest_elapsed = self.feed_buffer.back().unwrap().elapsed;
        let following_live = self.feed_playhead_frame.is_none();
        let raw_playhead = self.feed_playhead_frame.unwrap_or(newest_idx);
        let playhead = self
            .nearest_feed_frame_idx(raw_playhead)
            .unwrap_or(newest_idx);
        let playhead_frame = self.feed_frame_by_idx(playhead);
        let playhead_elapsed = playhead_frame
            .map(|f| f.elapsed)
            .unwrap_or(newest_elapsed);
        let playhead_label = playhead_frame.and_then(|f| top_label(&f.aioutput));

        // None outer → no change; Some(None) → snap to live; Some(Some(idx)) → scrub
        let mut requested: Option<Option<u64>> = None;

        ui.horizontal(|ui| {
            let follow_text = if following_live { "● LIVE" } else { "○ Live" };
            if ui
                .button(follow_text)
                .on_hover_text("Snap to and follow the newest frame")
                .clicked()
            {
                requested = Some(None);
            }
            ui.separator();
            if ui
                .button("⏮")
                .on_hover_text("Previous cached frame")
                .clicked()
            {
                if let Some(prev) = prev_feed_frame(&self.feed_buffer, playhead) {
                    requested = Some(Some(prev));
                }
            }
            if ui
                .button("⏭")
                .on_hover_text("Next cached frame")
                .clicked()
            {
                if let Some(next) = next_feed_frame(&self.feed_buffer, playhead) {
                    requested = Some(Some(next));
                }
            }
            ui.separator();
            ui.label(format!(
                "{}  ·  buffer {:.1}s / {}s",
                format_time(playhead_elapsed.as_secs_f64()),
                (newest_elapsed.saturating_sub(oldest_elapsed)).as_secs_f64(),
                self.feed_buffer_max_secs,
            ));
            if let Some(label) = playhead_label {
                ui.separator();
                ui.label(egui::RichText::new(label).strong());
            }
        });

        ui.add_space(gap);

        if let Some(idx) =
            self.draw_feed_seek_bar(ui, oldest_idx, newest_idx, seek_h, strip_h, playhead, following_live)
        {
            requested = Some(Some(idx));
        }

        if let Some(change) = requested {
            match change {
                None => {
                    self.feed_playhead_frame = None;
                    self.refresh_feed_texture(ui, newest_idx);
                }
                Some(idx) => {
                    self.feed_playhead_frame = Some(idx);
                    self.refresh_feed_texture(ui, idx);
                }
            }
        }

        self.feed_handle_results(ui);
    }

    fn draw_feed_header(&self, ui: &mut egui::Ui) {
        let Some(url) = self.feed_url.as_ref() else { return; };
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(url).strong());
            ui.separator();
            let status = if self.feed_state.is_processing {
                egui::RichText::new("● analysing").color(egui::Color32::from_rgb(110, 200, 120))
            } else if !self.feed_buffer.is_empty() {
                egui::RichText::new("paused").weak()
            } else {
                egui::RichText::new("idle").weak()
            };
            ui.label(status.small());
            if !self.feed_buffer.is_empty() {
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("{} cached", self.feed_buffer.len()))
                        .weak()
                        .small(),
                );
            }
        });
        ui.add_space(4.0);
    }

    fn draw_feed_seek_bar(
        &self,
        ui: &mut egui::Ui,
        oldest_idx: u64,
        newest_idx: u64,
        seek_h: f32,
        strip_h: f32,
        playhead: u64,
        following_live: bool,
    ) -> Option<u64> {
        let avail_w = ui.available_width();
        let total_h = seek_h + strip_h + 2.0;
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(avail_w, total_h), egui::Sense::click_and_drag());
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

        let bar_bg = if dark {
            egui::Color32::from_gray(55)
        } else {
            egui::Color32::from_gray(210)
        };
        p.rect_filled(bar_rect, 3.0, bar_bg);

        // Played portion = [oldest, playhead]. Green tint while following live so
        // the user has a clear "this is following" cue without reading text.
        let span = newest_idx.saturating_sub(oldest_idx).max(1) as f32;
        let t_play =
            ((playhead.saturating_sub(oldest_idx)) as f32 / span).clamp(0.0, 1.0);
        let played_x = bar_rect.left() + bar_rect.width() * t_play;
        let played_rect = egui::Rect::from_min_max(
            bar_rect.left_top(),
            egui::pos2(played_x, bar_rect.bottom()),
        );
        let played_color = if following_live {
            egui::Color32::from_rgb(110, 200, 120)
        } else if dark {
            egui::Color32::from_rgb(110, 180, 255)
        } else {
            egui::Color32::from_rgb(70, 120, 220)
        };
        p.rect_filled(played_rect, 3.0, played_color);

        let strip_bg = if dark {
            egui::Color32::from_gray(45)
        } else {
            egui::Color32::from_gray(225)
        };
        p.rect_filled(strip_rect, 1.0, strip_bg);

        let n_cols = strip_rect.width().max(1.0) as usize;
        for (col_start, col_end, class_id) in
            build_feed_strip_segments(&self.feed_buffer, oldest_idx, newest_idx, n_cols)
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

        if let Some(hover_pos) = response.hover_pos() {
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
            let t = ((hover_pos.x - bar_rect.left()) / bar_rect.width()).clamp(0.0, 1.0);
            let target = oldest_idx + (t * span) as u64;
            let nearest = self.nearest_feed_frame_idx(target);
            response.clone().on_hover_ui_at_pointer(|ui| {
                feed_tooltip_ui(ui, &self.feed_buffer, nearest);
            });
        }

        if response.clicked() || response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let t = ((pos.x - bar_rect.left()) / bar_rect.width()).clamp(0.0, 1.0);
                let target = oldest_idx + (t * span) as u64;
                if let Some(nearest) = self.nearest_feed_frame_idx(target) {
                    return Some(nearest);
                }
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
    let mut small = if img.width() <= FEED_THUMB_W {
        img.clone()
    } else {
        let h = (img.height() as f32 * FEED_THUMB_W as f32 / img.width() as f32) as u32;
        image::imageops::resize(
            img,
            FEED_THUMB_W,
            h.max(1),
            image::imageops::FilterType::Triangle,
        )
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

fn save_frame_jpeg(jpeg: &[u8], captured_at: SystemTime) {
    let _ = std::fs::create_dir_all(FEED_EXPORT_DIR);
    let stamp = local_stamp(captured_at);
    let path = format!("{}/feed_{}.jpg", FEED_EXPORT_DIR, stamp);
    let _ = std::fs::write(path, jpeg);
}

fn local_stamp(t: SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Local> = t.into();
    dt.format("%Y%m%d_%H%M%S_%3f").to_string()
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

fn prev_feed_frame(buf: &VecDeque<FeedFrame>, current: u64) -> Option<u64> {
    buf.iter()
        .rev()
        .find(|f| f.frame_idx < current)
        .map(|f| f.frame_idx)
}

fn next_feed_frame(buf: &VecDeque<FeedFrame>, current: u64) -> Option<u64> {
    buf.iter()
        .find(|f| f.frame_idx > current)
        .map(|f| f.frame_idx)
}

fn top_label(aio: &AIOutputs) -> Option<String> {
    let (_, label, prob) = aio.dominant_prob()?;
    Some(format!("{}  ·  {:.0}%", label, prob * 100.0))
}

/// Walk pixel columns, find the buffer entry closest to each column's mapped
/// frame_idx, and merge consecutive same-class columns into rectangles. Mirrors
/// `video_file::build_strip_segments` — cheap because the buffer is sorted by
/// frame_idx, so a single forward walker handles all lookups.
fn build_feed_strip_segments(
    buf: &VecDeque<FeedFrame>,
    oldest_idx: u64,
    newest_idx: u64,
    n_cols: usize,
) -> Vec<(usize, usize, u32)> {
    if buf.is_empty() || n_cols == 0 || newest_idx == oldest_idx {
        return Vec::new();
    }
    let span = (newest_idx - oldest_idx) as f64;
    let mut segments: Vec<(usize, usize, u32)> = Vec::new();
    let mut current: Option<(usize, u32)> = None;
    let mut last_walker: Option<usize> = None;
    let mut last_class: Option<u32> = None;
    let mut walker = 0usize;

    for col in 0..n_cols {
        let t = (col as f64 + 0.5) / n_cols as f64;
        let target = oldest_idx as f64 + t * span;
        while walker + 1 < buf.len() {
            let cur = (buf[walker].frame_idx as f64 - target).abs();
            let next = (buf[walker + 1].frame_idx as f64 - target).abs();
            if next < cur {
                walker += 1;
            } else {
                break;
            }
        }
        let class = if last_walker == Some(walker) {
            last_class
        } else {
            last_walker = Some(walker);
            let c = buf[walker]
                .aioutput
                .dominant_prob()
                .map(|(cid, _, _)| cid);
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
        segments.push((start, n_cols.saturating_sub(1), cur));
    }
    segments
}

fn feed_tooltip_ui(ui: &mut egui::Ui, buf: &VecDeque<FeedFrame>, target: Option<u64>) {
    let Some(target_idx) = target else { return; };
    let Some(frame) = buf.iter().find(|f| f.frame_idx == target_idx) else { return; };
    ui.label(
        egui::RichText::new(format!(
            "{}  ·  frame {}",
            format_time(frame.elapsed.as_secs_f64()),
            frame.frame_idx
        ))
        .strong(),
    );
    match &frame.aioutput {
        AIOutputs::ObjectDetection(bs) => {
            if bs.is_empty() {
                ui.label(egui::RichText::new("(no detections)").weak());
                return;
            }
            ui.separator();
            let n = bs.len();
            let header = if n == 1 { "1 detection".into() } else { format!("{} detections", n) };
            ui.label(egui::RichText::new(header).strong());
            let mut sorted: Vec<&XYXYc> = bs.iter().collect();
            sorted.sort_by(|a, b| {
                b.xyxy
                    .prob
                    .partial_cmp(&a.xyxy.prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for b in sorted.iter().take(6) {
                tooltip_row(ui, b.xyxy.class_id, &b.label, b.xyxy.prob);
            }
            if n > 6 {
                ui.label(egui::RichText::new(format!("…and {} more", n - 6)).weak());
            }
        }
        AIOutputs::Segmentation(ss) => {
            if ss.is_empty() {
                ui.label(egui::RichText::new("(no detections)").weak());
                return;
            }
            ui.separator();
            let n = ss.len();
            let header = if n == 1 { "1 segment".into() } else { format!("{} segments", n) };
            ui.label(egui::RichText::new(header).strong());
            let mut sorted: Vec<&SEGc> = ss.iter().collect();
            sorted.sort_by(|a, b| {
                b.bbox
                    .xyxy
                    .prob
                    .partial_cmp(&a.bbox.xyxy.prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for s in sorted.iter().take(6) {
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
            let mut sorted: Vec<&Prob> = probs.iter().collect();
            sorted.sort_by(|a, b| {
                b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal)
            });
            for p in sorted.iter().take(6) {
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
