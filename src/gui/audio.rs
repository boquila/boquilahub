use super::{imgbuf_to_texture, Gui, Mode, OpenDialog};
use crate::api::abstractions::*;
use crate::api::audio::AudioData;
use crate::api::bq::{process_audio, Modality};
use crate::api::processing::pre::compute_mel;
use crate::api::render::*;
use crate::localization::*;
use image::{ImageBuffer, Rgba};
use rodio::Source;
use std::time::Instant;

pub(super) const AUDIO_DISPLAY_SR: u32 = 22050;

impl Gui {
    pub(super) fn is_audio_model(&self) -> bool {
        self.ai_selected
            .map(|i| self.ais[i].modality == Modality::Audio)
            .unwrap_or(false)
    }

    pub(super) fn audio_mel_params(&self) -> (usize, usize, usize, f32) {
        const N_MELS: usize = 128;
        const FALLBACK_N_FFT: usize = 2048;
        const FALLBACK_HOP_LENGTH: usize = 512;
        const FALLBACK_TOP_DB: f32 = 80.0;

        if let Some(ai) = self.ai_selected {
            if let Some(ref ac) = self.ais[ai].audio_config {
                return (
                    ac.n_fft as usize,
                    ac.hop_length as usize,
                    N_MELS,
                    ac.top_db,
                );
            }
        }
        (FALLBACK_N_FFT, FALLBACK_HOP_LENGTH, N_MELS, FALLBACK_TOP_DB)
    }

    pub(super) fn audio_analysis_widget(&mut self, ui: &mut egui::Ui) {
        if self.audio_data.is_some() && (self.is_audio_model() || !self.ep_selected.is_local()) {
            ui.vertical_centered(|ui| {
                ui.heading(self.t(Key::audio_file));
                ui.heading(self.t(Key::analysis));
            });
            ui.separator();

            ui.vertical_centered(|ui| {
                if !self.audio_state.is_processing {
                    if ui
                        .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::analyze)))
                        .clicked()
                    {
                        self.start_audio_analysis();
                    }
                } else {
                    if ui
                        .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::cancel)))
                        .clicked()
                    {
                        self.audio_state.cancel();
                    }
                }
            });

            ui.add_space(8.0);

            let has_predictions = self
                .pred_audio
                .as_ref()
                .map(|p| p.wasprocessed)
                .unwrap_or(false);
            if has_predictions && !self.audio_state.is_processing {
                ui.vertical_centered(|ui| {
                    if ui
                        .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::export)))
                        .clicked()
                    {
                        self.dialog = OpenDialog::Export;
                    }
                });
            }

            self.audio_export_dialog(ui);
        }
    }

    fn audio_export_dialog(&mut self, ui: &egui::Ui) {
        if self.dialog != OpenDialog::Export || self.mode != Mode::Audio {
            return;
        }
        let mut close = false;
        let mut export = false;
        egui::Window::new(self.t(Key::export))
            .collapsible(false)
            .resizable(false)
            .show(ui, |ui| {
                if ui.button(self.t(Key::export_predictions)).clicked() {
                    export = true;
                    close = true;
                }
                if ui.button(self.t(Key::cancel)).clicked() {
                    close = true;
                }
            });
        if export {
            if let Some(pred) = self.pred_audio.clone() {
                let display = pred
                    .predictions_file_path()
                    .ok()
                    .and_then(|p| p.to_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| "predictions.json".to_string());
                tokio::spawn(async move {
                    let _ = pred.write_pred_audio_to_file().await;
                });
                self.process_done_at(display);
            }
        }
        if close {
            self.dialog = OpenDialog::None;
        }
    }

    pub(super) fn start_audio_analysis(&mut self) {
        self.audio_state.is_processing = true;
        let (tx, rx) = std::sync::mpsc::channel();
        self.audio_result_receiver = Some(rx);

        let audio = self.audio_data.as_ref().unwrap().clone();
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        self.audio_state.cancel_sender = Some(cancel_tx);

        tokio::spawn(async move {
            if cancel_rx.try_recv().is_ok() {
                return;
            }
            let result = tokio::task::spawn_blocking(move || process_audio(&audio)).await;
            let _ = tx.send(result);
        });
    }

    pub(super) fn start_playback_from_data(&mut self, start_pos: f64) {
        let Some(audio) = self.audio_data.as_ref() else { return; };
        let src = AudioBufferSource::new_from(audio, start_pos);
        if let Ok(mut stream) = rodio::DeviceSinkBuilder::open_default_sink() {
            stream.log_on_drop(false);
            let player = rodio::Player::connect_new(stream.mixer());
            player.append(src);
            self.audio_stream = Some(stream);
            self.audio_player = Some(player);
            self.audio_playing = true;
            self.audio_play_start = Some(Instant::now());
            self.audio_play_start_pos = start_pos;
            self.audio_playhead = Some(start_pos);
        }
    }

    pub(super) fn stop_playback(&mut self) {
        if let Some(player) = &self.audio_player {
            player.stop();
        }
        self.audio_player = None;
        self.audio_stream = None;
        self.audio_playing = false;
        self.audio_play_start = None;
    }

    fn draw_audio_header(&self, ui: &mut egui::Ui) {
        let Some(pred) = self.pred_audio.as_ref() else { return; };
        let name = pred
            .file_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("(unknown)");
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(name).strong());
            if !pred.wasprocessed {
                ui.separator();
                ui.label(
                    egui::RichText::new("not analysed")
                        .weak()
                        .small(),
                );
            } else if let Some(preds) = pred.audio_predictions() {
                ui.separator();
                let n = preds.len();
                let noun = if n == 1 { "segment" } else { "segments" };
                ui.label(
                    egui::RichText::new(format!("{} {}", n, noun)).strong(),
                );
            }
        });
        ui.add_space(4.0);
    }

    pub(super) fn audio_handle_results(&mut self) {
        if let Some(rx) = &self.audio_result_receiver {
            if let Ok(result) = rx.try_recv() {
                match result {
                    Ok(aio @ AIOutputs::AudioClassification(_)) => {
                        if let Some(pred) = self.pred_audio.as_mut() {
                            pred.aioutput = Some(aio);
                            pred.wasprocessed = true;
                        }
                    }
                    _ => {
                        self.process_error();
                    }
                }
                self.audio_state.is_processing = false;
                self.audio_state.texture = None;
                self.audio_result_receiver = None;
                self.process_done();
            }
        }
    }

    pub(super) fn ui_audio(&mut self, ui: &mut egui::Ui) {
        if self.audio_data.is_some() {
            self.draw_audio_header(ui);

            let duration =
                self.audio_data.as_ref().unwrap().duration().max(0.1);

            // Precompute the mel for the entire audio on first use.
            if self.audio_full_mel.is_none() {
                let (n_fft, hop_length, n_mels, top_db) = self.audio_mel_params();
                let resampled =
                    self.audio_data.as_ref().unwrap().resample(AUDIO_DISPLAY_SR);
                let mel =
                    compute_mel(&resampled, n_fft, hop_length, n_mels, top_db);
                self.audio_full_mel = Some(mel);
                self.audio_mel_meta = Some((n_fft, hop_length, n_mels, top_db));
                self.audio_state.texture = None;
            }

            // Clamp view range to audio bounds; require a minimum span.
            const MIN_VIEW_SECS: f64 = 0.1;
            let (vs, ve) = self.audio_view_range;
            let vs = vs.max(0.0).min((duration - MIN_VIEW_SECS).max(0.0));
            let ve = ve.max(vs + MIN_VIEW_SECS).min(duration);
            if (vs - self.audio_view_range.0).abs() > 1e-6
                || (ve - self.audio_view_range.1).abs() > 1e-6
            {
                self.audio_view_range = (vs, ve);
                self.audio_view_range_dirty = true;
                self.audio_state.texture = None;
            }

            // Drive the playhead off the playback clock.
            if self.audio_playing {
                if let Some(start) = self.audio_play_start {
                    let ph = (self.audio_play_start_pos
                        + start.elapsed().as_secs_f64())
                    .min(duration);
                    if ph >= duration {
                        self.stop_playback();
                    }
                    self.audio_playhead = Some(ph);

                    // Auto-follow: when playhead leaves the visible range
                    // to the right, shift the view forward by one span.
                    let (vs, ve) = self.audio_view_range;
                    if ph > ve {
                        let span = ve - vs;
                        let new_vs = (ph - span * 0.05).max(0.0);
                        let new_ve = (new_vs + span).min(duration);
                        self.audio_view_range = (new_vs, new_ve);
                        self.audio_view_range_dirty = true;
                        self.audio_state.texture = None;
                    }
                }
                ui.ctx().request_repaint();
            }

            ui.horizontal(|ui| {
                if self.audio_playing {
                    if ui.button("⏸").clicked() {
                        self.stop_playback();
                        // playhead stays where it was — visible marker.
                    }
                } else if ui.button("▶").clicked() {
                    let start_pos = self
                        .audio_playhead
                        .unwrap_or(self.audio_view_range.0)
                        .clamp(0.0, duration);
                    self.start_playback_from_data(start_pos);
                }
                if ui
                    .button("⏮")
                    .on_hover_text("Reset playhead to start")
                    .clicked()
                {
                    self.stop_playback();
                    self.audio_playhead = Some(0.0);
                }
                ui.separator();
                if ui
                    .button("Fit")
                    .on_hover_text("Fit view to whole audio")
                    .clicked()
                {
                    self.audio_view_range = (0.0, duration);
                    self.audio_view_range_dirty = true;
                    self.audio_state.texture = None;
                }
                ui.label(format!(
                    "view {:.2}s → {:.2}s   ·   span {:.2}s",
                    self.audio_view_range.0,
                    self.audio_view_range.1,
                    self.audio_view_range.1 - self.audio_view_range.0,
                ));
            });

            // Plot fills the remaining space. Texture is sized to
            // match the rendered plot so it stays sharp at any size.
            let avail = ui.available_size_before_wrap();
            let plot_w = avail.x.max(600.0);
            let plot_h = (avail.y - 4.0).max(320.0);
            let target_tex_w = (plot_w as usize).clamp(400, 4096);
            // ~72% of the plot height is the spectrogram itself
            // (the rest is the label strip + time axis).
            let target_tex_h =
                ((plot_h * 0.72) as usize).clamp(200, 1600);

            if self.audio_tex_dims != Some((target_tex_w, target_tex_h)) {
                self.audio_state.texture = None;
            }

            let window_preds: Vec<AudioProb> = self
                .pred_audio
                .as_ref()
                .and_then(|p| p.audio_predictions())
                .map(|preds| {
                    preds
                        .iter()
                        .filter(|p| {
                            p.end as f64 > self.audio_view_range.0
                                && (p.start as f64) < self.audio_view_range.1
                        })
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();

            let column_winner = dominant_per_column(
                &window_preds,
                self.audio_view_range.0,
                self.audio_view_range.1 - self.audio_view_range.0,
                target_tex_w,
            );

            if self.audio_state.texture.is_none() {
                if let (Some(full_mel), Some(meta)) =
                    (self.audio_full_mel.as_ref(), self.audio_mel_meta)
                {
                    let (_n_fft, hop_length, _n_mels, top_db) = meta;
                    let img = mel_slice_to_imgbuf(
                        full_mel,
                        AUDIO_DISPLAY_SR,
                        hop_length,
                        top_db,
                        self.audio_view_range.0,
                        self.audio_view_range.1,
                        target_tex_w,
                        target_tex_h,
                        &window_preds,
                        &column_winner,
                    );
                    self.audio_state.texture = imgbuf_to_texture(&img, ui);
                    self.audio_tex_dims =
                        Some((target_tex_w, target_tex_h));
                }
            }

            if let Some(texture) = self.audio_state.texture.clone() {
                let apply_bounds =
                    std::mem::take(&mut self.audio_view_range_dirty);
                let result = {
                    let full_mel =
                        self.audio_full_mel.as_ref().expect("mel precomputed");
                    let meta = self.audio_mel_meta.expect("mel meta");
                    render_audio_plot(
                        ui,
                        &texture,
                        full_mel,
                        meta,
                        self.audio_view_range,
                        duration,
                        &window_preds,
                        &column_winner,
                        self.audio_playhead,
                        apply_bounds,
                        plot_w,
                        plot_h,
                        target_tex_w,
                    )
                };

                if let Some(new_range) = result.new_view_range {
                    self.audio_view_range = new_range;
                    self.audio_state.texture = None;
                }

                if let Some(t) = result.clicked_time {
                    let was_playing = self.audio_playing;
                    self.stop_playback();
                    self.audio_playhead = Some(t);
                    if was_playing {
                        self.start_playback_from_data(t);
                    }
                }
            }
        }

        self.audio_handle_results();
    }
}

/// For each pixel column of the spectrogram, the index (into `preds`) of the
/// dominant prediction (highest prob) at that column's center time, or `None`
/// if no prediction covers it. This is the single source of truth used by both
/// the spectrogram coloring and the class-label strip — guaranteeing they stay
/// pixel-perfectly aligned even when predictions overlap.
fn dominant_per_column(
    preds: &[AudioProb],
    window_start: f64,
    window_duration: f64,
    n_cols: usize,
) -> Vec<Option<usize>> {
    let mut out = vec![None; n_cols];
    if preds.is_empty() {
        return out;
    }
    for col in 0..n_cols {
        let time = window_start + ((col as f64 + 0.5) / n_cols as f64) * window_duration;
        let mut best_idx: Option<usize> = None;
        let mut best_prob = f32::MIN;
        for (i, p) in preds.iter().enumerate() {
            if time >= p.start as f64 && time < p.end as f64 && p.prediction.prob > best_prob {
                best_prob = p.prediction.prob;
                best_idx = Some(i);
            }
        }
        out[col] = best_idx;
    }
    out
}

struct LabelSegment {
    col_start: usize,
    col_end: usize, // inclusive
    class_id: u32,
    label: String,
    max_prob: f32,
}

fn segments_from_columns(preds: &[AudioProb], columns: &[Option<usize>]) -> Vec<LabelSegment> {
    let mut segs: Vec<LabelSegment> = Vec::new();
    for (col, winner) in columns.iter().enumerate() {
        let Some(idx) = winner else { continue; };
        let p = &preds[*idx];
        let cid = p.prediction.class_id;
        let prob = p.prediction.prob;
        match segs.last_mut() {
            Some(last)
                if last.class_id == cid
                    && last.label == p.prediction.label
                    && last.col_end + 1 == col =>
            {
                last.col_end = col;
                if prob > last.max_prob {
                    last.max_prob = prob;
                }
            }
            _ => segs.push(LabelSegment {
                col_start: col,
                col_end: col,
                class_id: cid,
                label: p.prediction.label.clone(),
                max_prob: prob,
            }),
        }
    }
    segs
}

struct PlotInteraction {
    /// New view range if the user panned/zoomed (or None if unchanged).
    new_view_range: Option<(f64, f64)>,
    /// Time the user clicked at (no drag), if any.
    clicked_time: Option<f64>,
}

fn render_audio_plot(
    ui: &mut egui::Ui,
    texture: &egui::TextureHandle,
    full_mel: &ndarray::Array2<f32>,
    mel_meta: (usize, usize, usize, f32), // n_fft, hop_length, n_mels, top_db
    view_range: (f64, f64),
    duration: f64,
    window_preds: &[AudioProb],
    column_winner: &[Option<usize>],
    playhead: Option<f64>,
    apply_bounds: bool,
    plot_w: f32,
    plot_h: f32,
    n_cols: usize,
) -> PlotInteraction {
    use egui::{Align2, Color32, Stroke};
    use egui_plot::{
        GridMark, Line, Plot, PlotBounds, PlotImage, PlotPoint, PlotPoints, Polygon, Text,
    };

    let nyquist = AUDIO_DISPLAY_SR as f64 / 2.0;
    let mel_max = 2595.0 * (1.0 + nyquist / 700.0).log10();
    let strip_pad = mel_max * 0.04;
    let strip_h = mel_max * 0.12;
    let strip_y_lo = mel_max + strip_pad;
    let strip_y_hi = strip_y_lo + strip_h;
    let y_max = strip_y_hi;

    let (x_min, x_max) = view_range;
    let span = (x_max - x_min).max(1e-9);

    let segments = segments_from_columns(window_preds, column_winner);
    let texture_id = texture.id();
    let dark_mode = ui.visuals().dark_mode;

    let hz_ticks: Vec<f64> = [0.0_f64, 250.0, 1000.0, 2000.0, 4000.0, 8000.0]
        .iter()
        .filter(|h| **h <= nyquist)
        .copied()
        .collect();

    let (_n_fft, hop_length, _n_mels_meta, _top_db) = mel_meta;
    let n_mels = full_mel.nrows();
    let n_time = full_mel.ncols();
    let frames_per_sec = AUDIO_DISPLAY_SR as f64 / hop_length as f64;

    let plot_response = Plot::new("audio_spectrogram_plot")
        .width(plot_w)
        .height(plot_h)
        .allow_zoom([true, false])
        .allow_drag([true, false])
        .allow_scroll([true, false])
        .allow_boxed_zoom(false)
        .allow_double_click_reset(true)
        .show_x(true)
        .show_y(true)
        .show_grid([false, true])
        .show_axes([true, true])
        .show_background(false)
        .show_crosshair(true)
        .set_margin_fraction(egui::vec2(0.0, 0.0))
        .default_x_bounds(x_min, x_max)
        .default_y_bounds(0.0, y_max)
        .x_axis_formatter(move |mark, _range| {
            let t = mark.value;
            if span >= 10.0 {
                format!("{:.0}s", t)
            } else if span >= 2.0 {
                format!("{:.1}s", t)
            } else {
                format!("{:.2}s", t)
            }
        })
        .y_axis_formatter(move |mark, _range| {
            let mel = mark.value;
            if mel < -0.5 || mel > mel_max + 0.5 {
                return String::new();
            }
            let hz = 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
            if hz >= 1000.0 {
                format!("{:.0}k", hz / 1000.0)
            } else if hz < 1.0 {
                String::from("0")
            } else {
                format!("{:.0}", hz)
            }
        })
        .y_grid_spacer({
            let hz_ticks = hz_ticks.clone();
            move |_input| {
                hz_ticks
                    .iter()
                    .map(|hz| {
                        let mel = 2595.0 * (1.0 + hz / 700.0).log10();
                        GridMark {
                            value: mel,
                            step_size: mel_max / 6.0,
                        }
                    })
                    .collect()
            }
        })
        .label_formatter(move |name, pos| {
            if let Some(rest) = name.strip_prefix("seg|") {
                let parts: Vec<&str> = rest.splitn(4, '|').collect();
                if parts.len() == 4 {
                    return format!(
                        "{}\n{}% confidence\n{}s – {}s",
                        parts[0], parts[1], parts[2], parts[3]
                    );
                }
            }
            let t = pos.x;
            let mel = pos.y;
            let mins = (t / 60.0).floor() as i64;
            let secs = t - (mins as f64) * 60.0;
            let time_str = if mins > 0 {
                format!("{:02}:{:06.3}", mins, secs)
            } else {
                format!("{:.3} s", t)
            };
            let mut lines: Vec<String> = vec![time_str];
            if mel >= 0.0 && mel <= mel_max {
                let hz = 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
                let hz_str = if hz >= 1000.0 {
                    format!("{:.2} kHz", hz / 1000.0)
                } else {
                    format!("{:.0} Hz", hz)
                };
                let col_f = (t * frames_per_sec).clamp(0.0, (n_time - 1) as f64);
                let col = col_f.round() as usize;
                let row_f = ((mel / mel_max) * (n_mels as f64 - 1.0))
                    .clamp(0.0, (n_mels - 1) as f64);
                let row = row_f.round() as usize;
                let db = full_mel[[row, col]];
                lines.push(hz_str);
                lines.push(format!("{:+.1} dB", db));
            }
            // Predictions overlapping the cursor's time (top 3 by prob).
            let mut hits: Vec<(&str, f32, f32, f32)> = window_preds
                .iter()
                .filter(|p| t >= p.start as f64 && t < p.end as f64)
                .map(|p| {
                    (
                        p.prediction.label.as_str(),
                        p.prediction.prob,
                        p.start,
                        p.end,
                    )
                })
                .collect();
            hits.sort_by(|a, b| {
                a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
            });
            if !hits.is_empty() {
                lines.push(String::new());
                lines.push(String::from("AI predictions here:"));
                for (label, prob, s, e) in hits.iter().take(6) {
                    lines.push(format!(
                        "  {} — {:.0}%  ({:.2}–{:.2}s)",
                        label,
                        prob * 100.0,
                        s,
                        e
                    ));
                }
                if hits.len() > 6 {
                    lines.push(format!("  …and {} more", hits.len() - 6));
                }
            }
            lines.join("\n")
        })
        .show(ui, |plot_ui| {
            // External state takes precedence: snap the plot to our requested
            // view range on the frame that requested it. Bounds modifications
            // are queued and applied AFTER this closure runs (last-write-wins),
            // so we issue exactly one SetX/SetY pair per frame.
            //
            // y is never touched by interaction (drag/zoom/scroll are x-only),
            // so we only need to set y bounds when we're also forcing x.
            if apply_bounds {
                plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                    [x_min, 0.0],
                    [x_max, y_max],
                ));
            }

            plot_ui.image(
                PlotImage::new(
                    "spec",
                    texture_id,
                    PlotPoint::new((x_min + x_max) / 2.0, mel_max / 2.0),
                    egui::vec2(span as f32, mel_max as f32),
                )
                .allow_hover(false),
            );

            // Pixel width per plot x-unit, for the inline-label fit check.
            let xform = plot_ui.transform();
            let p_left = xform.position_from_point(&PlotPoint::new(x_min, 0.0)).x;
            let p_right = xform.position_from_point(&PlotPoint::new(x_max, 0.0)).x;
            let px_per_unit = ((p_right - p_left).abs() as f64) / span;
            let label_fit_units = 60.0 / px_per_unit.max(0.001);

            let n_cols_f = n_cols as f64;
            for (idx, seg) in segments.iter().enumerate() {
                let x1 = x_min + (seg.col_start as f64 / n_cols_f) * span;
                let x2 = x_min + ((seg.col_end + 1) as f64 / n_cols_f) * span;
                let c = class_color(seg.class_id);
                let alpha = (180.0 + 75.0 * seg.max_prob).clamp(180.0, 255.0) as u8;
                let fill = Color32::from_rgba_unmultiplied(c[0], c[1], c[2], alpha);

                let prob_pct = (seg.max_prob * 100.0).round() as i32;
                let name = format!("seg|{}|{}|{:.2}|{:.2}", seg.label, prob_pct, x1, x2);

                let pts: Vec<[f64; 2]> = vec![
                    [x1, strip_y_lo],
                    [x2, strip_y_lo],
                    [x2, strip_y_hi],
                    [x1, strip_y_hi],
                ];
                plot_ui.polygon(
                    Polygon::new(name, PlotPoints::from(pts))
                        .fill_color(fill)
                        .stroke(Stroke::NONE),
                );

                if (x2 - x1) > label_fit_units {
                    let luminance =
                        0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32;
                    let text_color = if luminance > 150.0 {
                        Color32::BLACK
                    } else {
                        Color32::WHITE
                    };
                    plot_ui.text(
                        Text::new(
                            format!("txt_{idx}"),
                            PlotPoint::new(
                                (x1 + x2) / 2.0,
                                (strip_y_lo + strip_y_hi) / 2.0,
                            ),
                            format!("{}  {}%", seg.label, prob_pct),
                        )
                        .color(text_color)
                        .anchor(Align2::CENTER_CENTER)
                        .allow_hover(false),
                    );
                }
            }

            if let Some(ph) = playhead {
                if ph >= x_min && ph <= x_max {
                    let color = if dark_mode {
                        Color32::WHITE
                    } else {
                        Color32::from_rgb(20, 20, 20)
                    };
                    let pts: Vec<[f64; 2]> = vec![[ph, 0.0], [ph, mel_max]];
                    plot_ui.line(
                        Line::new("playhead", PlotPoints::from(pts))
                            .color(color)
                            .width(2.0)
                            .allow_hover(false),
                    );
                }
            }
        });

    // Read back the plot's bounds — these reflect any pan/zoom interaction —
    // and clamp to [0, duration].
    let bounds = plot_response.transform.bounds();
    let new_min = bounds.min()[0].max(0.0);
    let new_max = bounds.max()[0].min(duration).max(new_min + 1e-6);
    let new_view_range = if (new_min - x_min).abs() > 1e-4 || (new_max - x_max).abs() > 1e-4 {
        Some((new_min, new_max))
    } else {
        None
    };

    // Detect a pure click (no drag) for seek-to-time.
    let mut clicked_time: Option<f64> = None;
    let resp = &plot_response.response;
    if resp.clicked() {
        if let Some(screen_pos) = resp.interact_pointer_pos() {
            let plot_pos = plot_response.transform.value_from_position(screen_pos);
            let t = plot_pos.x.clamp(0.0, duration);
            // Ignore clicks inside the strip area — those land on a bar, not the spec.
            if plot_pos.y >= 0.0 && plot_pos.y <= mel_max {
                clicked_time = Some(t);
            }
        }
    }

    PlotInteraction {
        new_view_range,
        clicked_time,
    }
}

/// Render a texture for an arbitrary visible time range from a precomputed mel
/// spectrogram covering the entire audio. The mel's time-axis indexing
/// (`time = frame * hop_length / sample_rate`) drives the bilinear sampling.
fn mel_slice_to_imgbuf(
    full_mel: &ndarray::Array2<f32>,
    sample_rate: u32,
    hop_length: usize,
    top_db: f32,
    view_start: f64,
    view_end: f64,
    target_width: usize,
    target_height: usize,
    window_preds: &[AudioProb],
    column_winner: &[Option<usize>],
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (n_mels, n_time) = full_mel.dim();
    let mut pixels = Vec::with_capacity(target_width * target_height * 4);

    let column_color: Vec<Option<[u8; 3]>> = column_winner
        .iter()
        .map(|w| w.map(|i| class_color(window_preds[i].prediction.class_id)))
        .collect();

    let frames_per_sec = sample_rate as f64 / hop_length as f64;
    let view_dur = (view_end - view_start).max(1e-9);

    // For each y-row of the target texture, sample the appropriate mel-bin row.
    let sy = target_height as f32 / n_mels as f32;

    for row in 0..target_height {
        let src_row = ((target_height - 1 - row) as f32 / sy).min(n_mels as f32 - 1.0);
        let row_lo = src_row.floor() as usize;
        let row_hi = (row_lo + 1).min(n_mels - 1);
        let fr = src_row - row_lo as f32;

        for col in 0..target_width {
            // Column center time → mel frame index (fractional).
            let t = view_start + ((col as f64 + 0.5) / target_width as f64) * view_dur;
            let src_col_f = (t * frames_per_sec).clamp(0.0, (n_time - 1) as f64);
            let col_lo = src_col_f.floor() as usize;
            let col_hi = (col_lo + 1).min(n_time - 1);
            let fc = (src_col_f - col_lo as f64) as f32;
            let v = full_mel[[row_lo, col_lo]] * (1.0 - fc) * (1.0 - fr)
                + full_mel[[row_lo, col_hi]] * fc * (1.0 - fr)
                + full_mel[[row_hi, col_lo]] * (1.0 - fc) * fr
                + full_mel[[row_hi, col_hi]] * fc * fr;
            let t_norm = ((v + top_db) / top_db).clamp(0.0, 1.0);
            match column_color[col] {
                Some(c) => {
                    let [r, g, b] = class_colormap(c, t_norm);
                    pixels.extend_from_slice(&[r, g, b, 255]);
                }
                None => {
                    let [r, g, b] = viridis(t_norm);
                    pixels.extend_from_slice(&[r, g, b, 255]);
                }
            }
        }
    }
    ImageBuffer::from_raw(target_width as u32, target_height as u32, pixels).unwrap()
}

pub(super) struct AudioBufferSource {
    samples: std::sync::Arc<Vec<f32>>,
    sample_rate: u32,
    channels: u16,
    pos: usize,
}

impl AudioBufferSource {
    fn new_from(audio: &AudioData, start_secs: f64) -> Self {
        let mut mono = audio.clone();
        if mono.channels > 1 {
            mono = mono.to_mono();
        }
        let start_sample = (start_secs * mono.sample_rate as f64).round() as usize;
        let samples = if start_sample < mono.samples.len() {
            mono.samples[start_sample..].to_vec()
        } else {
            vec![]
        };
        Self {
            samples: std::sync::Arc::new(samples),
            sample_rate: mono.sample_rate,
            channels: mono.channels.max(1),
            pos: 0,
        }
    }
}

impl Iterator for AudioBufferSource {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.samples.len() { return None; }
        let s = self.samples[self.pos];
        self.pos += 1;
        Some(s)
    }
}

impl Source for AudioBufferSource {
    fn current_span_len(&self) -> Option<usize> { None }
    fn channels(&self) -> rodio::ChannelCount { rodio::ChannelCount::new(self.channels).unwrap_or(rodio::ChannelCount::MIN) }
    fn sample_rate(&self) -> rodio::SampleRate { rodio::SampleRate::new(self.sample_rate).unwrap_or(rodio::SampleRate::MIN) }
    fn total_duration(&self) -> Option<std::time::Duration> {
        let frames = self.samples.len() / self.channels.max(1) as usize;
        Some(std::time::Duration::from_secs_f64(frames as f64 / self.sample_rate as f64))
    }
}
