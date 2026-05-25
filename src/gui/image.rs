use super::{imgbuf_to_texture, Gui, Mode, OpenDialog};
use crate::api::abstractions::*;
use crate::api::bq::process_imgbuf;
use crate::api::export;
use crate::api::render::*;
use crate::api::rest::detect_remotely;
use crate::localization::*;
use image::{ImageBuffer, Rgba};
use std::fs;

const MIN_PREVIEW_H: f32 = 240.0;
const SIDE_PANEL_W: f32 = 240.0;

impl Gui {
    // ---------- texture loading ----------

    pub(super) fn paint(&mut self, ui: &egui::Ui, i: usize) {
        let Some(predimg) = self.selected_files.get(i) else { return; };
        let Ok(loaded) = image::open(&predimg.file_path) else { return; };
        let img = loaded.into_rgba8();
        self.img_state.texture = imgbuf_to_texture(&img, ui);
        self.mask_textures = build_mask_textures(predimg, ui);
    }

    fn draw_gui(&self, predimg: &PredImg) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        let mut img = image::open(&predimg.file_path).unwrap().into_rgb8();
        if predimg.wasprocessed {
            if predimg.aioutput.as_ref().unwrap().is_empty() {
                draw_no_predictions(&mut img, Some(&self.lang));
            } else {
                draw_aioutput(&mut img, predimg.aioutput.as_ref().unwrap());
            }
        }
        image::DynamicImage::ImageRgb8(img).to_rgba8()
    }

    pub(super) fn save_gui(&self, predimg: &PredImg) {
        let img_data = image::DynamicImage::ImageRgba8(self.draw_gui(predimg)).to_rgb8();
        let filename = export::prepare_export_img(&predimg.file_path);
        img_data.save(&filename).unwrap();
    }

    // ---------- analysis lifecycle ----------

    pub(super) fn start_single_img_analysis(&mut self, target: usize) {
        if target >= self.selected_files.len() || self.img_state.is_processing {
            return;
        }
        self.selected_files[target].reset();

        let (tx, mut cancel_rx) = self.img_state.start();
        let predimg = self.selected_files[target].clone();

        let api_endpoint = self.get_endpoint();
        let is_remote = !self.ep_selected.is_local();
        tokio::spawn(async move {
            if cancel_rx.try_recv().is_ok() {
                return;
            }
            let bbox = if is_remote {
                let buffer = fs::read(&predimg.file_path).unwrap();
                match detect_remotely(api_endpoint.as_ref().unwrap(), buffer).await {
                    Ok(result) => result,
                    Err(_) => return,
                }
            } else {
                let img = image::open(&predimg.file_path).unwrap().into_rgb8();
                tokio::task::spawn_blocking(move || process_imgbuf(&img))
                    .await
                    .unwrap()
            };

            let _ = tx.send((target, bbox));
        });
    }

    pub(super) fn start_img_analysis(&mut self) {
        if self.process_all_imgs {
            self.selected_files
                .iter_mut()
                .for_each(|pred_img| pred_img.reset());
        }
        let (tx, mut cancel_rx) = self.img_state.start();
        let copy_predigms = self.selected_files.clone();

        let api_endpoint = self.get_endpoint();
        let is_remote = !self.ep_selected.is_local();
        tokio::spawn(async move {
            for (i, predimg) in copy_predigms.iter().enumerate() {
                if predimg.wasprocessed {
                    continue;
                }
                if cancel_rx.try_recv().is_ok() {
                    break;
                }
                let bbox = if is_remote {
                    let buffer = fs::read(&predimg.file_path).unwrap();
                    match detect_remotely(api_endpoint.as_ref().unwrap(), buffer).await {
                        Ok(result) => result,
                        Err(_) => break,
                    }
                } else {
                    let img = image::open(&predimg.file_path).unwrap().into_rgb8();
                    tokio::task::spawn_blocking(move || process_imgbuf(&img))
                        .await
                        .unwrap()
                };

                if tx.send((i, bbox)).is_err() {
                    break;
                }
            }
        });
    }

    pub(super) fn img_handle_results(&mut self, ui: &egui::Ui) {
        let (updates, closed) = self.img_state.drain();
        for (i, bbox) in updates {
            self.selected_files[i].aioutput = Some(bbox);
            self.selected_files[i].wasprocessed = true;
            if i == self.image_texture_n - 1 {
                self.paint(ui, i);
            }
        }
        if closed {
            self.img_state.finish();
        }
        self.img_state.progress_bar = self.selected_files.get_progress();
        ui.request_repaint();
    }

    pub(super) fn process_all_dialog(&mut self, ui: &egui::Ui) {
        if self.dialog != OpenDialog::ProcessAll {
            return;
        }
        egui::Window::new(self.t(Key::process_everything))
            .collapsible(false)
            .resizable(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button(self.t(Key::yes)).clicked() {
                        self.process_all_imgs = true;
                        self.start_img_analysis();
                        self.dialog = OpenDialog::None;
                    }
                    ui.add_space(8.0);
                    if ui.button(self.t(Key::no_only_missing_data)).clicked() {
                        self.process_all_imgs = false;
                        self.start_img_analysis();
                        self.dialog = OpenDialog::None;
                    }
                    ui.add_space(8.0);
                    if ui.button(self.t(Key::cancel)).clicked() {
                        self.dialog = OpenDialog::None;
                    }
                });
            });
    }

    // ---------- left-panel widget (Analyze / Export) ----------

    pub(super) fn img_analysis_widget(&mut self, ui: &mut egui::Ui) {
        if self.selected_files.is_empty() || !self.can_run_image_ai() {
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
                if self.selected_files.get_progress() == 0.0 {
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

        if self.dialog == OpenDialog::Export && self.mode == Mode::Image {
            egui::Window::new(self.t(Key::export))
                .collapsible(false)
                .resizable(false)
                .show(ui, |ui| {
                    if ui.button(self.t(Key::export_predictions)).clicked() {
                        for file in self.selected_files.clone() {
                            tokio::spawn(async move {
                                let _ = file.write_predictions();
                            });
                        }
                        let msg = self.t(Key::saved_next_to_originals).to_string();
                        self.process_done_with(msg);
                        self.dialog = OpenDialog::None;
                    }

                    if ui
                        .button(self.t(Key::export_imgs_with_predictions))
                        .clicked()
                    {
                        for file in &self.selected_files {
                            if file.wasprocessed {
                                self.save_gui(file);
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

        self.process_all_dialog(ui);
    }

    // ---------- main image viewer ----------

    pub(super) fn ui_image(&mut self, ui: &mut egui::Ui) {
        if self.selected_files.is_empty() {
            return;
        }
        let n = self.selected_files.len();
        if self.image_texture_n < 1 {
            self.image_texture_n = 1;
        }
        if self.image_texture_n > n {
            self.image_texture_n = n;
        }

        self.draw_image_header(ui, n);

        let i = self.image_texture_n - 1;
        let is_classification = matches!(
            self.selected_files[i].aioutput.as_ref(),
            Some(AIOutputs::Classification(_))
        );
        let has_non_empty_output = self.selected_files[i].wasprocessed
            && self
                .selected_files[i]
                .aioutput
                .as_ref()
                .map(|a| !a.is_empty())
                .unwrap_or(false);
        let show_side_panel = is_classification && has_non_empty_output;
        let has_spatial_output = matches!(
            self.selected_files[i].aioutput.as_ref(),
            Some(AIOutputs::ObjectDetection(b)) if !b.is_empty()
        ) || matches!(
            self.selected_files[i].aioutput.as_ref(),
            Some(AIOutputs::Segmentation(s)) if !s.is_empty()
        );

        let avail = ui.available_size_before_wrap();
        // Below this width, side-by-side leaves the preview too cramped — drop
        // the classification panel under it instead.
        const STACK_BELOW_WIDTH: f32 = 720.0;
        let stack_side = show_side_panel && avail.x < STACK_BELOW_WIDTH;
        let beside_side = show_side_panel && !stack_side;

        let side_gap = if beside_side { 12.0 } else { 0.0 };
        let side_w = if beside_side {
            SIDE_PANEL_W.min((avail.x * 0.32).max(180.0))
        } else {
            0.0
        };
        let preview_w = (avail.x - side_w - side_gap).max(200.0);
        let echo_strip_h: f32 = if has_spatial_output { 30.0 } else { 0.0 };
        let echo_strip_gap: f32 = if has_spatial_output { 6.0 } else { 0.0 };
        // When stacking, the preview keeps the lion's share and the
        // classification panel takes the rest — central panel scrolls if
        // its content needs more than that.
        let preview_h = if stack_side {
            (avail.y * 0.62 - echo_strip_h - echo_strip_gap).max(MIN_PREVIEW_H)
        } else {
            (avail.y - 8.0 - echo_strip_h - echo_strip_gap).max(MIN_PREVIEW_H)
        };

        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_max_width(preview_w);
                let echo = self.draw_image_preview(ui, preview_w, preview_h);
                if has_spatial_output {
                    ui.add_space(echo_strip_gap);
                    draw_echo_strip(ui, echo.as_ref(), echo_strip_h, preview_w);
                }
                if stack_side {
                    ui.add_space(8.0);
                    self.draw_classification_panel(ui);
                }
            });

            if beside_side {
                ui.add_space(side_gap);
                ui.vertical(|ui| {
                    ui.set_max_width(side_w);
                    self.draw_classification_panel(ui);
                });
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
            let predimg = &self.selected_files[new_index - 1];
            let name = predimg
                .file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(self.t(Key::unknown_file));
            super::nav_filename(ui, name, new_index, n);

            if predimg.wasprocessed {
                if let Some(aio) = predimg.aioutput.as_ref() {
                    if let Some(summary) = summary_line(aio, &self.lang) {
                        ui.separator();
                        ui.label(egui::RichText::new(summary).strong());
                    }
                }
            } else {
                ui.separator();
                ui.label(
                    egui::RichText::new(self.t(Key::not_analysed))
                        .weak()
                        .small(),
                );
            }

            if can_analyze {
                ui.separator();
                let resp = ui
                    .add_enabled(
                        !self.img_state.is_processing,
                        egui::Button::new(self.t(Key::analyze_this_image)),
                    )
                    .on_hover_text(self.t(Key::analyze_this_image_hint));
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
            let predimg = &self.selected_files[i];

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

    fn draw_classification_panel(&self, ui: &mut egui::Ui) {
        let i = self.image_texture_n - 1;
        let probs = match self.selected_files[i].aioutput.as_ref() {
            Some(AIOutputs::Classification(probs)) => probs,
            _ => return,
        };
        if probs.is_empty() {
            ui.label(egui::RichText::new(self.t(Key::no_predictions)).weak());
            return;
        }

        ui.vertical(|ui| {
            ui.label(egui::RichText::new(self.t(Key::predictions)).heading());
            ui.add_space(4.0);

            let mut ranked: Vec<&Prob> = probs.iter().collect();
            ranked.sort_by(|a, b| {
                b.prob
                    .partial_cmp(&a.prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Cap rows so a 1000-class model doesn't render 1000 bars + galleys
            // every frame. Always show at least 3 for context.
            const MAX_ROWS: usize = 20;
            const MIN_PROB_VISIBLE: f32 = 0.005;
            let signal_count = ranked
                .iter()
                .position(|p| p.prob < MIN_PROB_VISIBLE)
                .unwrap_or(ranked.len());
            let visible = signal_count.min(MAX_ROWS).max(3).min(ranked.len());
            let hidden = ranked.len().saturating_sub(visible);

            let dark = ui.visuals().dark_mode;
            let track_color = if dark {
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 20)
            } else {
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 18)
            };
            let bar_h: f32 = 22.0;

            for (rank, p) in ranked.iter().take(visible).enumerate() {
                let c = class_color(p.class_id);
                let color = egui::Color32::from_rgb(c[0], c[1], c[2]);

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("■").color(color).monospace());
                    let label_text = if rank == 0 {
                        egui::RichText::new(&p.label).strong()
                    } else {
                        egui::RichText::new(&p.label)
                    };
                    ui.label(label_text);
                });

                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width().max(80.0), bar_h),
                    egui::Sense::hover(),
                );
                let painter = ui.painter();
                painter.rect_filled(rect, 4.0, track_color);
                let fill_w = rect.width() * p.prob.clamp(0.0, 1.0);
                let fill_rect = egui::Rect::from_min_size(
                    rect.min,
                    egui::vec2(fill_w.max(1.0), rect.height()),
                );
                painter.rect_filled(fill_rect, 4.0, color);
                let txt = format!("{:.1}%", p.prob * 100.0);
                let galley = painter.layout_no_wrap(
                    txt,
                    egui::FontId::proportional(12.0),
                    egui::Color32::WHITE,
                );
                let inside_fill = galley.size().x + 10.0 < fill_w;
                let text_x = if inside_fill {
                    fill_rect.right() - galley.size().x - 6.0
                } else {
                    fill_rect.right() + 6.0
                };
                let text_color = if inside_fill {
                    egui::Color32::WHITE
                } else if dark {
                    egui::Color32::from_gray(220)
                } else {
                    egui::Color32::from_gray(40)
                };
                painter.galley(
                    egui::pos2(
                        text_x,
                        rect.center().y - galley.size().y / 2.0,
                    ),
                    galley,
                    text_color,
                );

                ui.add_space(4.0);
            }

            if hidden > 0 {
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(and_more(hidden, &self.lang))
                        .weak()
                        .small(),
                );
            }
        });
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

    let c = class_color(e.class_id);
    let color = egui::Color32::from_rgb(c[0], c[1], c[2]);

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

fn summary_line(aio: &AIOutputs, lang: &Lang) -> Option<String> {
    match aio {
        AIOutputs::ObjectDetection(b) if b.is_empty() => {
            Some(translate(Key::no_predictions, lang).into())
        }
        AIOutputs::ObjectDetection(b) => {
            let noun = if b.len() == 1 {
                translate(Key::detection, lang)
            } else {
                translate(Key::detections, lang)
            };
            Some(format!("{} {}", b.len(), noun))
        }
        AIOutputs::Segmentation(s) if s.is_empty() => {
            Some(translate(Key::no_predictions, lang).into())
        }
        AIOutputs::Segmentation(s) => {
            let noun = if s.len() == 1 {
                translate(Key::segment, lang)
            } else {
                translate(Key::segments, lang)
            };
            Some(format!("{} {}", s.len(), noun))
        }
        // Classification: the side panel is the source of truth — singling out
        // a "top" class in the header is misleading when probabilities are noisy.
        AIOutputs::Classification(_) => None,
        AIOutputs::AudioClassification(_) => None,
    }
}

fn and_more(n: usize, lang: &Lang) -> String {
    translate(Key::and_more_fmt, lang).replace("{}", &n.to_string())
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
            let hovered = pick_hovered_bbox(bboxes, hover_pos, rect.min, scale);
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
            let hovered = pick_hovered_seg(segs, hover_pos, rect.min, scale);
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
    }
}

fn pick_hovered_bbox(
    bboxes: &[XYXYc],
    hover_pos: Option<egui::Pos2>,
    origin: egui::Pos2,
    scale: egui::Vec2,
) -> Option<usize> {
    let p = hover_pos?;
    let mut best: Option<(usize, f32)> = None;
    for (idx, b) in bboxes.iter().enumerate() {
        let r = bbox_screen_rect(b, origin, scale);
        if r.contains(p) {
            let area = r.width() * r.height();
            if best.map_or(true, |(_, a)| area < a) {
                best = Some((idx, area));
            }
        }
    }
    best.map(|(i, _)| i)
}

fn pick_hovered_seg(
    segs: &[SEGc],
    hover_pos: Option<egui::Pos2>,
    origin: egui::Pos2,
    scale: egui::Vec2,
) -> Option<usize> {
    let p = hover_pos?;
    let mut best: Option<(usize, f32)> = None;
    for (idx, s) in segs.iter().enumerate() {
        let r = bbox_screen_rect(&s.bbox, origin, scale);
        if r.contains(p) {
            let area = r.width() * r.height();
            if best.map_or(true, |(_, a)| area < a) {
                best = Some((idx, area));
            }
        }
    }
    best.map(|(i, _)| i)
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
    let c = class_color(bbox_color_id(b));
    let color = egui::Color32::from_rgb(c[0], c[1], c[2]);

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
    let galley = painter.layout_no_wrap(
        text.into(),
        egui::FontId::proportional(13.0),
        egui::Color32::WHITE,
    );
    let pad = 8.0;
    let chip_h = 24.0;
    let chip_w = galley.size().x + 14.0;
    let bg = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180);
    let pos = egui::pos2(img_resp.rect.min.x + pad, img_resp.rect.min.y + pad);
    let chip = egui::Rect::from_min_size(pos, egui::vec2(chip_w, chip_h));
    painter.rect_filled(chip, 12.0, bg);
    painter.galley(
        chip.min + egui::vec2(7.0, (chip_h - galley.size().y) / 2.0),
        galley,
        egui::Color32::WHITE,
    );
}

fn bbox_tooltip_ui(ui: &mut egui::Ui, b: &XYXYc, lang: &Lang) {
    let cls_id = bbox_color_id(b);
    let c = class_color(cls_id);
    let color = egui::Color32::from_rgb(c[0], c[1], c[2]);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("■").color(color).monospace());
        ui.label(egui::RichText::new(&b.label).strong());
    });
    ui.label(format!(
        "{:.0}{}",
        b.xyxy.prob * 100.0,
        translate(Key::confidence_pct, lang)
    ));
    let w = (b.xyxy.x2 - b.xyxy.x1).max(0.0).round() as i32;
    let h = (b.xyxy.y2 - b.xyxy.y1).max(0.0).round() as i32;
    ui.label(
        egui::RichText::new(format!("{} × {} px", w, h))
            .weak()
            .small(),
    );

    if let Some(extras) = b.extra_cls.as_ref() {
        if !extras.is_empty() {
            ui.separator();
            ui.label(egui::RichText::new(translate(Key::refined, lang)).strong());
            let mut sorted: Vec<&Prob> = extras.iter().collect();
            sorted.sort_by(|a, b| {
                b.prob
                    .partial_cmp(&a.prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for p in sorted.iter().take(4) {
                tooltip_row(ui, p.class_id, &p.label, p.prob);
            }
        }
    }
}

fn seg_tooltip_ui(ui: &mut egui::Ui, s: &SEGc, lang: &Lang) {
    let cls_id = bbox_color_id(&s.bbox);
    let c = class_color(cls_id);
    let color = egui::Color32::from_rgb(c[0], c[1], c[2]);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("■").color(color).monospace());
        ui.label(egui::RichText::new(&s.bbox.label).strong());
        ui.label(
            egui::RichText::new(format!("· {}", translate(Key::segment, lang)))
                .weak()
                .small(),
        );
    });
    ui.label(format!(
        "{:.0}{}",
        s.bbox.xyxy.prob * 100.0,
        translate(Key::confidence_pct, lang)
    ));
    let w = (s.bbox.xyxy.x2 - s.bbox.xyxy.x1).max(0.0).round() as i32;
    let h = (s.bbox.xyxy.y2 - s.bbox.xyxy.y1).max(0.0).round() as i32;
    ui.label(
        egui::RichText::new(format!("bbox {} × {} px", w, h))
            .weak()
            .small(),
    );
    ui.label(
        egui::RichText::new(format!("mask {} × {}", s.mask.width, s.mask.height))
            .weak()
            .small(),
    );

    if let Some(extras) = s.bbox.extra_cls.as_ref() {
        if !extras.is_empty() {
            ui.separator();
            ui.label(egui::RichText::new(translate(Key::refined, lang)).strong());
            let mut sorted: Vec<&Prob> = extras.iter().collect();
            sorted.sort_by(|a, b| {
                b.prob
                    .partial_cmp(&a.prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for p in sorted.iter().take(4) {
                tooltip_row(ui, p.class_id, &p.label, p.prob);
            }
        }
    }
}

fn classification_tooltip_ui(ui: &mut egui::Ui, probs: &[Prob], lang: &Lang) {
    if probs.is_empty() {
        ui.label(egui::RichText::new(translate(Key::no_predictions, lang)).weak());
        return;
    }
    ui.label(egui::RichText::new(translate(Key::classification, lang)).strong());
    let mut sorted: Vec<&Prob> = probs.iter().collect();
    sorted.sort_by(|a, b| {
        b.prob
            .partial_cmp(&a.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for p in sorted.iter().take(6) {
        tooltip_row(ui, p.class_id, &p.label, p.prob);
    }
    if probs.len() > 6 {
        ui.label(
            egui::RichText::new(and_more(probs.len() - 6, lang)).weak(),
        );
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
