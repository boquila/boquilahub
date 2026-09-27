use super::{Gui, Message, Mode};
use crate::{api::abstractions::Pred, localization::*};
use std::time::SystemTime;

impl Gui {
    pub(super) fn file_header(&mut self, ui: &mut egui::Ui) {
        let can_analyze = match self.mode {
            Mode::Audio if self.ep_selected.is_local() => self.is_audio_model(),
            Mode::Audio => self.rest_client.is_some(),
            _ => self.can_run_image_ai(),
        };
        let (changed, analyze) = match self.mode {
            Mode::Image => self.image_browser.show(
                ui,
                &self.selected_imgs,
                &mut self.image_texture_n,
                &self.lang,
                can_analyze.then_some(!self.img_state.is_processing),
            ),
            Mode::Audio => self.audio_browser.show(
                ui,
                &self.selected_audios,
                &mut self.audio_texture_n,
                &self.lang,
                can_analyze.then_some(!self.audio_state.is_processing),
            ),
            Mode::Video => self.video_browser.show(
                ui,
                &self.selected_videos,
                &mut self.video_texture_n,
                &self.lang,
                can_analyze.then_some(!self.video_state.is_processing),
            ),
            Mode::Feed => return,
        };
        if changed {
            match self.mode {
                Mode::Image => {
                    self.image_view.reset();
                    self.paint(ui, self.image_texture_n - 1);
                }
                Mode::Audio => {
                    if self.load_current_audio().is_err() {
                        self.push_toast(Message::Error);
                    }
                }
                Mode::Video => self.load_current_video(ui),
                Mode::Feed => unreachable!(),
            }
        }
        if analyze {
            match self.mode {
                Mode::Image => self.start_single_img_analysis(self.image_texture_n - 1),
                Mode::Audio => self.start_single_audio_analysis(self.audio_texture_n - 1),
                Mode::Video => self.start_video_analysis(),
                Mode::Feed => unreachable!(),
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum DateField {
    Modified,
    Created,
}

impl DateField {
    const ALL: [Self; 2] = [Self::Modified, Self::Created];

    fn label(self, lang: &Lang) -> &'static str {
        translate(
            match self {
                Self::Modified => Key::file_modified,
                Self::Created => Key::file_created,
            },
            lang,
        )
    }

    fn read(self, file: &impl Pred) -> Option<SystemTime> {
        let metadata = file.metadata()?;
        match self {
            Self::Modified => metadata.modified(),
            Self::Created => metadata.created(),
        }
        .ok()
    }
}

/// Preserve source indexes for workers; sorting changes only navigation order.
#[derive(Default)]
pub(super) struct Browser {
    order: Vec<usize>,
    field: Option<DateField>,
    descending: bool,
}

impl Browser {
    /// Returns (selection changed, Analyze clicked). `analyze` controls button availability.
    fn show(
        &mut self,
        ui: &mut egui::Ui,
        files: &[impl Pred],
        index: &mut usize,
        lang: &Lang,
        analyze: Option<bool>,
    ) -> (bool, bool) {
        if files.is_empty() {
            return (false, false);
        }
        let old_index = *index;
        *index = (*index).clamp(1, files.len());
        let mut analyze_clicked = false;
        let mut position = 1;
        // Align the header with the slider track, leaving room for its value editor.
        let width = if files.len() > 1 {
            (ui.available_width() - 110.0).max(180.0)
        } else {
            ui.available_width()
        };
        // Bound the row's height before aligning Sort at the right edge.
        ui.horizontal(|ui| {
            ui.set_max_width(width);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.sort_menu(ui, files, index, lang);
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    position = self
                        .order
                        .iter()
                        .position(|i| *i + 1 == *index)
                        .unwrap_or(0)
                        + 1;
                    if files.len() > 1 {
                        for (target, icon, key) in [
                            (position - 1, "⏮", Key::prev),
                            (position + 1, "⏭", Key::next),
                        ] {
                            if ui
                                .add_enabled(
                                    (1..=files.len()).contains(&target),
                                    egui::Button::new(icon),
                                )
                                .on_hover_text(translate(key, lang))
                                .clicked()
                            {
                                position = target;
                            }
                        }
                        ui.separator();
                    }
                    let file = &files[self.order[position - 1]];
                    if !file.is_processed() {
                        ui.separator();
                        ui.label(
                            egui::RichText::new(translate(Key::not_analysed, lang))
                                .weak()
                                .small(),
                        );
                    }
                    if let Some(enabled) = analyze {
                        ui.separator();
                        analyze_clicked = ui
                            .add_enabled(enabled, egui::Button::new(translate(Key::analyze, lang)))
                            .clicked();
                    }
                    let name = file
                        .file_path()
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or(translate(Key::unknown_file, lang));
                    if files.len() > 1 {
                        ui.label(
                            egui::RichText::new(format!("·  {} / {}", position, files.len()))
                                .weak(),
                        );
                    }
                    ui.add(egui::Label::new(egui::RichText::new(name).strong()).truncate())
                        .on_hover_text(name);
                });
            });
        });
        if files.len() > 1 {
            ui.spacing_mut().slider_width = width;
            ui.add(egui::Slider::new(&mut position, 1..=files.len()));
        }
        *index = self.order[position - 1] + 1;
        ui.add_space(4.0);
        (old_index != *index, analyze_clicked)
    }

    fn sort_menu(
        &mut self,
        ui: &mut egui::Ui,
        files: &[impl Pred],
        index: &mut usize,
        lang: &Lang,
    ) {
        let mut changed = false;
        if self.field.is_some() {
            let (icon, key) = if self.descending {
                ("↓", Key::date_descending)
            } else {
                ("↑", Key::date_ascending)
            };
            // The row flows right to left, so draw the arrow before Sort.
            changed = ui
                .button(icon)
                .on_hover_text(translate(key, lang))
                .clicked();
            if changed {
                self.descending = !self.descending;
            }
        }
        egui::Popup::menu(&ui.button(translate(Key::sort, lang)))
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                for field in DateField::ALL {
                    changed |= ui
                        .radio_value(&mut self.field, Some(field), field.label(lang))
                        .clicked();
                }
            });
        if changed || self.order.len() != files.len() {
            self.sort_by_date(files);
        }
        if changed && self.field.is_some() {
            *index = self.order[0] + 1;
        }
    }

    fn sort_by_date(&mut self, files: &[impl Pred]) {
        self.order = (0..files.len()).collect();
        let Some(field) = self.field else {
            return;
        };
        let dates: Vec<_> = files.iter().map(|file| field.read(file)).collect();
        self.order.sort_by(|a, b| match (dates[*a], dates[*b]) {
            (Some(a), Some(b)) => {
                if self.descending {
                    b.cmp(&a)
                } else {
                    a.cmp(&b)
                }
            }
            // Missing dates stay last, and equal dates retain input order.
            (a, b) => a.is_none().cmp(&b.is_none()),
        });
    }
}
