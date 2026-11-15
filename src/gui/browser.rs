use super::{Gui, Message, Mode};
use crate::{api::abstractions::Pred, localization::*};

impl Gui {
    pub(super) fn file_header(&mut self, ui: &mut egui::Ui) {
        let can_analyze = match self.mode {
            Mode::Audio if self.ep_selected.is_local() => self.is_audio_model(),
            Mode::Audio => self.rest_client.is_some(),
            _ => self.can_run_image_ai(),
        };
        let (changed, analyze, plot) = match self.mode {
            Mode::Image => self.image_browser.show(
                ui,
                &self.selected_imgs,
                &mut self.image_texture_n,
                &self.lang,
                can_analyze.then_some(!self.img_state.is_processing),
                self.embedding_kinds.0 || self.embedding_kinds.1,
            ),
            Mode::Audio => self.audio_browser.show(
                ui,
                &self.selected_audios,
                &mut self.audio_texture_n,
                &self.lang,
                can_analyze.then_some(!self.audio_state.is_processing),
                false,
            ),
            Mode::Video => self.video_browser.show(
                ui,
                &self.selected_videos,
                &mut self.video_texture_n,
                &self.lang,
                can_analyze.then_some(!self.video_state.is_processing),
                false,
            ),
            Mode::Feed => return,
        };
        if plot {
            if let Some(existing) = self.embedding_plot.as_mut() {
                existing.open = true;
            } else {
                self.embedding_plot = Some(super::embedding_plot::EmbeddingPlot::new(
                    &self.selected_imgs,
                ));
            }
        }
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

/// Preserve source indexes for workers; sorting changes only navigation order.
#[derive(Default)]
pub(super) struct Browser {
    order: Vec<usize>,
    positions: Vec<usize>,
    field: Option<DateField>,
    descending: bool,
}

impl Browser {
    /// Returns (selection changed, Analyze clicked, Plot clicked).
    fn show(
        &mut self,
        ui: &mut egui::Ui,
        files: &[impl Pred],
        index: &mut usize,
        lang: &Lang,
        analyze: Option<bool>,
        show_plot: bool,
    ) -> (bool, bool, bool) {
        if files.is_empty() {
            return (false, false, false);
        }
        let old_index = *index;
        *index = (*index).clamp(1, files.len());
        let mut analyze_clicked = false;
        let mut plot_clicked = false;
        ui.scope(|ui| {
            let file = &files[*index - 1];
            // Horizontal rows bound the height, including inside a ScrollArea.
            ui.horizontal_wrapped(|ui| {
                let name = file
                    .file_path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(translate(Key::unknown_file, lang));
                ui.label(egui::RichText::new(name).strong())
                    .on_hover_text(name);
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
                    let key = if enabled {
                        Key::analyze
                    } else {
                        Key::analysing
                    };
                    analyze_clicked = ui
                        .add_enabled(enabled, egui::Button::new(translate(key, lang)))
                        .clicked();
                }
            });
            if files.len() > 1 || show_plot {
                ui.spacing_mut().interact_size = egui::vec2(28.0, 28.0);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if files.len() > 1 {
                            self.sort_menu(ui, files, index, lang);
                        }
                        if show_plot {
                            plot_clicked = ui.button(translate(Key::plot, lang)).clicked();
                        }
                        if files.len() == 1 {
                            return;
                        }
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            let mut position = self.positions[*index - 1];
                            let button_size = egui::Vec2::splat(28.0);
                            if ui
                                .add_enabled(
                                    position > 1,
                                    egui::Button::new("‹").min_size(button_size),
                                )
                                .on_hover_text(translate(Key::prev, lang))
                                .clicked()
                            {
                                position -= 1;
                            }
                            ui.add(
                                egui::DragValue::new(&mut position)
                                    .range(1..=files.len())
                                    .speed(1.0)
                                    .update_while_editing(false),
                            )
                            .on_hover_text(translate(Key::file_number_hint, lang));
                            ui.label(egui::RichText::new(format!("/ {}", files.len())).weak());
                            if ui
                                .add_enabled(
                                    position < files.len(),
                                    egui::Button::new("›").min_size(button_size),
                                )
                                .on_hover_text(translate(Key::next, lang))
                                .clicked()
                            {
                                position += 1;
                            }
                            ui.spacing_mut().slider_width = ui.available_width().max(1.0);
                            ui.add(
                                egui::Slider::new(&mut position, 1..=files.len()).show_value(false),
                            );
                            *index = self.order[position - 1] + 1;
                        });
                    });
                });
            }
        });
        let changed = old_index != *index;
        if changed {
            ui.ctx().request_repaint();
        }
        ui.add_space(4.0);
        (changed, analyze_clicked, plot_clicked)
    }

    fn sort_menu(
        &mut self,
        ui: &mut egui::Ui,
        files: &[impl Pred],
        index: &mut usize,
        lang: &Lang,
    ) {
        let mut changed = false;
        ui.menu_button("↑↓", |ui| {
            for (field, key) in [
                (None, Key::original_order),
                (Some(DateField::Modified), Key::file_modified),
                (Some(DateField::Created), Key::file_created),
            ] {
                changed |= ui
                    .selectable_value(&mut self.field, field, translate(key, lang))
                    .changed();
            }
            ui.separator();
            ui.add_enabled_ui(self.field.is_some(), |ui| {
                for (descending, icon, key) in [
                    (false, "↑", Key::date_ascending),
                    (true, "↓", Key::date_descending),
                ] {
                    changed |= ui
                        .selectable_value(
                            &mut self.descending,
                            descending,
                            format!("{icon} {}", translate(key, lang)),
                        )
                        .changed();
                }
            });
        })
        .response
        .on_hover_text(translate(Key::sort_by, lang));
        if changed || self.order.len() != files.len() {
            let position = self.positions.get(*index - 1).copied().unwrap_or(*index);
            self.sort_by_date(files);
            self.positions.resize(files.len(), 0);
            for (position, &index) in self.order.iter().enumerate() {
                self.positions[index] = position + 1;
            }
            if changed {
                // A sort changes which file occupies the current browser position.
                *index = self.order[position - 1] + 1;
            }
        }
    }

    fn sort_by_date(&mut self, files: &[impl Pred]) {
        self.order = (0..files.len()).collect();
        let Some(field) = self.field else {
            return;
        };
        let dates: Vec<_> = files
            .iter()
            .map(|file| {
                let metadata = file.metadata()?;
                match field {
                    DateField::Modified => metadata.modified(),
                    DateField::Created => metadata.created(),
                }
                .ok()
            })
            .collect();
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
