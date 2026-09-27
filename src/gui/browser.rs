use super::{nav_filename, nav_prev_next, nav_slider};
use crate::{api::abstractions::Pred, localization::*};
use std::time::SystemTime;

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
    /// Shared file controls. `analyze` is None when unavailable, otherwise its enabled state.
    pub(super) fn show<T: Pred>(
        &mut self,
        ui: &mut egui::Ui,
        files: &[T],
        index: &mut usize,
        lang: &Lang,
        analyze: Option<bool>,
        status: impl FnOnce(&T) -> Option<String>,
    ) -> bool {
        if files.is_empty() {
            return false;
        }
        let mut analyze_clicked = false;
        // Bound the row's height before aligning Sort at the right edge.
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.sort_menu(ui, files, index, lang);
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    let mut position = self
                        .order
                        .iter()
                        .position(|i| *i + 1 == *index)
                        .unwrap_or(0)
                        + 1;
                    nav_prev_next(
                        ui,
                        &mut position,
                        files.len(),
                        translate(Key::prev, lang),
                        translate(Key::next, lang),
                    );
                    *index = self.order[position - 1] + 1;
                    let file = &files[*index - 1];
                    let status = status(file).or_else(|| {
                        (!file.is_processed())
                            .then(|| translate(Key::not_analysed, lang).to_owned())
                    });
                    if let Some(status) = status {
                        ui.separator();
                        ui.label(egui::RichText::new(status).weak().small());
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
                    nav_filename(ui, name, position, files.len());
                });
            });
        });
        if let Some(position) = self.order.iter().position(|i| *i + 1 == *index) {
            let mut position = position + 1;
            nav_slider(ui, &mut position, self.order.len());
            *index = self.order[position - 1] + 1;
        }
        analyze_clicked
    }

    fn sort_menu<T: Pred>(
        &mut self,
        ui: &mut egui::Ui,
        files: &[T],
        index: &mut usize,
        lang: &Lang,
    ) {
        let (icon, key) = if self.descending {
            ("↓", Key::date_descending)
        } else {
            ("↑", Key::date_ascending)
        };
        // The row flows right to left, so draw the arrow before Sort.
        let mut changed = ui
            .button(icon)
            .on_hover_text(translate(key, lang))
            .clicked();
        if changed {
            self.descending = !self.descending;
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

    fn sort_by_date<T: Pred>(&mut self, files: &[T]) {
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

#[cfg(test)]
#[path = "../../tests/sorting/benchmark.rs"]
mod tests;
