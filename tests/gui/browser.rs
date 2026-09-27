use super::*;
use crate::{
    api::abstractions::{AIOutputs, PredAudio, PredImg, PredVideo},
    gui::Gui,
};
use std::{path::PathBuf, time::Duration};

struct Files(PathBuf);

impl Files {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "boquilahub-sort-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        for (name, nanos, color) in [
            ("later.png", 500_000_000, [255, 0, 0]),
            ("earlier.png", 100_000_000, [0, 0, 255]),
        ] {
            let path = dir.join(name);
            image::RgbImage::from_pixel(16, 16, image::Rgb(color))
                .save(&path)
                .unwrap();
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_times(
                    std::fs::FileTimes::new()
                        .set_modified(SystemTime::UNIX_EPOCH + Duration::new(1_700_000_000, nanos)),
                )
                .unwrap();
        }
        Self(dir)
    }

    fn predictions(&self) -> Vec<PredImg> {
        ["later.png", "earlier.png"]
            .map(|name| PredImg::new_simple(self.0.join(name)))
            .into()
    }
}

impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Default)]
struct Ui {
    context: egui::Context,
    labels: Vec<(String, egui::Pos2)>,
}

impl Ui {
    fn frame(&mut self, gui: &mut Gui, events: Vec<egui::Event>) {
        let mut output = self.context.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                egui::Panel::left("setup")
                    .default_size(240.0)
                    .show(ui, |ui| {
                        ui.label("Setup");
                    });
                egui::CentralPanel::default().show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        if gui.img_state.texture.is_none() {
                            gui.paint(ui, gui.image_texture_n - 1);
                        }
                        gui.ui_image(ui);
                    });
                });
            },
        );
        self.labels = output
            .shapes
            .iter()
            .filter_map(|shape| match &shape.shape {
                egui::epaint::Shape::Text(text) => Some((
                    text.galley.job.text.clone(),
                    text.pos + text.galley.size() / 2.0,
                )),
                _ => None,
            })
            .collect();
        let texture = gui.img_state.texture.as_ref().unwrap().id();
        let preview = output.shapes.iter().find_map(|shape| match &shape.shape {
            egui::epaint::Shape::Rect(rect)
                if rect
                    .brush
                    .as_ref()
                    .is_some_and(|brush| brush.fill_texture_id == texture) =>
            {
                Some(rect.rect)
            }
            egui::epaint::Shape::Mesh(mesh) if mesh.texture_id == texture => {
                Some(mesh.calc_bounds())
            }
            _ => None,
        });
        // egui requires consuming texture updates even in headless tests.
        output.textures_delta.clear();
        let preview = preview.expect("preview must be painted inside the scroll area");
        assert!(preview.is_finite() && preview.min.y > 20.0 && preview.max.y <= 600.0);
        let sort = self
            .labels
            .iter()
            .find(|(text, _)| text == "Sort")
            .unwrap()
            .1;
        assert!(
            sort.x > 700.0 && sort.x < 800.0 && sort.y < 40.0,
            "Sort must stay in a compact row at the right edge: {sort:?}"
        );
        let arrows: Vec<_> = self
            .labels
            .iter()
            .filter(|(text, _)| text == "↑" || text == "↓")
            .collect();
        assert_eq!(arrows.len(), 1);
        let arrow = arrows[0].1;
        assert!(arrow.x > sort.x && arrow.x < 800.0 && (arrow.y - sort.y).abs() < 1.0);
        assert!(!self.labels.iter().any(|(text, _)| text == "Sort by"));
    }

    fn click(&mut self, gui: &mut Gui, label: &str) {
        let pos = self
            .labels
            .iter()
            .find(|(text, _)| text == label)
            .unwrap_or_else(|| panic!("missing {label:?}: {:?}", self.labels))
            .1;
        for pressed in [true, false] {
            self.frame(
                gui,
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            );
        }
        self.frame(gui, Vec::new());
    }
}

#[test]
fn sorting_keeps_input_order_until_chosen_and_reloads_visible_previews() {
    let files = Files::new();
    let mut gui = Gui {
        selected_imgs: files.predictions(),
        image_texture_n: 1,
        lang: Lang::EN,
        ..Default::default()
    };
    let mut ui = Ui::default();
    ui.frame(&mut gui, Vec::new());
    assert_eq!(gui.image_browser.order, [0, 1]);
    ui.click(&mut gui, "⏭");
    assert_eq!(gui.image_texture_n, 2);
    ui.click(&mut gui, "↑");
    assert!(gui.image_browser.descending);
    assert_eq!(gui.image_browser.order, [0, 1]);
    assert_eq!(gui.image_texture_n, 2);
    ui.click(&mut gui, "↓");
    ui.click(&mut gui, "Sort");
    assert!(gui.image_browser.field.is_none());
    ui.click(&mut gui, "File modified");
    assert_eq!(gui.image_browser.order, [1, 0]);
    assert_eq!(gui.image_texture_n, 2);
    let earlier = gui.img_state.texture.as_ref().unwrap().id();
    ui.click(&mut gui, "↑");
    assert_eq!(gui.image_texture_n, 1);
    assert!(gui.image_browser.descending);
    let later = gui.img_state.texture.as_ref().unwrap().id();
    assert_ne!(later, earlier);
    ui.click(&mut gui, "↓");
    assert_eq!(gui.image_browser.order, [1, 0]);
    assert_eq!(gui.image_texture_n, 2);
    assert_ne!(gui.img_state.texture.as_ref().unwrap().id(), later);
    assert!(!gui.image_browser.descending);
    ui.click(&mut gui, "Sort");
    ui.click(&mut gui, "File created");
    assert!(gui.image_browser.field == Some(DateField::Created));
}

#[test]
fn sorting_preserves_precision_equal_dates_and_missing_dates_last() {
    let fixture = Files::new();
    let mut files = fixture.predictions();
    files.push(PredImg::new_simple(fixture.0.join("missing.png")));
    files.push(files[1].clone());
    let mut browser = Browser {
        field: Some(DateField::Modified),
        descending: false,
        ..Default::default()
    };
    browser.sort_by_date(&files);
    assert_eq!(browser.order, [1, 3, 0, 2]);
    browser.descending = true;
    browser.sort_by_date(&files);
    assert_eq!(browser.order, [0, 1, 3, 2]);
}

#[test]
fn prediction_sidecars_still_load_for_each_media_type() {
    let fixture = Files::new();
    let path = fixture.0.join("later.png");
    let mut image = PredImg::new_simple(path.clone());
    image.aioutput = Some(AIOutputs::Classification(Vec::new()));
    image.write_predictions().unwrap();
    assert!(PredImg::new_simple(path.clone()).is_processed());
    assert!(PredAudio::new_simple(path).is_processed());
    let path = fixture.0.join("cached.mp4");
    let mut video = PredVideo::new_simple(path.clone());
    video.hydrate(640, 480, 30.0, 10);
    video.record(3, AIOutputs::Classification(Vec::new()));
    video.write_predictions().unwrap();
    let loaded = PredVideo::new_simple(path);
    assert_eq!(loaded.processed_count(), 1);
    assert!(loaded.prediction_at(3).is_some());
    assert_eq!((loaded.width, loaded.height), (640, 480));
}
