use super::localization::*;
use crate::api;
use crate::api::abstractions::*;
use crate::api::bq::get_bqs;
use crate::api::eps::LIST_EPS;
use crate::api::inference::*;
use api::import::*;
use egui::{ColorImage, TextureHandle, TextureOptions};
use ffmpeg_next::codec::video;
use image::open;
use rfd::FileDialog;
use std::fs::{self};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

pub struct MainApp {
    // Large types first (Vec, Option<PathBuf>, Option<String>)
    ais: Vec<AI>,
    selected_files: Vec<PredImg>,
    video_file_path: Option<PathBuf>,
    feed_url: Option<String>,
    processing_receiver: Option<tokio::sync::mpsc::UnboundedReceiver<(usize, Vec<XYXYc>)>>,
    cancel_sender: Option<tokio::sync::oneshot::Sender<()>>,

    // Medium-sized types (TextureHandle options)
    screen_texture: Option<TextureHandle>,
    video_frame: Option<TextureHandle>,
    feed_frame: Option<TextureHandle>,

    // usize fields (8 bytes on 64-bit)
    ai_selected: Option<usize>,
    ep_selected: usize,
    image_texture_n: usize,

    // Option<usize> fields (likely 16 bytes due to Option overhead)
    step_frame: Option<usize>,
    total_frames: Option<usize>,
    current_frame: Option<usize>,

    is_done: bool,
    done_time: Option<Instant>,

    progress_bar: f32,

    // Enums
    lang: Lang,

    // Media State
    isapi_deployed: bool,
    is_processing: bool,
    should_continue: bool,
    save_img_from_strema: bool,
    error_ocurred: bool,
    is_analysis_complete: bool,

    // what it can do
    img_mode: bool,
    video_mode: bool,
    feed_mode: bool,

    // the active mode
    mode: Mode,
}

impl MainApp {
    pub fn new() -> Self {
        // set_model("models/boquilanet-gen.bq".to_owned(), LIST_EPS[1].clone());
        Self {
            ais: get_bqs(),
            selected_files: Vec::new(),
            video_file_path: None,
            feed_url: None,
            processing_receiver: None,
            cancel_sender: None,
            screen_texture: None,
            video_frame: None,
            feed_frame: None,
            ai_selected: None,
            ep_selected: 0,     // CPU is the default
            image_texture_n: 1, // this starts at 1
            step_frame: None,
            total_frames: None,
            current_frame: None,
            is_done: false,
            done_time: None,
            progress_bar: 0.0,
            lang: Lang::EN,
            isapi_deployed: false,
            is_processing: false,
            should_continue: true,
            save_img_from_strema: false,
            error_ocurred: false,
            is_analysis_complete: false,
            img_mode: false,
            video_mode: false,
            feed_mode: false,
            mode: Mode::Image,
        }
    }

    pub fn process_done(&mut self) {
        self.is_done = true;
        self.done_time = Some(Instant::now());
    }

    pub fn t(&self, key: Key) -> &'static str {
        translate(key, &self.lang)
    }

    pub fn paint(&mut self, ctx: &egui::Context, i: usize) {
        self.screen_texture = Some(imgpred_to_texture(&self.selected_files[i], ctx))
    }

    pub fn any_mode(&self) -> bool {
        self.img_mode || self.video_mode || self.feed_mode
    }
}

impl eframe::App for MainApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui
        egui_extras::install_image_loaders(ctx);

        let cond1 = self.selected_files.len() >= 1;
        let cond2 = self.video_file_path.is_some();
        let cond3 = self.feed_url.is_some();
        self.img_mode = cond1 && (cond2 || cond3);
        self.video_mode = cond2 && (cond1 || cond3);
        self.feed_mode = cond3 && (cond1 || cond2);

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                ui.menu_button(self.t(Key::about), |ui| {
                    ui.hyperlink_to(self.t(Key::website), self.t(Key::website_url));
                    ui.hyperlink_to(self.t(Key::donate), self.t(Key::donate_url));
                    ui.hyperlink_to(self.t(Key::model_hub), self.t(Key::model_hub_url));
                    ui.hyperlink_to(
                        self.t(Key::source_code),
                        "https://github.com/boquila/boquilahub/",
                    );
                });

                ui.menu_button(self.t(Key::models), |ui| {
                    ui.hyperlink_to(self.t(Key::model_hub), self.t(Key::model_hub_url));
                });

                ui.menu_button(self.t(Key::idiom), |ui| {
                    ui.radio_value(&mut self.lang, Lang::EN, "English");
                    ui.radio_value(&mut self.lang, Lang::ES, "Español");
                });

                egui::widgets::global_theme_preference_switch(ui);
            });
        });

        egui::SidePanel::left("left_panel").show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading(format!("💻 {}", self.t(Key::setup)));
            });
            ui.separator();

            ui.label(self.t(Key::select_ai));

            let previous_ai = self.ai_selected;
            // AI Selection Widget
            egui::ComboBox::from_id_salt("AI")
                .selected_text(match self.ai_selected {
                    Some(i) => &self.ais[i].name,
                    None => "",
                })
                .show_ui(ui, |ui| {
                    for (i, ai) in self.ais.iter().enumerate() {
                        ui.selectable_value(&mut self.ai_selected, Some(i), &ai.name)
                            .on_hover_text(&ai.classes.join(", "));
                    }
                });

            if self.ai_selected != previous_ai {
                if self.ai_selected.is_some() {
                    set_model(
                        &self.ais[self.ai_selected.unwrap()].get_path(),
                        &LIST_EPS[self.ep_selected],
                    );
                }
            }

            ui.add_space(8.0);

            // EP Selection Widget
            ui.label(self.t(Key::select_ep));

            let previous_ep = self.ep_selected;
            egui::ComboBox::from_id_salt("EP")
                .selected_text(LIST_EPS[self.ep_selected].name)
                .show_ui(ui, |ui| {
                    for (i, ep) in LIST_EPS.iter().enumerate() {
                        ui.selectable_value(&mut self.ep_selected, i, ep.name)
                            .on_hover_text(format!(
                                "Version: {:.1}, Local: {}, Dependencies: {}",
                                ep.version, ep.local, ep.dependencies
                            ));
                    }
                });

            if self.ep_selected != previous_ep {
                set_model(
                    &self.ais[self.ai_selected.unwrap()].get_path(),
                    &LIST_EPS[self.ep_selected],
                );
            }

            ui.add_space(8.0);
            ui.label("API ");

            if !self.isapi_deployed {
                if ui.button(self.t(Key::deploy)).clicked() {
                    tokio::spawn(async {
                        thread::sleep(Duration::from_secs(2));
                    });
                    self.isapi_deployed = true;
                }
            }

            if self.isapi_deployed {
                ui.label(self.t(Key::deployed_api));
            }

            ui.separator();

            ui.vertical_centered(|ui| {
                ui.heading(format!("📎 {}", self.t(Key::select_your_data)));
            });
            ui.separator();

            ui.spacing_mut().button_padding = egui::vec2(12.0, 8.0);
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);

            // Data Selection Widget
            egui::Grid::new("file_selection_grid")
                .num_columns(2)
                .spacing([10.0, 10.0])
                .show(ui, |ui| {
                    // FOLDER SELECTION SECTION
                    if ui
                        .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::folder)))
                        .clicked()
                    {
                        match FileDialog::new().pick_folder() {
                            Some(folder_path) => {
                                // Read directory contents and filter for image files
                                match fs::read_dir(&folder_path) {
                                    Ok(entries) => {
                                        let mut image_files = Vec::new();

                                        for entry in entries {
                                            if let Ok(entry) = entry {
                                                let path = entry.path();

                                                // Only process files (not subdirectories)
                                                if path.is_file() {
                                                    // Check if file extension matches IMAGE_FORMATS
                                                    if let Some(extension) = path.extension() {
                                                        if let Some(ext_str) = extension.to_str() {
                                                            if IMAGE_FORMATS.iter().any(|&format| {
                                                                ext_str.to_lowercase()
                                                                    == format.to_lowercase()
                                                            }) {
                                                                image_files.push(path);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        if !image_files.is_empty() {
                                            // Set the first image as the screen texture
                                            self.selected_files = image_files
                                                .into_iter()
                                                .map(|path| PredImg::new_simple(path))
                                                .collect();

                                            self.paint(ctx, 0);
                                            self.mode = Mode::Image;
                                        }
                                    }
                                    Err(_e) => {
                                        self.error_ocurred = true;
                                    }
                                }
                            }
                            None => (), // No folder selected
                        }
                    }

                    // IMAGE FILE SELECTION SECTION
                    if ui
                        .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::image)))
                        .clicked()
                    {
                        match FileDialog::new()
                            .add_filter("Image", &IMAGE_FORMATS)
                            .pick_files()
                        {
                            Some(paths) => {
                                self.selected_files = paths
                                    .into_iter()
                                    .map(|path| PredImg::new_simple(path))
                                    .collect();
                                self.paint(ctx, 0);
                                self.mode = Mode::Image;
                            }
                            _ => (), // no selection, do nothing
                        }
                    }
                    ui.end_row();

                    // VIDEO FILE SELECTION SECTION
                    if ui
                        .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::video_file)))
                        .clicked()
                    {
                        match FileDialog::new()
                            .add_filter("Video", &VIDEO_FORMATS)
                            .pick_file()
                        {
                            Some(path) => {
                                self.video_file_path = Some(path);
                                self.mode = Mode::Video;
                            }
                            _ => (), // no selection, do nothing
                        }
                    }

                    // Camera feed
                    if ui
                        .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::camera_feed)))
                        .clicked()
                    {
                        self.feed_url = Some("test".to_owned());
                        self.mode = Mode::Feed;
                    }
                });

            if self.selected_files.len() >= 1 && self.ai_selected.is_some() {
                ui.separator();
                ui.vertical_centered(|ui| {
                    ui.heading(format!("📋 {}", self.t(Key::analysis)));
                });
                ui.separator();

                // Analyze button Widget
                ui.vertical_centered(|ui| {
                    if ui
                        .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::analyze)))
                        .clicked()
                        && self.processing_receiver.is_none()
                    {
                        self.should_continue = true;
                        self.is_processing = true;

                        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                        self.processing_receiver = Some(rx);

                        let file_paths: Vec<_> = self
                            .selected_files
                            .iter()
                            .map(|f| f.file_path.to_str().unwrap().to_string())
                            .collect();
                        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
                        self.cancel_sender = Some(cancel_tx);

                        tokio::spawn(async move {
                            for (i, path) in file_paths.iter().enumerate() {
                                // CHECK FOR CANCELLATION HERE
                                if cancel_rx.try_recv().is_ok() {
                                    break;
                                }

                                let img = open(path).unwrap().into_rgb8();
                                let bbox = tokio::task::spawn_blocking(move || {
                                    detect_bbox_from_imgbuf(&img)
                                })
                                .await
                                .unwrap();
                                if tx.send((i, bbox)).is_err() {
                                    break;
                                }
                            }
                        });
                    }

                    if self.is_processing {
                        if ui
                            .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::cancel)))
                            .clicked()
                        {
                            self.should_continue = false;
                            if let Some(cancel_tx) = self.cancel_sender.take() {
                                let _ = cancel_tx.send(());
                            }
                            self.is_processing = false;
                            self.processing_receiver = None;
                        }
                    }
                });

                // Handle results
                if let Some(rx) = &mut self.processing_receiver {
                    let mut updates = Vec::new();
                    while let Ok((i, bbox)) = rx.try_recv() {
                        updates.push((i, bbox));
                    }

                    for (i, bbox) in updates {
                        self.selected_files[i].list_bbox = bbox;
                        self.selected_files[i].wasprocessed = true;
                        if i == self.image_texture_n - 1 {
                            self.paint(ctx, i);
                        }
                    }

                    if self.selected_files.iter().all(|f| f.wasprocessed) {
                        self.is_processing = false;
                        self.processing_receiver = None;
                    }
                    self.progress_bar = self.selected_files.get_progress();
                    ctx.request_repaint();
                }

                if self.selected_files.len() > 0 {
                    ui.add(
                        egui::ProgressBar::new(self.progress_bar)
                            .show_percentage()
                            .animate(self.is_processing),
                    );
                }

                ui.add_space(8.0);

                // Export button Widget
                if self.mode == Mode::Image {
                    ui.vertical_centered(|ui| {
                        if ui
                            .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::export)))
                            .clicked()
                        {
                            // EXPORT logic
                            self.process_done();
                        }
                    });
                }

                if let Some(done_time) = self.done_time {
                    if done_time.elapsed().as_secs_f32() < 3.0 {
                        ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                            ui.label(self.t(Key::done));
                        });
                        ctx.request_repaint();
                    } else {
                        self.done_time = None;
                    }
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.img_mode {
                    let text = self.t(Key::image_processing);
                    ui.selectable_value(&mut self.mode, Mode::Image, text);
                }
                if self.video_mode {
                    let text = self.t(Key::video_processing);
                    ui.selectable_value(&mut self.mode, Mode::Video, text);
                }
                if self.feed_mode {
                    let text = self.t(Key::feed_processing);
                    ui.selectable_value(&mut self.mode, Mode::Feed, text);
                }
            });

            if self.any_mode() {
                ui.separator();
            }
            match self.mode {
                Mode::Image => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.style_mut().spacing.slider_width = 300.0;
                        if self.selected_files.len() > 1 {
                            let response = ui.add(
                                egui::Slider::new(
                                    &mut self.image_texture_n,
                                    1..=self.selected_files.len(),
                                )
                                .text(""),
                            );
                            if response.changed() {
                                self.paint(ctx, self.image_texture_n - 1);
                            }
                        }

                        // If any textuure has been defined, we render it
                        match &self.screen_texture {
                            Some(texture) => {
                                ui.add(
                                    egui::Image::new(texture)
                                        .max_height(800.0)
                                        .corner_radius(10.0),
                                );
                            }
                            None => {}
                        }
                    });
                }
                Mode::Video => {
                    todo!()
                }
                Mode::Feed => {
                    todo!()
                }
            }
        });
    }
}

fn load_image_from_buffer_ref(
    image_buffer: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
) -> ColorImage {
    let size: [usize; 2] = [image_buffer.width() as _, image_buffer.height() as _];
    let pixels: image::FlatSamples<&[u8]> = image_buffer.as_flat_samples();
    ColorImage::from_rgba_unmultiplied(size, pixels.as_slice())
}

fn imgpred_to_texture(predimg: &PredImg, ctx: &egui::Context) -> TextureHandle {
    let image_data = if predimg.wasprocessed {
        load_image_from_buffer_ref(&predimg.draw2())
    } else {
        load_image_from_buffer_ref(&open(predimg.file_path.clone()).unwrap().into_rgba8())
    };

    ctx.load_texture("current_img", image_data, TextureOptions::default())
}

#[derive(PartialEq)]
enum Mode {
    Image,
    Video,
    Feed,
}
