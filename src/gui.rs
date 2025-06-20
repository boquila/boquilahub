use super::localization::*;
use crate::api::abstractions::*;
use crate::api::bq::get_bqs;
use crate::api::eps::LIST_EPS;
use crate::api::export::write_pred_img_to_file;
use crate::api::inference::*;
use crate::api::video_file::VideofileProcessor;
use crate::api::{self};
use api::import::*;
use egui::{ColorImage, TextureHandle, TextureOptions};
use image::{open, ImageBuffer, Rgba};
use rfd::FileDialog;
use std::fs::{self};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

pub struct Gui {
    // Large types first
    ais: Vec<AI>,
    selected_files: Vec<PredImg>,
    video_file_path: Option<PathBuf>,
    feed_url: Option<String>,
    image_processing_receiver: Option<tokio::sync::mpsc::UnboundedReceiver<(usize, Vec<XYXYc>)>>,
    feed_processing_receiver: Option<tokio::sync::mpsc::UnboundedReceiver<(usize, Vec<XYXYc>)>>,
    video_processing_receiver:
        Option<tokio::sync::mpsc::UnboundedReceiver<ImageBuffer<Rgba<u8>, Vec<u8>>>>,
    video_file_processor: Option<VideofileProcessor>,

    // Option<Instant> (likely 24 bytes: 8-byte discriminant + 16-byte Instant)
    done_time: Option<Instant>,

    // usize and Option<usize> fields grouped together (8 bytes each on 64-bit)
    ai_selected: Option<usize>,
    step_frame: Option<usize>,
    total_frames: Option<usize>,
    current_frame: Option<usize>,
    ep_selected: usize,
    image_texture_n: usize,

    // Enums (size depends on variants, but typically 1-8 bytes)
    lang: Lang,
    mode: Mode,

    // bool fields grouped together (1 byte each, but will be padded)
    is_done: bool,
    isapi_deployed: bool,
    should_continue: bool,
    save_img_from_strema: bool,
    error_ocurred: bool,
    is_analysis_complete: bool,
    show_export_dialog: bool,
    it_ran: bool,
    img_state: State,
    video_state: State,
    feed_state: State,
}

pub struct State {
    cancel_sender: Option<tokio::sync::oneshot::Sender<()>>,
    texture: Option<TextureHandle>,
    is_processing: bool,
    progress_bar: f32,
}

impl State {
    pub fn init() -> Self {
        State {
            cancel_sender: None,
            is_processing: false,
            progress_bar: 0.0,
            texture: None,
        }
    }
}

impl Gui {
    pub fn new() -> Self {
        let local_lang = {
            let locale = sys_locale::get_locale().unwrap_or_else(|| "en-US".to_owned());
            let lang_code = locale.get(0..2).unwrap_or("en").to_lowercase();
            match lang_code.as_str() {
                "en" => Lang::EN,
                "es" => Lang::ES,
                "fr" => Lang::FR,
                "de" => Lang::DE,
                "zh" => Lang::ZH,
                _ => Lang::EN,
            }
        };

        Self {
            ais: get_bqs(),
            selected_files: Vec::new(),
            video_file_path: None,
            feed_url: None,
            image_processing_receiver: None,
            feed_processing_receiver: None,
            video_processing_receiver: None,
            video_file_processor: None,
            ai_selected: None,
            ep_selected: 0,     // CPU is the default
            image_texture_n: 1, // this starts at 1
            step_frame: None,
            total_frames: None,
            current_frame: None,
            is_done: false,
            done_time: None,
            lang: local_lang,
            isapi_deployed: false,
            should_continue: true,
            save_img_from_strema: false,
            error_ocurred: false,
            is_analysis_complete: false,
            show_export_dialog: false,
            mode: Mode::Image,
            it_ran: false,
            img_state: State::init(),
            video_state: State::init(),
            feed_state: State::init(),
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
        self.img_state.texture = Some(imgpred_to_texture(&self.selected_files[i], ctx))
    }

    pub fn show_done_message(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
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

    pub fn api_widget(&mut self, ui: &mut egui::Ui) {
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
    }

    pub fn ai_widget(&mut self, ui: &mut egui::Ui, previous_ai: Option<usize>) {
        ui.label(self.t(Key::select_ai));
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

        if (self.ai_selected != previous_ai) && self.ai_selected.is_some() {
            set_model(
                &self.ais[self.ai_selected.unwrap()].get_path(),
                &LIST_EPS[self.ep_selected],
            );
        }
    }

    pub fn ep_widget(&mut self, ui: &mut egui::Ui, previous_ep: usize) {
        ui.label(self.t(Key::select_ep));
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

        if (self.ep_selected != previous_ep) && self.ai_selected.is_some() {
            set_model(
                &self.ais[self.ai_selected.unwrap()].get_path(),
                &LIST_EPS[self.ep_selected],
            );
        }
    }

    pub fn data_selection_widget(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
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
                                        self.img_state.progress_bar =
                                            self.selected_files.get_progress()
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
                            self.img_state.progress_bar = self.selected_files.get_progress()
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
                            self.video_file_processor = Some(VideofileProcessor::new(
                                &self.video_file_path.clone().unwrap().to_str().unwrap(),
                            ));
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
    }

    pub fn img_analysis_widget(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        if self.selected_files.len() >= 1 && self.ai_selected.is_some() {
            ui.vertical_centered(|ui| {
                ui.heading(format!("📋 {}", self.t(Key::analysis)));
            });
            ui.separator();

            // Analyze button Widget
            ui.vertical_centered(|ui| {
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::analyze)))
                    .clicked()
                    && self.image_processing_receiver.is_none()
                {
                    if !self.img_state.is_processing {
                        self.should_continue = true;
                        self.img_state.is_processing = true;

                        //  Async processing: Images
                        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                        self.image_processing_receiver = Some(rx);

                        let file_paths: Vec<String> = self
                            .selected_files
                            .iter()
                            .map(|f| f.file_path.to_str().unwrap().to_string())
                            .collect();
                        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
                        self.img_state.cancel_sender = Some(cancel_tx);

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
                }

                // Cancel button widget
                if self.img_state.is_processing {
                    if ui
                        .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::cancel)))
                        .clicked()
                    {
                        self.should_continue = false;
                        if let Some(cancel_tx) = self.img_state.cancel_sender.take() {
                            let _ = cancel_tx.send(());
                        }
                        self.img_state.is_processing = false;
                        self.image_processing_receiver = None;
                    }
                }
            });

            // Progress Bar: Image
            if self.selected_files.len() > 0 {
                ui.add(
                    egui::ProgressBar::new(self.img_state.progress_bar)
                        .show_percentage()
                        .animate(self.img_state.is_processing),
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
                        self.show_export_dialog = true;
                    }
                });
            }

            // Export dialog logic
            if self.show_export_dialog {
                egui::Window::new(self.t(Key::export))
                    .collapsible(false)
                    .resizable(false)
                    .show(ctx, |ui| {
                        // Export option 1
                        if ui.button(self.t(Key::export_predictions)).clicked() {
                            for file in self.selected_files.clone() {
                                tokio::spawn(async move {
                                    let _ = write_pred_img_to_file(&file).await;
                                });
                            }

                            self.process_done();
                            self.show_export_dialog = false;
                        }

                        // Export option 2
                        if ui
                            .button(self.t(Key::export_imgs_with_predictions))
                            .clicked()
                        {
                            for file in &self.selected_files {
                                file.save();
                            }
                            self.process_done();
                            self.show_export_dialog = false;
                        }

                        // Export option 3
                        if ui.button(self.t(Key::copy_with_classification)).clicked() {
                            let timestamp =
                                chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
                            tokio::spawn({
                                let selected_files = self.selected_files.clone(); // Make sure it's Send + 'static
                                let path = format!("export/export_{}", timestamp);
                                async move {
                                    let _ =
                                        api::export::copy_to_folder(&selected_files, &path).await;
                                }
                            });
                            self.process_done();
                            self.show_export_dialog = false;
                        }

                        // Cancel any export
                        if ui.button(self.t(Key::cancel)).clicked() {
                            self.show_export_dialog = false;
                        }
                    });
            }
        }
    }

    pub fn video_analysis_widget(&mut self, ui: &mut egui::Ui) {
        if self.video_file_path.is_some() {
            if ui.button("▶️").clicked() {
                if !self.video_state.is_processing {
                    // Async processing: Video
                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                    self.video_processing_receiver = Some(rx);
                    let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
                    self.video_state.cancel_sender = Some(cancel_tx);

                    let mut processor = self.video_file_processor.take().unwrap();
                    tokio::spawn(async move {
                        let n = processor.get_n_frames();
                        let (frame_tx, mut frame_rx) = tokio::sync::mpsc::unbounded_channel();

                        // Spawn blocking task to generate frames
                        let processor_handle = tokio::task::spawn_blocking(move || {
                            for _i in 0..=n {
                                let (img, _b) = processor.run(None).unwrap();
                                if frame_tx.send(img).is_err() {
                                    break; // Receiver dropped
                                }
                            }
                        });

                        // Stream frames as they're produced
                        while let Some(img) = frame_rx.recv().await {
                            if cancel_rx.try_recv().is_ok() {
                                break;
                            }

                            if tx.send(img).is_err() {
                                break;
                            }
                        }

                        // Clean up the blocking task
                        let _ = processor_handle.await;
                    });
                }
            }
        }
    }

    pub fn img_handle_results(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &mut self.image_processing_receiver {
            let mut updates = Vec::new();
            while let Ok((i, bbox)) = rx.try_recv() {
                updates.push((i, bbox));
            }

            for (i, bbox) in updates {
                self.selected_files[i].list_bbox = bbox;
                self.selected_files[i].wasprocessed = true;
                // if the img is the same that the user is seeing, we'll repaint it    
                if i == self.image_texture_n - 1 {
                    self.paint(ctx, i);
                }
            }

            if self.selected_files.iter().all(|f| f.wasprocessed) {
                self.img_state.is_processing = false;
                self.image_processing_receiver = None;
            }

            self.img_state.progress_bar = self.selected_files.get_progress();
            ctx.request_repaint();
        }
    }

    pub fn video_handle_results(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &mut self.video_processing_receiver {
            let mut updates = Vec::new();
            while let Ok(img) = rx.try_recv() {
                updates.push(img);
            }

            for img in updates {
                self.video_state.texture = Some(ctx.load_texture(
                    "current_frame",
                    imgbuf_to_colorimg(&img),
                    TextureOptions::default(),
                ));
            }
            ctx.request_repaint();
        }
    }
}

impl eframe::App for Gui {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.it_ran {
            egui_extras::install_image_loaders(ctx);
            self.it_ran = true;
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button(self.t(Key::about), |ui| {
                    ui.hyperlink_to(self.t(Key::website), self.t(Key::website_url));
                    ui.hyperlink_to(self.t(Key::donate), self.t(Key::donate_url));
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
                    ui.radio_value(&mut self.lang, Lang::FR, "Français");
                    ui.radio_value(&mut self.lang, Lang::DE, "Deutsch");
                    ui.radio_value(&mut self.lang, Lang::ZH, "简体中文");
                });

                egui::widgets::global_theme_preference_switch(ui);
            });
        });

        egui::SidePanel::left("left_panel").show(ctx, |ui| {
            let previous_ai = self.ai_selected;
            let previous_ep = self.ep_selected;

            ui.vertical_centered(|ui| {
                ui.heading(format!("💻 {}", self.t(Key::setup)));
            });
            ui.separator();

            self.ai_widget(ui, previous_ai);

            ui.add_space(8.0);

            self.ep_widget(ui, previous_ep);

            ui.add_space(8.0);

            self.api_widget(ui);

            ui.separator();

            self.data_selection_widget(ctx, ui);

            ui.separator();

            match self.mode {
                Mode::Image => {
                    self.img_analysis_widget(ctx, ui);
                }
                Mode::Video => {
                    self.video_analysis_widget(ui);
                }
                Mode::Feed => {
                    todo!()
                }
            }

            self.show_done_message(ui, ctx);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let cond1 = self.selected_files.len() >= 1;
            let cond2 = self.video_file_path.is_some();
            let cond3 = self.feed_url.is_some();
            // it has a mode AND another
            let img_mode = cond1 && (cond2 || cond3);
            let video_mode = cond2 && (cond1 || cond3);
            let feed_mode = cond3 && (cond1 || cond2);
            ui.horizontal(|ui| {
                if img_mode {
                    let text = self.t(Key::image_processing);
                    ui.selectable_value(&mut self.mode, Mode::Image, text);
                }
                if video_mode {
                    let text = self.t(Key::video_processing);
                    ui.selectable_value(&mut self.mode, Mode::Video, text);
                }
                if feed_mode {
                    let text = self.t(Key::feed_processing);
                    ui.selectable_value(&mut self.mode, Mode::Feed, text);
                }
            });

            if img_mode || video_mode || feed_mode {
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

                        if let Some(texture) = &self.img_state.texture {
                            ui.add(
                                egui::Image::new(texture)
                                    .max_height(800.0)
                                    .corner_radius(10.0),
                            );
                        }
                    });
                    self.img_handle_results(ctx);
                }
                Mode::Video => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        if let Some(texture) = &self.video_state.texture {
                            ui.add(
                                egui::Image::new(texture)
                                    .max_height(800.0)
                                    .corner_radius(10.0),
                            );
                        }
                    });
                    self.video_handle_results(ctx);
                }
                Mode::Feed => {
                    todo!()
                }
            }
        });
    }
}

fn imgbuf_to_colorimg(
    image_buffer: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
) -> ColorImage {
    let size: [usize; 2] = [image_buffer.width() as _, image_buffer.height() as _];
    ColorImage::from_rgba_unmultiplied(size, image_buffer.as_flat_samples().as_slice())
}

#[inline(always)]
fn imgpred_to_texture(predimg: &PredImg, ctx: &egui::Context) -> TextureHandle {
    ctx.load_texture("current_img", imgbuf_to_colorimg(&predimg.draw()), TextureOptions::default())
}

#[derive(PartialEq)]
enum Mode {
    Image,
    Video,
    Feed,
}
