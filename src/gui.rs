use super::localization::*;
use crate::api::abstractions::*;
use crate::api::bq::get_bqs;
use crate::api::eps::LIST_EPS;
use crate::api::export::write_pred_img_to_file;
use crate::api::inference::*;
use crate::api::models::processing::inference::AIOutputs;
use crate::api::render::draw_aioutput;
use crate::api::rest::{
    check_boquila_hub_api, detect_bbox_from_buf_remotely, get_ipv4_address,
    rgba_image_to_jpeg_buffer, run_api,
};
use crate::api::stream::Feed;
use crate::api::video_file::VideofileProcessor;
use crate::api::{self};
use api::import::*;
use egui::{ColorImage, TextureHandle, TextureOptions};
use image::{open, ImageBuffer, Rgba};
use rfd::FileDialog;
use std::fs::{self};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub fn run_gui() {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                // NOTE: Adding an icon is optional
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "BoquilaHUB",
        native_options,
        Box::new(|_cc| Ok(Box::new(Gui::new()))),
    );
}

pub struct Gui {
    // Large types first
    ais: Vec<AI>,
    selected_files: Vec<PredImg>,
    video_file_path: Option<PathBuf>,
    feed_url: Option<String>,
    host_server_url: Option<String>,
    api_server_url: Option<String>,
    temp_str: String,
    temp_api_str: String,
    api_result_receiver: Option<std::sync::mpsc::Receiver<bool>>,
    image_processing_receiver: Option<tokio::sync::mpsc::UnboundedReceiver<(usize, AIOutputs)>>,
    feed_processing_receiver:
        Option<tokio::sync::mpsc::UnboundedReceiver<(AIOutputs, ImageBuffer<Rgba<u8>, Vec<u8>>)>>,
    video_processing_receiver:
        Option<tokio::sync::mpsc::UnboundedReceiver<(u64, ImageBuffer<Rgba<u8>, Vec<u8>>)>>,
    video_file_processor: Arc<Mutex<Option<VideofileProcessor>>>,

    // Option<Instant> (likely 24 bytes: 8-byte discriminant + 16-byte Instant)
    done_time: Option<Instant>,
    error_time: Option<Instant>,

    // usize and Option<usize> fields grouped together (8 bytes each on 64-bit)
    ai_selected: Option<usize>,
    step_frame: Option<usize>,
    current_frame: u64,
    total_frames: Option<u64>,
    ep_selected: usize,
    image_texture_n: usize,

    // Enums (size depends on variants, but typically 1-8 bytes)
    lang: Lang,
    mode: Mode,

    // bool fields grouped together (1 byte each, but will be padded)
    is_done: bool,
    isapi_deployed: bool,
    save_img_from_strema: bool,
    error_ocurred: bool,
    show_export_dialog: bool,
    show_feed_url_dialog: bool,
    show_api_server_dialog: bool,
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
        Self {
            ais: get_bqs(),
            selected_files: Vec::new(),
            video_file_path: None,
            feed_url: None,
            host_server_url: None,
            api_server_url: None,
            api_result_receiver: None,
            temp_str: "".to_owned(),
            temp_api_str: "".to_owned(),
            image_processing_receiver: None,
            feed_processing_receiver: None,
            video_processing_receiver: None,
            video_file_processor: Arc::new(Mutex::new(None)),
            ai_selected: None,
            ep_selected: 0,     // CPU is the default
            image_texture_n: 1, // this starts at 1
            step_frame: None,
            current_frame: 0,
            total_frames: None,
            is_done: false,
            done_time: None,
            error_time: None,
            lang: get_locale(),
            isapi_deployed: false,
            save_img_from_strema: false,
            error_ocurred: false,
            show_export_dialog: false,
            show_feed_url_dialog: false,
            show_api_server_dialog: false,
            mode: Mode::Image,
            it_ran: false,
            img_state: State::init(),
            video_state: State::init(),
            feed_state: State::init(),
        }
    }

    pub fn is_any_processing(&self) -> bool {
        self.video_state.is_processing && self.img_state.is_processing & self.is_any_processing()
    }

    pub fn is_remote(&self) -> bool {
        self.ep_selected == 2
    }

    pub fn process_done(&mut self) {
        self.is_done = true;
        self.done_time = Some(Instant::now());
    }

    pub fn process_error(&mut self) {
        self.error_ocurred = true;
        self.error_time = Some(Instant::now());
    }

    pub fn t(&self, key: Key) -> &'static str {
        translate(key, &self.lang)
    }

    pub fn paint(&mut self, ctx: &egui::Context, i: usize) {
        self.img_state.texture = imgbuf_to_texture(&self.selected_files[i].draw(), ctx)
    }

    fn show_timed_message(
        time: &mut Option<std::time::Instant>,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        message: &str,
    ) {
        if let Some(start_time) = *time {
            if start_time.elapsed().as_secs_f32() < 3.0 {
                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                    ui.label(message);
                });
                ctx.request_repaint();
            } else {
                *time = None;
            }
        }
    }

    pub fn show_done_message(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let message = &self.t(Key::done);
        let time = &mut self.done_time;
        Gui::show_timed_message(time, ui, ctx, message);
    }

    pub fn show_error_message(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let message = &self.t(Key::error_ocurred);
        let time = &mut self.error_time;
        Gui::show_timed_message(time, ui, ctx, message);
    }

    pub fn api_widget(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.ai_selected.is_some() && !self.is_remote() {
            ui.label(self.t(Key::api));
            if !self.isapi_deployed {
                if ui.button(self.t(Key::deploy)).clicked() {
                    let (tx, rx) = std::sync::mpsc::channel();
                    tokio::spawn(async move {
                        let result = run_api(8791).await;
                        let _ = tx.send(result.is_ok());
                    });

                    self.api_result_receiver = Some(rx);
                    self.host_server_url =
                        Some(format!("http://{}:8791", get_ipv4_address().unwrap()));
                    self.isapi_deployed = true;
                }
            }

            if let Some(url) = &self.host_server_url {
                ui.label(url);
            }

            if let Some(rx) = &self.api_result_receiver {
                if let Ok(success) = rx.try_recv() {
                    if !success {
                        self.isapi_deployed = false;
                        self.host_server_url = None;
                        self.process_error();
                    }
                    self.api_result_receiver = None;
                }
            }

            self.input_api_url_dialog(ctx);
        }
    }

    pub fn ai_widget(&mut self, ui: &mut egui::Ui) {
        if self.ep_selected != 2 {
            let previous_ai = self.ai_selected;
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

            if (self.ai_selected != previous_ai) && (self.ai_selected.is_some()) {
                if !self.is_remote() {
                    set_model(
                        &self.ais[self.ai_selected.unwrap()].get_path(),
                        &LIST_EPS[self.ep_selected],
                    );
                }
            }
        }
    }

    pub fn ep_widget(&mut self, ui: &mut egui::Ui) {
        ui.label(self.t(Key::select_ep));
        let mut temp_ep_selected = self.ep_selected;

        egui::ComboBox::from_id_salt("EP")
            .selected_text(LIST_EPS[self.ep_selected].name)
            .show_ui(ui, |ui| {
                for (i, ep) in LIST_EPS.iter().enumerate() {
                    ui.selectable_value(&mut temp_ep_selected, i, ep.name)
                        .on_hover_text(format!(
                            "Version: {:.1}, Local: {}, Dependencies: {}",
                            ep.version, ep.local, ep.dependencies
                        ));
                }
            });

        if temp_ep_selected != self.ep_selected {
            let new_ep: &api::eps::EP = &LIST_EPS[temp_ep_selected];

            match new_ep.ep_type {
                api::eps::EPType::BoquilaHUBRemote => {
                    self.show_api_server_dialog = true;
                }
                _ => {
                    self.ep_selected = temp_ep_selected;

                    if let Some(ai_index) = self.ai_selected {
                        set_model(&self.ais[ai_index].get_path(), &LIST_EPS[self.ep_selected]);
                    }
                }
            }
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
                                    let image_files: Vec<PathBuf> = entries
                                        .filter_map(|entry| entry.ok())
                                        .map(|entry| entry.path())
                                        .filter(|path| path.is_file())
                                        .filter(|path| {
                                            path.extension()
                                                .and_then(|ext| ext.to_str())
                                                .map(|ext_str| {
                                                    IMAGE_FORMATS.iter().any(|&format| {
                                                        ext_str.eq_ignore_ascii_case(format)
                                                    })
                                                })
                                                .unwrap_or(false)
                                        })
                                        .collect();

                                    if !image_files.is_empty() {
                                        self.selected_files = image_files
                                            .into_iter()
                                            .map(|path: PathBuf| PredImg::new_simple(path))
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
                            let processor = Some(VideofileProcessor::new(
                                &self.video_file_path.clone().unwrap().to_str().unwrap(),
                            ))
                            .unwrap();
                            self.total_frames = Some(processor.get_n_frames());
                            self.video_file_processor = Arc::new(Mutex::new(Some(processor)));
                            self.mode = Mode::Video;
                            self.current_frame = 0;
                        }
                        _ => (), // no selection, do nothing
                    }
                }

                // Camera feed
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::camera_feed)))
                    .clicked()
                {
                    self.show_feed_url_dialog = true
                }

                // Feed url dialog
                if self.show_feed_url_dialog {
                    egui::Window::new(self.t(Key::input_url))
                        .collapsible(false)
                        .resizable(false)
                        .show(ctx, |ui| {
                            ui.text_edit_singleline(&mut self.temp_str);
                            ui.horizontal(|ui| {
                                if ui.button(self.t(Key::ok)).clicked() {
                                    let url = self.temp_str.clone();
                                    match Feed::new(&url) {
                                        Ok(mut feed) => match feed.next() {
                                            Some(frame) => {
                                                self.feed_state.texture = imgbuf_to_texture(
                                                    &image::DynamicImage::ImageRgb8(frame)
                                                        .to_rgba8(),
                                                    ctx,
                                                );
                                                self.feed_url = Some(url);
                                                self.mode = Mode::Feed;
                                                if self.video_state.is_processing {
                                                    self.cancel_video_processing();
                                                }
                                            }
                                            None => {
                                                self.process_error();
                                            }
                                        },
                                        Err(_e) => {
                                            self.process_error();
                                        }
                                    }
                                    self.show_feed_url_dialog = false;
                                }
                                ui.add_space(8.0);
                                if ui.button(self.t(Key::cancel)).clicked() {
                                    self.show_feed_url_dialog = false;
                                    self.feed_url = None
                                }
                            });
                        });
                }
            });
    }

    pub fn input_api_url_dialog(&mut self, ctx: &egui::Context) {
        if self.show_api_server_dialog {
            egui::Window::new(self.t(Key::input_url))
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.text_edit_singleline(&mut self.temp_api_str);
                    ui.horizontal(|ui| {
                        if ui.button(self.t(Key::ok)).clicked() {
                            let url = self.temp_api_str.clone();

                            // This tells tokio to move this blocking operation to another thread
                            let is_valid_api = tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current()
                                    .block_on(check_boquila_hub_api(&url))
                            });

                            if is_valid_api {
                                self.show_api_server_dialog = false;
                                self.api_server_url = Some(url);
                                self.ep_selected = 2;
                            }
                        }
                        ui.add_space(8.0);
                        if ui.button(self.t(Key::cancel)).clicked() {
                            self.show_api_server_dialog = false;
                            self.api_server_url = None;
                        }
                    });
                });
        }
    }

    pub fn img_analysis_widget(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        if self.selected_files.len() >= 1 && (self.ai_selected.is_some() || self.is_remote()) {
            ui.vertical_centered(|ui| {
                ui.heading(self.t(Key::image));
                ui.heading(self.t(Key::analysis));
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

                        let api_endpoint = if self.is_remote() {
                            self.api_server_url
                                .as_ref()
                                .map(|url| format!("{}/upload", url))
                        } else {
                            None
                        };
                        let is_remote = self.is_remote();

                        tokio::spawn(async move {
                            for (i, path) in file_paths.iter().enumerate() {
                                // CHECK FOR CANCELLATION HERE
                                if cancel_rx.try_recv().is_ok() {
                                    break;
                                }
                                let bbox = if is_remote {
                                    let buffer = fs::read(path).unwrap();
                                    detect_bbox_from_buf_remotely(
                                        api_endpoint.as_ref().unwrap(),
                                        buffer,
                                    )
                                } else {
                                    let img = open(path).unwrap().into_rgb8();
                                    tokio::task::spawn_blocking(move || {
                                        detect_bbox_from_imgbuf(&img)
                                    })
                                    .await
                                    .unwrap()
                                };

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
                                if file.wasprocessed && !file.aioutput.as_ref().unwrap().is_empty() {
                                    file.save();
                                }
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
        if self.video_file_path.is_some() && (self.ai_selected.is_some() || self.is_remote()) {
            ui.vertical_centered(|ui| {
                ui.heading(self.t(Key::video_file));
                ui.heading(self.t(Key::analysis));
            });
            ui.separator();

            ui.vertical_centered(|ui| {
                if !self.video_state.is_processing {
                    if ui.button("▶").clicked() {
                        self.video_state.is_processing = true;
                        // Async processing: Video
                        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                        self.video_processing_receiver = Some(rx);
                        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
                        self.video_state.cancel_sender = Some(cancel_tx);

                        let processor = Arc::clone(&self.video_file_processor);
                        let n = self.total_frames.unwrap();
                        let current = self.current_frame;

                        let api_endpoint = if self.is_remote() {
                            self.api_server_url
                                .as_ref()
                                .map(|url| format!("{}/upload", url))
                        } else {
                            None
                        };
                        let is_remote = self.is_remote();
                        tokio::spawn(async move {
                            // Spawn blocking task to generate frames
                            let processor_handle = tokio::task::spawn_blocking(move || {
                                for i in current..=n {
                                    if cancel_rx.try_recv().is_ok() {
                                        break;
                                    }

                                    let (time, mut img) =
                                        processor.lock().unwrap().as_mut().unwrap().next().unwrap();

                                    let bbox = if is_remote {
                                        let buffer = rgba_image_to_jpeg_buffer(
                                            &image::DynamicImage::ImageRgb8(img.clone()).to_rgba8(),
                                            95,
                                        );
                                        detect_bbox_from_buf_remotely(
                                            api_endpoint.as_ref().unwrap(),
                                            buffer,
                                        )
                                    } else {
                                        detect_bbox_from_imgbuf(&img)
                                    };

                                    draw_aioutput(&mut img, &bbox);

                                    processor
                                        .lock()
                                        .unwrap()
                                        .as_mut()
                                        .unwrap()
                                        .decode(&img, time);
                                    let img = image::DynamicImage::ImageRgb8(img).to_rgba8();
                                    if tx.send((i, img)).is_err() {
                                        break;
                                    }
                                }
                            });

                            // Clean up the blocking task
                            let _ = processor_handle.await;
                        });
                    }
                } else {
                    if ui.button("||").clicked() {
                        self.cancel_video_processing();
                    }
                }
            });

            // Progress Bar: Video
            if self.video_file_path.is_some() {
                ui.add(
                    egui::ProgressBar::new(self.video_state.progress_bar)
                        .show_percentage()
                        .animate(self.video_state.is_processing),
                );
            }
        }
    }

    pub fn cancel_video_processing(&mut self) {
        if let Some(cancel_tx) = self.video_state.cancel_sender.take() {
            let _ = cancel_tx.send(());
        }
        self.video_state.is_processing = false;
    }

    pub fn feed_analysis_widget(&mut self, ui: &mut egui::Ui) {
        if self.feed_url.is_some() && (self.ai_selected.is_some() || self.is_remote()) {
            ui.vertical_centered(|ui| {
                ui.heading(self.t(Key::camera_feed));
                ui.heading(self.t(Key::analysis));
            });
            ui.separator();

            ui.vertical_centered(|ui| {
                if !self.feed_state.is_processing {
                    if ui.button("▶").clicked() {
                        self.feed_state.is_processing = true;
                        // Async processing: Video
                        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                        self.feed_processing_receiver = Some(rx);
                        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
                        self.feed_state.cancel_sender = Some(cancel_tx);

                        let url = self.feed_url.clone();
                        let mut feed = Feed::new(&url.unwrap()).unwrap();

                        let api_endpoint = if self.is_remote() {
                            self.api_server_url
                                .as_ref()
                                .map(|url| format!("{}/upload", url))
                        } else {
                            None
                        };
                        let is_remote = self.is_remote();
                        tokio::spawn(async move {
                            // Spawn blocking task to generate frames
                            let processor_handle = tokio::task::spawn_blocking(move || loop {
                                if cancel_rx.try_recv().is_ok() {
                                    break;
                                }
                                if let Some(mut img) = feed.next() {
                                    let bbox = if is_remote {
                                        let buffer = rgba_image_to_jpeg_buffer(
                                            &image::DynamicImage::ImageRgb8(img.clone()).to_rgba8(),
                                            95,
                                        );
                                        detect_bbox_from_buf_remotely(
                                            api_endpoint.as_ref().unwrap(),
                                            buffer,
                                        )
                                    } else {
                                        detect_bbox_from_imgbuf(&img)
                                    };

                                    api::render::draw_aioutput(&mut img, &bbox);
                                    let img = image::DynamicImage::ImageRgb8(img).to_rgba8();
                                    if tx.send((bbox, img)).is_err() {
                                        break;
                                    }
                                }
                            });

                            // Clean up the blocking task
                            let _ = processor_handle.await;
                        });
                    }
                } else {
                    if ui.button("||").clicked() {
                        if let Some(cancel_tx) = self.feed_state.cancel_sender.take() {
                            let _ = cancel_tx.send(());
                        }
                        self.feed_state.is_processing = false;
                    }
                }
            });
        }
    }

    pub fn img_handle_results(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &mut self.image_processing_receiver {
            let mut updates = Vec::new();
            while let Ok((i, bbox)) = rx.try_recv() {
                updates.push((i, bbox));
            }

            for (i, bbox) in updates {
                self.selected_files[i].aioutput = Some(bbox);
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

            for (i, img) in updates {
                self.video_state.texture = imgbuf_to_texture(&img, ctx);

                self.video_state.progress_bar = (i + 1) as f32 / self.total_frames.unwrap() as f32;
                self.current_frame = i;
                if i == self.total_frames.unwrap() {
                    self.video_state.is_processing = false;
                    self.video_processing_receiver = None;
                }
            }

            ctx.request_repaint();
        }
    }

    pub fn feed_handle_results(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &mut self.feed_processing_receiver {
            let mut updates = Vec::new();
            while let Ok(img) = rx.try_recv() {
                updates.push(img);
            }

            for (bbox, img) in updates {
                self.feed_state.texture = imgbuf_to_texture(&img, ctx)
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
            ui.vertical_centered(|ui| {
                ui.heading(format!("💻 {}", self.t(Key::setup)));
            });
            ui.separator();

            self.ai_widget(ui);

            ui.add_space(8.0);

            self.ep_widget(ui);

            ui.add_space(8.0);

            self.api_widget(ui, ctx);

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
                    self.feed_analysis_widget(ui);
                }
            }

            self.show_done_message(ui, ctx);
            self.show_error_message(ui, ctx);
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
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        if let Some(texture) = &self.feed_state.texture {
                            ui.add(
                                egui::Image::new(texture)
                                    .max_height(800.0)
                                    .corner_radius(10.0),
                            );
                        }
                    });
                    self.feed_handle_results(ctx);
                }
            }
        });
    }
}

fn imgbuf_to_colorimg(image_buffer: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>) -> ColorImage {
    let size: [usize; 2] = [image_buffer.width() as _, image_buffer.height() as _];
    ColorImage::from_rgba_unmultiplied(size, image_buffer.as_flat_samples().as_slice())
}

#[inline(always)]
pub fn imgbuf_to_texture(
    img: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    ctx: &egui::Context,
) -> Option<TextureHandle> {
    Some(ctx.load_texture(
        "current_frame",
        imgbuf_to_colorimg(&img),
        TextureOptions::default(),
    ))
}

#[derive(PartialEq)]
enum Mode {
    Image,
    Video,
    Feed,
}
