use super::{api::*, localization::*};
use abstractions::*;
use audio::AudioData;
use bq::*;
use image::{open, ImageBuffer, Rgba};
use models::Task;
use processing::pre::compute_mel;
use processing::post::PostProcessing;
use render::*;
use rest::{
    check_boquila_hub_api, detect_remotely, get_ipv4_address, rgb_image_to_jpeg_buffer, run_api,
};
use std::fs::{self};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rodio::Source;
use video_file::VideofileProcessor;

pub fn run_gui() {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "BoquilaHUB",
        native_options,
        Box::new(|cc| {
            Gui::setup(&cc.egui_ctx);
            Ok(Box::new(Gui::new()))
        }),
    );
}

macro_rules! ai_config_window {
    ($self:expr, $ctx:expr, $show_field:expr, $config_field:ident, $temp_config_field:ident, $variant:expr, $current_ai_fn:ident) => {
        if $show_field {
            egui::Window::new($self.t(Key::configure_ai))
                .collapsible(false)
                .resizable(false)
                .show($ctx, |ui| {
                    let text_slide1 = $self.t(Key::confidence_level);
                    ui.add(
                        egui::Slider::new(
                            &mut $self.$temp_config_field.confidence_threshold,
                            0.10..=0.99,
                        )
                        .text(text_slide1),
                    );
                    let has_nms = $self
                        .$current_ai_fn()
                        .post_processing
                        .contains(&PostProcessing::NMS);
                    if has_nms {
                        let text_slide2 = $self.t(Key::overlap_filter);
                        ui.add(
                            egui::Slider::new(
                                &mut $self.$temp_config_field.nms_threshold,
                                0.10..=0.99,
                            )
                            .text(text_slide2),
                        );
                    }
                    let has_geofence = $self
                        .$current_ai_fn()
                        .post_processing
                        .contains(&PostProcessing::GeoFence);
                    if has_geofence {
                        ui.label($self.t(Key::region_filter));
                        egui::ComboBox::from_id_salt("Region")
                            .selected_text($self.$temp_config_field.geo_fence.clone())
                            .show_ui(ui, |ui| {
                                for str in utils::COUNTRY_CODES {
                                    ui.selectable_value(
                                        &mut $self.$temp_config_field.geo_fence,
                                        str.to_owned(),
                                        str,
                                    );
                                }
                            });
                    }
                    ui.horizontal(|ui| {
                        if ui.button($self.t(Key::ok)).clicked() {
                            $self.$config_field = $self.$temp_config_field.clone();
                            $variant.update_config($self.$config_field.clone());
                            $show_field = false;
                        }
                        ui.add_space(8.0);
                        if ui.button($self.t(Key::cancel)).clicked() {
                            $self.$temp_config_field = $self.$config_field.clone();
                            $show_field = false;
                        }
                    });
                });
        }
    };
}

#[derive(Default)]
struct Gui {
    // Large types first
    ais: Vec<AIMetadata>,
    ais_cls_only: Vec<AIMetadata>,
    selected_files: Vec<PredImg>,
    video_file_path: Option<PathBuf>,
    audio_file_path: Option<PathBuf>,
    audio_data: Option<AudioData>,
    audio_full_mel: Option<ndarray::Array2<f32>>,
    audio_mel_meta: Option<(usize, usize, usize, f32)>, // n_fft, hop_length, n_mels, top_db
    audio_view_range: (f64, f64),
    audio_view_range_dirty: bool,
    audio_predictions: Option<Vec<AudioProb>>,
    audio_result_receiver: Option<std::sync::mpsc::Receiver<Result<AIOutputs, tokio::task::JoinError>>>,
    audio_playing: bool,
    audio_play_start: Option<Instant>,
    audio_play_start_pos: f64,
    audio_playhead: Option<f64>,
    audio_stream: Option<rodio::MixerDeviceSink>,
    audio_player: Option<rodio::Player>,
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

    // Model Configurations
    ai_config: ModelConfig,
    ai_cls_config: ModelConfig,
    temp_ai_config: ModelConfig,
    temp_ai_cls_config: ModelConfig,

    // usize and Option<usize> fields grouped together (8 bytes each on 64-bit)
    ai_selected: Option<usize>,
    ai_cls_selected: Option<usize>,
    ep_selected: Ep,
    video_step_frame: usize,
    feed_step_frame: usize,
    current_frame: u64,
    total_frames: Option<u64>,

    image_texture_n: usize,

    // Enums (size depends on variants, but typically 1-8 bytes)
    lang: Lang,
    mode: Mode,

    // bool fields grouped together (1 byte each, but will be padded)
    show_ai_cls: bool,
    isapi_deployed: bool,
    save_img_from_feed: bool,
    process_all_imgs: bool,
    show_config: ShowConfig,
    dialog: OpenDialog,
    img_state: State,
    video_state: State,
    feed_state: State,
    audio_state: State,
}

#[derive(Default)]
struct State {
    cancel_sender: Option<tokio::sync::oneshot::Sender<()>>,
    texture: Option<egui::TextureHandle>,
    is_processing: bool,
    progress_bar: f32,
}

impl State {
    fn cancel(&mut self) {
        if let Some(cancel_tx) = self.cancel_sender.take() {
            let _ = cancel_tx.send(());
        }
        self.is_processing = false;
    }
}

#[derive(Default)]
struct ShowConfig {
    ai: bool,
    ai_cls: bool,
    _img: bool,
    video: bool,
    feed: bool,
}

#[derive(Default, PartialEq)]
enum OpenDialog {
    #[default]
    None,
    ProcessAll,
    Export,
    FeedUrl,
    ApiServer,
}

impl Gui {
    fn setup(ctx: &egui::Context) {
        let mut fonts = egui::FontDefinitions::default();

        fonts.font_data.insert(
            "Noto".to_owned(),
            egui::FontData::from_static(&render::FONT_BYTES.as_ref()).into(),
        );

        fonts
            .families
            .get_mut(&egui::FontFamily::Proportional)
            .unwrap()
            .insert(0, "Noto".to_owned());

        ctx.set_fonts(fonts);
    }

    fn new() -> Self {
        let ais: Vec<AIMetadata> = BQModel::get_list();
        let classify_ais: Vec<AIMetadata> = ais
            .iter()
            .filter(|ai| ai.task == Task::Classify && ai.modality == Modality::Image)
            .cloned()
            .collect();

        Self {
            ais: ais,
            ais_cls_only: classify_ais,
            image_texture_n: 1,
            video_step_frame: 3,
            feed_step_frame: 3,
            process_all_imgs: true,
            ..Default::default()
        }
    }

    fn is_any_processing(&self) -> bool {
        self.video_state.is_processing
            || self.img_state.is_processing
            || self.feed_state.is_processing
            || self.audio_state.is_processing
    }

    fn process_done(&mut self) {
        self.done_time = Some(Instant::now());
    }

    fn process_error(&mut self) {
        self.error_time = Some(Instant::now());
    }

    fn t(&self, key: Key) -> &'static str {
        translate(key, &self.lang)
    }

    fn current_ai(&self) -> &AIMetadata {
        return &self.ais[self.ai_selected.unwrap()];
    }

    fn current_ai_cls(&self) -> &AIMetadata {
        return &self.ais_cls_only[self.ai_cls_selected.unwrap()];
    }

    fn is_audio_model(&self) -> bool {
        self.ai_selected
            .map(|i| self.ais[i].modality == Modality::Audio)
            .unwrap_or(false)
    }

    fn is_image_model(&self) -> bool {
        !self.is_audio_model()
    }

    fn audio_mel_params(&self) -> (usize, usize, usize, f32) {
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

    fn paint(&mut self, ui: &egui::Ui, i: usize) {
        let img = &self.draw_gui(&self.selected_files[i]);
        self.img_state.texture = imgbuf_to_texture(img, ui)
    }

    fn draw_gui(&self, predimg: &PredImg) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
        let mut img = image::open(&predimg.file_path).unwrap().into_rgb8();

        if predimg.wasprocessed {
            if predimg.aioutput.as_ref().unwrap().is_empty() {
                draw_no_predictions(&mut img, Some(&self.lang));
            } else {
                draw_aioutput(&mut img, &predimg.aioutput.as_ref().unwrap());
            }
}

        image::DynamicImage::ImageRgb8(img).to_rgba8()
    }

    fn save_gui(&self, predimg: &PredImg) {
        let img_data = image::DynamicImage::ImageRgba8(self.draw_gui(predimg)).to_rgb8();
        let filename = export::prepare_export_img(&predimg.file_path);
        img_data.save(&filename).unwrap();
    }

    fn show_timed_message(
        time: &mut Option<std::time::Instant>,
        ui: &mut egui::Ui,
        message: &str,
    ) {
        if let Some(start_time) = *time {
            if start_time.elapsed().as_secs_f32() < 3.0 {
                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                    ui.label(message);
                });
                ui.request_repaint();
            } else {
                *time = None;
            }
        }
    }

    fn show_done_message(&mut self, ui: &mut egui::Ui) {
        let message = &self.t(Key::done);
        let time = &mut self.done_time;
        Gui::show_timed_message(time, ui, message);
    }

    fn show_error_message(&mut self, ui: &mut egui::Ui) {
        let message = &self.t(Key::error_ocurred);
        let time = &mut self.error_time;
        Gui::show_timed_message(time, ui, message);
    }

    fn api_widget(&mut self, ui: &mut egui::Ui) {
        if self.ai_selected.is_some() && self.ep_selected.is_local() && self.is_image_model() {
            ui.label(self.t(Key::api));
            if !self.isapi_deployed {
                if ui
                    .button(self.t(Key::deploy))
                    .on_hover_text(self.t(Key::deployed_api_allows))
                    .clicked()
                {
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
        }

        self.input_api_url_dialog(ui);
    }

    fn ai_widget(&mut self, ui: &mut egui::Ui) {
        if self.ep_selected.is_local() {
            let previous_ai = self.ai_selected;
            ui.label(self.t(Key::select_ai));

            ui.horizontal(|ui| {
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

                if self.ai_selected.is_some() {
                    if ui
                        .button("⚙")
                        .on_hover_text(self.t(Key::configure_ai))
                        .clicked()
                    {
                        self.show_config.ai = true;
                    }
                }

                // '+' button, select a escond AI
                if self.ai_selected.is_some() && !self.show_ai_cls && !self.ais_cls_only.is_empty() && self.is_image_model()
                {
                    if self.ais[self.ai_selected.unwrap()].task != Task::Classify {
                        if ui
                            .button("+")
                            .on_hover_text(self.t(Key::add_classification_model_to_complement))
                            .clicked()
                        {
                            self.show_ai_cls = true;
                        }
                    }
                }
            });

            if (self.ai_selected != previous_ai) && (self.ai_selected.is_some()) {
                if self.is_audio_model() {
                    self.show_ai_cls = false;
                    self.ai_cls_selected = None;
                    GlobalBQ::Second.clear();
                    if self.audio_data.is_none() {
                        self.audio_data = None;
                        self.audio_file_path = None;
                        self.audio_predictions = None;
                    }
                    // Mel params may differ for the new audio model — invalidate cache.
                    self.audio_full_mel = None;
                    self.audio_mel_meta = None;
                    self.audio_state.texture = None;
                }
                let model_path = self.ais[self.ai_selected.unwrap()].get_path();
                if GlobalBQ::First.set_model(
                    &model_path,
                    self.ep_selected,
                    Some(self.ai_config.clone()),
                ).is_err() {
                    self.process_error();
                }
            }

            ai_config_window!(
                self,
                ui,
                self.show_config.ai,
                ai_config,
                temp_ai_config,
                GlobalBQ::First,
                current_ai
            );

            ui.add_space(8.0);
        }
    }

    fn ai_cls_widget(&mut self, ui: &mut egui::Ui) {
        if self.ep_selected.is_local() && self.show_ai_cls {
            let previous_ai = self.ai_cls_selected;
            ui.label(self.t(Key::select_2nd_ai));
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("AI_CLS")
                    .selected_text(match self.ai_cls_selected {
                        Some(i) => &self.ais_cls_only[i].name,
                        None => "",
                    })
                    .show_ui(ui, |ui| {
                        for (i, ai) in self.ais_cls_only.iter().enumerate() {
                            ui.selectable_value(&mut self.ai_cls_selected, Some(i), &ai.name)
                                .on_hover_text(&ai.classes.join(", "));
                        }
                    });

                // Button to remove AI, and unload it from memory.
                if self.ai_cls_selected.is_some() {
                    if ui
                        .button("⚙")
                        .on_hover_text(self.t(Key::configure_ai))
                        .clicked()
                    {
                        self.show_config.ai_cls = true;
                    }

                    if ui.button("-").clicked() {
                        self.show_ai_cls = false;
                        self.ai_cls_selected = None;
                        GlobalBQ::Second.clear();
                    }
                }
            });
            if (self.ai_cls_selected != previous_ai) && (self.ai_cls_selected.is_some()) {
                let model_path = self.ais_cls_only[self.ai_cls_selected.unwrap()].get_path();
                if GlobalBQ::Second.set_model(
                    &model_path,
                    self.ep_selected,
                    Some(self.ai_cls_config.clone()),
                ).is_err() {
                    self.process_error();
                }
            }

            ai_config_window!(
                self,
                ui,
                self.show_config.ai_cls,
                ai_cls_config,
                temp_ai_cls_config,
                GlobalBQ::Second,
                current_ai_cls
            );

            ui.add_space(8.0);
        }
    }

    fn get_endpoint(&self) -> Option<String> {
        if !self.ep_selected.is_local() {
            self.api_server_url
                .as_ref()
                .map(|url| format!("{}/upload", url))
        } else {
            None
        }
    }

    fn set_ai(&mut self) {
        if let Some(ai_index) = self.ai_selected {
            let _ = GlobalBQ::First.set_model(
                &self.ais[ai_index].get_path(),
                self.ep_selected,
                Some(self.ai_config.clone()),
            );
        }
    }

    fn set_ai_cls(&mut self) {
        if let Some(_ai_cls_index) = self.ai_cls_selected {
            let _ = GlobalBQ::Second.set_model(
                &self.current_ai_cls().get_path(),
                self.ep_selected,
                Some(self.ai_cls_config.clone()),
            );
        }
    }

    fn ep_widget(&mut self, ui: &mut egui::Ui) {
        ui.label(self.t(Key::select_ep));
        let mut temp_ep_selected = self.ep_selected;

        egui::ComboBox::from_id_salt("EP")
            .selected_text(self.ep_selected.name())
            .show_ui(ui, |ui| {
                for ep in [Ep::Cpu, Ep::Cuda, Ep::BoquilaHubRemote] {
                    ui.selectable_value(&mut temp_ep_selected, ep, ep.name())
                        .on_hover_text(format!(
                            "Version: {:.1}, Local: {}, Dependencies: {}",
                            ep.version().unwrap_or(0.0),
                            ep.is_local(),
                            ep.dependencies()
                        ));
                }
            });

        if temp_ep_selected != self.ep_selected {
            match temp_ep_selected {
                Ep::BoquilaHubRemote => {
                    self.dialog = OpenDialog::ApiServer;
                }
                Ep::Cuda => {
                    let cuda_version = match temp_ep_selected.version() {
                        Ok(cuda_v) => cuda_v,
                        Err(error) => {
                            eprintln!("Could not find CUDA version with error: {error}");
                            return;
                        }
                    };

                    if cuda_version >= 12.8 {
                        self.ep_selected = temp_ep_selected;

                        self.set_ai();
                        self.set_ai_cls();
                    } else {
                        self.process_error();
                    }
                }
                _ => {
                    self.ep_selected = temp_ep_selected;

                    self.set_ai();
                    self.set_ai_cls();
                }
            }
        }
        ui.add_space(8.0);
    }

    fn data_selection_widget(&mut self, ui: &mut egui::Ui) {
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
                    if let Some(folder_path) = rfd::FileDialog::new().pick_folder() {
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
                                                formats::IMAGE_FORMATS.iter().any(|&format| {
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
                                    self.image_texture_n = 1;
                                    self.paint(ui, 0);
                                    self.mode = Mode::Image;
                                    self.img_state.progress_bar = self.selected_files.get_progress()
                                }
                            }
                            Err(_e) => {
                                self.process_error();
                            }
                        }
                    }
                }

                // IMAGE FILE SELECTION SECTION
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::image)))
                    .clicked()
                {
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter("Image", &formats::IMAGE_FORMATS)
                        .pick_files()
                    {
                        self.selected_files = paths
                            .into_iter()
                            .map(|path| PredImg::new_simple(path))
                            .collect();
                        self.image_texture_n = 1;
                        self.paint(ui, 0);
                        self.mode = Mode::Image;
                        self.img_state.progress_bar = self.selected_files.get_progress()
                    }
                }
                ui.end_row();

                // VIDEO FILE SELECTION SECTION
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::video_file)))
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Video", &formats::VIDEO_FORMATS)
                        .pick_file()
                    {
                        match VideofileProcessor::first_frame(path.clone().to_str().unwrap()) {
                            Ok(frame) => {
                                self.video_state.texture = imgbuf_to_texture(
                                    &image::DynamicImage::ImageRgb8(frame).to_rgba8(),
                                    ui,
                                );
                                self.video_file_path = Some(path);
                                let processor = Some(VideofileProcessor::new(
                                    &self.video_file_path.clone().unwrap().to_str().unwrap(),
                                ))
                                .unwrap();
                                self.total_frames = Some(processor.get_n_frames());
                                self.video_file_processor = Arc::new(Mutex::new(Some(processor)));
                                self.mode = Mode::Video;
                                self.current_frame = 0;
                                self.video_state.progress_bar = 0.0;
                            }
                            Err(_) => {
                                self.process_error();
                            }
                        }
                    }
                }

                // Camera feed
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::camera_feed)))
                    .clicked()
                {
                    self.dialog = OpenDialog::FeedUrl;
                }

                ui.end_row();
            });

        ui.vertical_centered(|ui| {
                // AUDIO FILE SELECTION SECTION
                if ui
                    .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::audio_file)))
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Audio", &formats::AUDIO_FORMATS)
                        .pick_file()
                    {
                        match audio::AudioData::from_file(&path) {
                            Ok(audio_data) => {
                                let mono = audio_data.to_mono();
                                let dur = mono.duration();
                                self.audio_data = Some(mono);
                                self.audio_file_path = Some(path);
                                self.audio_full_mel = None;
                                self.audio_mel_meta = None;
                                self.audio_view_range = (0.0, dur.max(0.1));
                                self.audio_view_range_dirty = true;
                                self.audio_state.texture = None;
                                self.audio_predictions = None;
                                self.audio_playhead = None;
                                self.audio_playing = false;
                                self.audio_play_start = None;
                                self.audio_play_start_pos = 0.0;
                                self.audio_player = None;
                                self.audio_stream = None;
                                self.mode = Mode::Audio;
                                self.audio_state.progress_bar = 0.0;
                            }
                            Err(_) => {
                                self.process_error();
                            }
                        }
                    }
                }
            });

        // Feed url dialog
        if self.dialog == OpenDialog::FeedUrl {
            self.feed_input_dialog(ui);
        }
    }

    fn feed_input_dialog(&mut self, ui: &egui::Ui) {
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
                                    self.feed_state.texture = imgbuf_to_texture(
                                        &image::DynamicImage::ImageRgb8(frame).to_rgba8(),
                                        ui,
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
                        self.dialog = OpenDialog::None;
                    }
                    ui.add_space(8.0);
                    if ui.button(self.t(Key::cancel)).clicked() {
                        self.dialog = OpenDialog::None;
                        self.feed_url = None
                    }
                });
            });
    }

    fn input_api_url_dialog(&mut self, ui: &egui::Ui) {
        if self.dialog == OpenDialog::ApiServer {
            egui::Window::new(self.t(Key::input_url))
                .collapsible(false)
                .resizable(false)
                .show(ui, |ui| {
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
                                self.dialog = OpenDialog::None;
                                self.api_server_url = Some(url);
                                self.ep_selected = Ep::BoquilaHubRemote;
                            } else {
                                self.process_error();
                            }
                        }
                        ui.add_space(8.0);
                        if ui.button(self.t(Key::cancel)).clicked() {
                            self.dialog = OpenDialog::None;
                            self.api_server_url = None;
                        }
                    });
                });
        }
    }

    fn start_img_analysis(&mut self) {
        self.img_state.is_processing = true;
        // process all, even if they were process before
        if self.process_all_imgs {
            self.selected_files
                .iter_mut()
                .for_each(|pred_img| pred_img.reset());
        }
        //  Async processing: Images
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.image_processing_receiver = Some(rx);

        let copy_predigms = self.selected_files.clone();
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        self.img_state.cancel_sender = Some(cancel_tx);

        let api_endpoint = self.get_endpoint();
        let is_remote = !self.ep_selected.is_local();
        tokio::spawn(async move {
            for (i, predimg) in copy_predigms.iter().enumerate() {
                if predimg.wasprocessed {
                    continue;
                }
                // CHECK FOR CANCELLATION HERE
                if cancel_rx.try_recv().is_ok() {
                    break;
                }
                let bbox = if is_remote {
                    let buffer = fs::read(&predimg.file_path).unwrap();
                    match detect_remotely(api_endpoint.as_ref().unwrap(), buffer).await {
                        Ok(result) => result,
                        Err(_) => {
                            break;
                        }
                    }
                } else {
                    let img = open(&predimg.file_path).unwrap().into_rgb8();
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

    fn process_all_dialog(&mut self, ui: &egui::Ui) {
        if self.dialog == OpenDialog::ProcessAll {
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
    }

    fn img_analysis_widget(&mut self, ui: &mut egui::Ui) {
        if self.selected_files.len() >= 1 && (self.is_image_model() || !self.ep_selected.is_local()) {
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
                {
                    if !self.img_state.is_processing {
                        if self.selected_files.get_progress() == 0.0 {
                            self.start_img_analysis();
                        } else {
                            self.dialog = OpenDialog::ProcessAll;
                        }
                    }
                }

                // Cancel button widget
                if self.img_state.is_processing {
                    if ui
                        .add_sized([85.0, 40.0], egui::Button::new(self.t(Key::cancel)))
                        .clicked()
                    {
                        self.img_state.cancel();
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
                        self.dialog = OpenDialog::Export;
                    }
                });
            }

            // Export dialog logic
            if self.dialog == OpenDialog::Export {
                egui::Window::new(self.t(Key::export))
                    .collapsible(false)
                    .resizable(false)
                    .show(ui, |ui| {
                        // Export option 1
                        if ui.button(self.t(Key::export_predictions)).clicked() {
                            for file in self.selected_files.clone() {
                                tokio::spawn(async move {
                                    let _ = file.write_pred_img_to_file().await;
                                });
                            }

                            self.process_done();
                            self.dialog = OpenDialog::None;
                        }

                        // Export option 2
                        if ui
                            .button(self.t(Key::export_imgs_with_predictions))
                            .clicked()
                        {
                            for file in &self.selected_files {
                                if file.wasprocessed {
                                    self.save_gui(file);
                                }
                            }
                            self.process_done();
                            self.dialog = OpenDialog::None;
                        }

                        // Export option 3
                        if ui.button(self.t(Key::copy_with_classification)).clicked() {
                            let timestamp =
                                chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
                            tokio::spawn({
                                let selected_files = self.selected_files.clone(); // Make sure it's Send + 'static
                                let path = format!("export/export_{}", timestamp);
                                async move {
                                    let _ = export::copy_to_folder(&selected_files, &path).await;
                                }
                            });
                            self.process_done();
                            self.dialog = OpenDialog::None;
                        }

                        // Cancel any export
                        if ui.button(self.t(Key::cancel)).clicked() {
                            self.dialog = OpenDialog::None;
                        }
                    });
            }

            self.process_all_dialog(ui);
        }
    }

    fn audio_analysis_widget(&mut self, ui: &mut egui::Ui) {
        if self.audio_data.is_some() && (self.is_audio_model() || !self.ep_selected.is_local()) {
            ui.vertical_centered(|ui| {
                ui.heading(self.t(Key::audio_file));
                ui.heading(self.t(Key::analysis));
            });
            ui.separator();

            if let Some(path) = &self.audio_file_path {
                ui.label(
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("(unknown)")
                );
            }

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
        }
    }

    fn start_audio_analysis(&mut self) {
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

    fn start_video_analysis(&mut self) {
        self.video_state.is_processing = true;
        // Async processing: Video
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.video_processing_receiver = Some(rx);
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        self.video_state.cancel_sender = Some(cancel_tx);

        let processor = Arc::clone(&self.video_file_processor);
        let n = self.total_frames.unwrap();
        let current = self.current_frame;

        let api_endpoint = self.get_endpoint();
        let is_remote = !self.ep_selected.is_local();
        let step: usize = self.video_step_frame;
        let mut cached_aioutput: Option<AIOutputs> = None;

        tokio::spawn(async move {
            for i in current..=n {
                if cancel_rx.try_recv().is_ok() {
                    break;
                }

                let data = processor.lock().unwrap().as_mut().unwrap().next();

                match data {
                    Some((time, mut img)) => {
                        // Only process if frequency says so
                        if i % step as u64 == 0 {
                            let aioutput;
                            if is_remote {
                                let buffer = rgb_image_to_jpeg_buffer(&img, 95);
                                match detect_remotely(api_endpoint.as_ref().unwrap(), buffer).await
                                {
                                    Ok(result) => aioutput = result,
                                    Err(_) => break,
                                }
                            } else {
                                match tokio::task::spawn_blocking(move || {
                                    let result = process_imgbuf(&img);
                                    (img, result) // return img back alongside the result
                                })
                                .await
                                {
                                    Ok((returned_img, result)) => {
                                        img = returned_img;
                                        aioutput = result;
                                    }
                                    Err(_) => break,
                                }
                            }
                            cached_aioutput = Some(aioutput);
                        }

                        if let Some(ref aioutput) = cached_aioutput {
                            draw_aioutput(&mut img, aioutput);
                        }

                        // Process final image
                        processor
                            .lock()
                            .unwrap()
                            .as_mut()
                            .unwrap()
                            .encode(&img, time);

                        let final_img = image::DynamicImage::ImageRgb8(img).to_rgba8();

                        if tx.send((i, final_img)).is_err() {
                            break;
                        }
                    }
                    None => {
                        let _ = processor.lock().unwrap().as_mut().unwrap().encoder.finish();
                        break;
                    }
                }
            }
        });
    }

    fn video_analysis_widget(&mut self, ui: &mut egui::Ui) {
        if self.video_file_path.is_some() && (self.is_image_model() || !self.ep_selected.is_local()) {
            ui.vertical_centered(|ui| {
                ui.heading(self.t(Key::video_file));
                ui.heading(self.t(Key::analysis));

                ui.separator();

                ui.add_enabled_ui(!self.video_state.is_processing, |ui| {
                    if ui.button("⚙").clicked() {
                        self.show_config.video = !self.show_config.video;
                    }
                    if self.show_config.video {
                        ui.label(self.t(Key::freq));
                        ui.style_mut().spacing.slider_width = 125.0;
                        ui.add(egui::Slider::new(&mut self.video_step_frame, 1..=90));
                        ui.add_space(8.0);
                    }
                });

                if !self.video_state.is_processing {
                    if ui.button("▶").clicked() {
                        self.start_video_analysis();
                    }
                } else {
                    if ui.button("⏸").clicked() {
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

    fn cancel_video_processing(&mut self) {
        self.video_state.cancel();
        self.video_processing_receiver = None;
    }

    fn start_feed_analysis(&mut self) {
        self.feed_state.is_processing = true;
        // Async processing: Video
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.feed_processing_receiver = Some(rx);
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
        self.feed_state.cancel_sender = Some(cancel_tx);

        let mut feed = stream::Feed::new(&self.feed_url.clone().unwrap()).unwrap();

        let api_endpoint = self.get_endpoint();
        let is_remote = !self.ep_selected.is_local();
        let step: usize = self.feed_step_frame;
        tokio::spawn(async move {
            let mut frame_counter = 0;
            loop {
                if cancel_rx.try_recv().is_ok() {
                    break;
                }

                if let Some(mut img) = feed.next() {
                    frame_counter = frame_counter + 1;

                    if frame_counter % step == 0 {
                        let aioutput: AIOutputs = if is_remote {
                            let buffer = rgb_image_to_jpeg_buffer(&img, 95);
                            match detect_remotely(api_endpoint.as_ref().unwrap(), buffer).await {
                                Ok(result) => result,
                                Err(_) => break,
                            }
                        } else {
                            match tokio::task::spawn_blocking(move || {
                                let result = process_imgbuf(&img);
                                (img, result)
                            })
                            .await
                            {
                                Ok((returned_img, result)) => {
                                    img = returned_img;
                                    result
                                }
                                Err(_) => break,
                            }
                        };

                        draw_aioutput(&mut img, &aioutput);
                        let img = image::DynamicImage::ImageRgb8(img).to_rgba8();
                        if tx.send((aioutput, img)).is_err() {
                            break;
                        }
                    }
                }
            }
        });
    }

    fn feed_analysis_widget(&mut self, ui: &mut egui::Ui) {
        if self.feed_url.is_some() && (self.is_image_model() || !self.ep_selected.is_local()) {
            ui.vertical_centered(|ui| {
                ui.heading(self.t(Key::camera_feed));
                ui.heading(self.t(Key::analysis));

                ui.separator();
                ui.add_space(8.0);

                if ui.button("⚙").clicked() {
                    self.show_config.feed = !self.show_config.feed;
                }

                if self.show_config.feed {
                    ui.label(self.t(Key::export_obs));
                    ui.checkbox(&mut self.save_img_from_feed, "");
                    ui.add_space(8.0);
                    ui.add_enabled_ui(!self.feed_state.is_processing, |ui| {
                        ui.label(self.t(Key::freq));
                        ui.style_mut().spacing.slider_width = 120.0;
                        ui.add(egui::Slider::new(&mut self.feed_step_frame, 1..=90));
                        ui.add_space(8.0);
                    });
                }
                if !self.feed_state.is_processing {
                    if ui.button("▶").clicked() {
                        self.start_feed_analysis();
                    }
                } else {
                    if ui.button("⏸").clicked() {
                        self.feed_state.cancel();
                    }
                }
            });
        }
    }

    fn img_handle_results(&mut self, ui: &egui::Ui) {
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
                    self.paint(ui, i);
                }
            }

            if self.selected_files.iter().all(|f| f.wasprocessed) {
                self.img_state.is_processing = false;
                self.image_processing_receiver = None;
            }

            self.img_state.progress_bar = self.selected_files.get_progress();
            ui.request_repaint();
        }
    }

    fn video_handle_results(&mut self, ui: &egui::Ui) {
        if let Some(rx) = &mut self.video_processing_receiver {
            let mut updates = Vec::new();
            let mut channel_closed = false;
            loop {
                match rx.try_recv() {
                    Ok(img) => updates.push(img),
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        channel_closed = true;
                        break;
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                }
            }

            for (i, img) in updates {
                self.video_state.texture = imgbuf_to_texture(&img, ui);
                self.video_state.progress_bar = (i + 1) as f32 / self.total_frames.unwrap() as f32;
                self.current_frame = i;
            }

            if channel_closed {
                self.video_state.progress_bar = 1.0;
                self.video_state.is_processing = false;
                self.video_processing_receiver = None;
                let _ = self
                    .video_file_processor
                    .lock()
                    .unwrap()
                    .as_mut()
                    .unwrap()
                    .encoder
                    .finish();
            }

            ui.request_repaint();
        }
    }

    fn feed_handle_results(&mut self, ui: &egui::Ui) {
        if let Some(rx) = &mut self.feed_processing_receiver {
            let mut updates = Vec::new();
            while let Ok(img) = rx.try_recv() {
                updates.push(img);
            }

            for (aioutput, img) in updates {
                self.feed_state.texture = imgbuf_to_texture(&img, ui);
                if self.save_img_from_feed && !aioutput.is_empty() {
                    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
                    let _ = img.save(format!("export/feed/{}.png", timestamp));
                }
            }

            ui.request_repaint();
        }
    }

    fn start_playback_from_data(&mut self, start_pos: f64) {
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

    fn stop_playback(&mut self) {
        if let Some(player) = &self.audio_player {
            player.stop();
        }
        self.audio_player = None;
        self.audio_stream = None;
        self.audio_playing = false;
        self.audio_play_start = None;
    }

    fn audio_handle_results(&mut self) {
        if let Some(rx) = &self.audio_result_receiver {
            if let Ok(result) = rx.try_recv() {
                match result {
                    Ok(AIOutputs::AudioClassification(preds)) => {
                        self.audio_predictions = Some(preds);
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
}

impl eframe::App for Gui {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn ui(&mut self, main_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top_panel").show_inside(main_ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button(self.t(Key::about), |ui| {
                    ui.hyperlink_to(
                        self.t(Key::source_code),
                        "https://github.com/boquila/boquilahub/",
                    );
                });

                ui.menu_button(self.t(Key::models), |ui| {
                    ui.hyperlink_to(self.t(Key::model_hub), self.t(Key::model_hub_url));
                });

                ui.menu_button(self.t(Key::idiom), |ui| {
                    for (lang, label) in LANGUAGES {
                        ui.radio_value(&mut self.lang, lang, label);
                    }
                });

                egui::widgets::global_theme_preference_switch(ui);
            });
        });

        egui::Panel::left("left_panel").show_inside(main_ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading(format!("💻 {}", self.t(Key::setup)));
            });
            ui.separator();

            ui.scope(|ui| {
                if self.is_any_processing() {
                    ui.disable();
                }
                self.ai_widget(ui);
                self.ai_cls_widget(ui);
                self.ep_widget(ui);

                self.api_widget(ui);

                ui.separator();

                self.data_selection_widget(ui);

                ui.separator();
            });

            match self.mode {
                Mode::Image => {
                    self.img_analysis_widget(ui);
                }
                Mode::Video => {
                    self.video_analysis_widget(ui);
                }
                Mode::Feed => {
                    self.feed_analysis_widget(ui);
                }
                Mode::Audio => {
                    self.audio_analysis_widget(ui);
                }
            }

            self.show_done_message(ui);
            self.show_error_message(ui);
        });

        egui::CentralPanel::default().show_inside(main_ui, |ui| {
            let cond1 = self.selected_files.len() >= 1;
            let cond2 = self.video_file_path.is_some();
            let cond3 = self.feed_url.is_some();
            let cond4 = self.audio_file_path.is_some();
            // it has a mode AND another
            let img_mode = cond1 && (cond2 || cond3 || cond4);
            let video_mode = cond2 && (cond1 || cond3 || cond4);
            let feed_mode = cond3 && (cond1 || cond2 || cond4);
            let audio_mode = cond4 && (cond1 || cond2 || cond3);
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
                if audio_mode {
                    ui.selectable_value(&mut self.mode, Mode::Audio, "Audio");
                }
            });

            if img_mode || video_mode || feed_mode || audio_mode {
                ui.separator();
            }
            egui::ScrollArea::vertical().show(ui, |ui| match self.mode {
                Mode::Image => {
                    ui.style_mut().spacing.slider_width = 300.0;
                    if self.selected_files.len() > 1 {
                        let images_slider = ui.add(
                            egui::Slider::new(
                                &mut self.image_texture_n,
                                1..=self.selected_files.len(),
                            )
                            .text(""),
                        );

                        if images_slider.changed() {
                            self.paint(ui, self.image_texture_n - 1);
                        }
                    }

                    if let Some(texture) = &self.img_state.texture {
                        ui.add(
                            egui::Image::new(texture)
                                .max_height(800.0)
                                .corner_radius(10.0),
                        );
                    }

                    self.img_handle_results(ui);
                }
                Mode::Video => {
                    if let Some(texture) = &self.video_state.texture {
                        ui.add(
                            egui::Image::new(texture)
                                .max_height(800.0)
                                .corner_radius(10.0),
                        );
                    }

                    self.video_handle_results(ui);
                }
                Mode::Feed => {
                    if let Some(texture) = &self.feed_state.texture {
                        ui.add(
                            egui::Image::new(texture)
                                .max_height(800.0)
                                .corner_radius(10.0),
                        );
                    }

                    self.feed_handle_results(ui);
                }
                Mode::Audio => {
                    if self.audio_data.is_some() {
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

                        let window_preds: Vec<AudioProb> = self
                            .audio_predictions
                            .as_ref()
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
                            PLOT_SPEC_W,
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
                                    PLOT_SPEC_W,
                                    PLOT_SPEC_H,
                                    &window_preds,
                                    &column_winner,
                                );
                                self.audio_state.texture = imgbuf_to_texture(&img, ui);
                            }
                        }

                        if let Some(texture) = self.audio_state.texture.clone() {
                            let apply_bounds =
                                std::mem::take(&mut self.audio_view_range_dirty);
                            let result = render_audio_plot(
                                ui,
                                &texture,
                                self.audio_view_range,
                                duration,
                                &window_preds,
                                &column_winner,
                                self.audio_playhead,
                                apply_bounds,
                            );

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
            });
        });
    }
}

#[inline(always)]
fn imgbuf_to_texture(
    img: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    ui: &egui::Ui,
) -> Option<egui::TextureHandle> {
    let size: [usize; 2] = [img.width() as _, img.height() as _];
    let color_img =
        egui::ColorImage::from_rgba_unmultiplied(size, img.as_flat_samples().as_slice());
    Some(ui.load_texture("current_frame", color_img, egui::TextureOptions::default()))
}

const AUDIO_DISPLAY_SR: u32 = 22050;
const PLOT_SPEC_W: usize = 800;
const PLOT_SPEC_H: usize = 280;
const PLOT_W: f32 = 870.0;
const PLOT_H: f32 = 360.0;

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
    view_range: (f64, f64),
    duration: f64,
    window_preds: &[AudioProb],
    column_winner: &[Option<usize>],
    playhead: Option<f64>,
    apply_bounds: bool,
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

    let plot_response = Plot::new("audio_spectrogram_plot")
        .width(PLOT_W)
        .height(PLOT_H)
        .allow_zoom([true, false])
        .allow_drag([true, false])
        .allow_scroll([true, false])
        .allow_boxed_zoom(false)
        .allow_double_click_reset(true)
        .show_x(false)
        .show_y(false)
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
            let mel = pos.y;
            if mel >= 0.0 && mel <= mel_max {
                let hz = 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
                let hz_str = if hz >= 1000.0 {
                    format!("{:.1} kHz", hz / 1000.0)
                } else {
                    format!("{:.0} Hz", hz)
                };
                format!("{:.2} s\n{}", pos.x, hz_str)
            } else {
                format!("{:.2} s", pos.x)
            }
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

            for (idx, seg) in segments.iter().enumerate() {
                let n_cols = PLOT_SPEC_W as f64;
                let x1 = x_min + (seg.col_start as f64 / n_cols) * span;
                let x2 = x_min + ((seg.col_end + 1) as f64 / n_cols) * span;
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
                    let g = (t_norm * 255.0) as u8;
                    pixels.extend_from_slice(&[g, g, g, 255]);
                }
            }
        }
    }
    ImageBuffer::from_raw(target_width as u32, target_height as u32, pixels).unwrap()
}


#[derive(Default, PartialEq)]
enum Mode {
    #[default]
    Image,
    Video,
    Feed,
    Audio,
}

struct AudioBufferSource {
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