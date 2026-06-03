mod audio;
mod feed;
#[path = "image.rs"]
mod image_view;
mod video_file;

use super::{api::*, localization::*};
use abstractions::*;
use crate::api::audio::AudioData;
use bq::*;
use models::Task;
use processing::post::PostProcessing;
use rest::{check_boquila_hub_api, get_ipv4_address, run_api};
use std::collections::{HashMap, VecDeque};
use std::fs::{self};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use crate::api::video_file::VideofileProcessor;
use crate::gui::feed::FeedFrame;
use crate::gui::video_file::{AnalysisFrame, ExportProgress};

pub fn run_gui() {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                eframe::icon_data::from_png_bytes(&include_bytes!("../../assets/icon-256.png")[..])
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
    selected_audios: Vec<PredAudio>,
    // AudioData is heavy (hours of float samples). We only keep it for the
    // currently displayed audio file — switching invalidates and reloads.
    audio_data: Option<AudioData>,
    audio_full_mel: Option<ndarray::Array2<f32>>,
    audio_mel_meta: Option<(usize, usize, usize, f32)>, // n_fft, hop_length, n_mels, top_db
    audio_tex_dims: Option<(usize, usize)>,
    audio_view_range: (f64, f64),
    audio_view_range_dirty: bool,
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
    // Per-segment alpha masks for the currently displayed image. Rebuilt in
    // `paint()` so we don't re-upload every frame.
    mask_textures: Vec<egui::TextureHandle>,

    // Feed buffer + scrub state. `feed_playhead_frame == None` means "follow
    // live"; otherwise the user is parked on a specific cached frame.
    feed_buffer: VecDeque<FeedFrame>,
    feed_buffer_max_secs: u32,
    feed_playhead_frame: Option<u64>,
    feed_last_displayed_frame: Option<u64>,

    // Video pipeline state. Thumbnails + decoder are only ever for the
    // currently displayed video; switching wipes them and rebuilds lazily.
    selected_videos: Vec<PredVideo>,
    video_thumbnails: HashMap<u64, Vec<u8>>,
    video_file_processor: Arc<Mutex<Option<VideofileProcessor>>>,
    video_export_receiver: Option<std::sync::mpsc::Receiver<ExportProgress>>,
    video_export_path: Option<String>,
    video_playhead_frame: Option<u64>,
    video_playing: bool,
    video_play_start: Option<Instant>,
    video_play_start_frame: u64,
    video_last_displayed_frame: Option<u64>,

    // Option<Instant> (likely 24 bytes: 8-byte discriminant + 16-byte Instant)
    done_time: Option<Instant>,
    error_time: Option<Instant>,
    // Optional override for the "Done" toast — used to surface the export
    // path so the user isn't left guessing where their files landed.
    done_message: Option<String>,

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

    image_texture_n: usize,
    audio_texture_n: usize,
    video_texture_n: usize,

    // Enums (size depends on variants, but typically 1-8 bytes)
    lang: Lang,
    mode: Mode,

    // bool fields grouped together (1 byte each, but will be padded)
    show_ai_cls: bool,
    isapi_deployed: bool,
    save_img_from_feed: bool,
    process_all_imgs: bool,
    process_all_audios: bool,
    show_config: ShowConfig,
    dialog: OpenDialog,
    img_state: State<(usize, AIOutputs)>,
    video_state: State<AnalysisFrame>,
    feed_state: State<FeedFrame>,
    audio_state: State<(usize, AIOutputs)>,
}

/// One per modality (image/audio/video/feed). The receiver lives here so the
/// channel-disconnect rule — "worker done iff `rx.try_recv() == Disconnected`" —
/// is enforced in one place via `drain()`. Callers cannot accidentally check
/// the wrong predicate (e.g. `selected_files.iter().all(wasprocessed)`, which
/// stays false forever during single-item analysis in a batch).
struct State<T> {
    rx: Option<tokio::sync::mpsc::UnboundedReceiver<T>>,
    cancel_sender: Option<tokio::sync::oneshot::Sender<()>>,
    texture: Option<egui::TextureHandle>,
    is_processing: bool,
    progress_bar: f32,
}

impl<T> Default for State<T> {
    fn default() -> Self {
        Self {
            rx: None,
            cancel_sender: None,
            texture: None,
            is_processing: false,
            progress_bar: 0.0,
        }
    }
}

impl<T> State<T> {
    /// Begin a cancellable job. Returns `(tx, cancel_rx)` for the worker.
    /// Sets `is_processing = true`.
    fn start(&mut self) -> (
        tokio::sync::mpsc::UnboundedSender<T>,
        tokio::sync::oneshot::Receiver<()>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        self.rx = Some(rx);
        self.cancel_sender = Some(cancel_tx);
        self.is_processing = true;
        (tx, cancel_rx)
    }

    /// Begin a receive-only streaming job (no cancel channel, no
    /// `is_processing` flag). The worker runs to EOF on its own.
    fn start_streaming(&mut self) -> tokio::sync::mpsc::UnboundedSender<T> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.rx = Some(rx);
        tx
    }

    /// Drain available messages. Returns `(messages, channel_closed)`.
    /// `channel_closed = true` is the only authoritative "worker done" signal.
    fn drain(&mut self) -> (Vec<T>, bool) {
        let Some(rx) = self.rx.as_mut() else { return (Vec::new(), false); };
        let mut out = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(item) => out.push(item),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => return (out, false),
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => return (out, true),
            }
        }
    }

    /// Whether any worker (analysis or streaming) is delivering messages.
    fn is_active(&self) -> bool {
        self.rx.is_some()
    }

    /// Natural completion: drop the receiver and clear `is_processing`.
    fn finish(&mut self) {
        self.rx = None;
        self.is_processing = false;
    }

    /// User-initiated cancel (also doubles as "abandon" for streaming jobs
    /// where the cancel sender is None — the take() is a no-op).
    fn cancel(&mut self) {
        if let Some(cancel_tx) = self.cancel_sender.take() {
            let _ = cancel_tx.send(());
        }
        self.rx = None;
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
            audio_texture_n: 1,
            video_texture_n: 1,
            video_step_frame: 3,
            feed_step_frame: 3,
            feed_buffer_max_secs: feed::FEED_BUFFER_DEFAULT_SECS,
            process_all_imgs: true,
            process_all_audios: true,
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
        self.done_message = None;
    }

    fn process_done_at(&mut self, location: impl Into<String>) {
        self.done_time = Some(Instant::now());
        let prefix = self.t(Key::saved_to);
        self.done_message = Some(format!("✅ {} {}", prefix, location.into()));
    }

    fn process_done_with(&mut self, message: impl Into<String>) {
        self.done_time = Some(Instant::now());
        self.done_message = Some(format!("✅ {}", message.into()));
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

    fn is_image_model(&self) -> bool {
        !self.is_audio_model()
    }

    fn can_run_image_ai(&self) -> bool {
        if self.ep_selected.is_local() {
            self.ai_selected.is_some() && self.is_image_model()
        } else {
            self.api_server_url.is_some()
        }
    }

    fn show_timed_message(
        time: &mut Option<std::time::Instant>,
        ui: &mut egui::Ui,
        message: &str,
        duration_secs: f32,
    ) {
        if let Some(start_time) = *time {
            if start_time.elapsed().as_secs_f32() < duration_secs {
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
        let default = self.t(Key::done);
        let message: &str = self.done_message.as_deref().unwrap_or(default);
        // Path-bearing toasts need long enough to actually be read; the plain
        // "Done" toast was 3s and felt fine, but a file path requires more.
        let duration = if self.done_message.is_some() { 10.0 } else { 3.0 };
        let time = &mut self.done_time;
        Gui::show_timed_message(time, ui, message, duration);
    }

    fn show_error_message(&mut self, ui: &mut egui::Ui) {
        let message = &self.t(Key::error_ocurred);
        let time = &mut self.error_time;
        Gui::show_timed_message(time, ui, message, 3.0);
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
                    if self.ais[self.ai_selected.unwrap()].can_add_cls() {
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
                                let (image_files, audio_files, video_files) =
                                    partition_media_in_dir(entries);

                                let had_any = !image_files.is_empty()
                                    || !audio_files.is_empty()
                                    || !video_files.is_empty();

                                if !image_files.is_empty() {
                                    self.selected_files = image_files
                                        .into_iter()
                                        .map(PredImg::new_simple)
                                        .collect();
                                    self.image_texture_n = 1;
                                    self.paint(ui, 0);
                                    self.img_state.progress_bar =
                                        self.selected_files.get_progress();
                                }

                                if !audio_files.is_empty() {
                                    self.selected_audios = audio_files
                                        .into_iter()
                                        .map(PredAudio::new_simple)
                                        .collect();
                                    self.audio_texture_n = 1;
                                    self.audio_state.progress_bar =
                                        self.selected_audios.get_progress();
                                    let _ = self.load_current_audio();
                                }

                                if !video_files.is_empty() {
                                    self.selected_videos = video_files
                                        .into_iter()
                                        .map(PredVideo::new_simple)
                                        .collect();
                                    self.video_texture_n = 1;
                                    self.load_current_video(ui);
                                }

                                // Pick a sensible default mode. Image wins when
                                // present (matches the prior single-modality
                                // behaviour); otherwise fall back to whichever
                                // modality the folder actually contained.
                                if had_any {
                                    if !self.selected_files.is_empty() {
                                        self.mode = Mode::Image;
                                    } else if !self.selected_videos.is_empty() {
                                        self.mode = Mode::Video;
                                    } else if !self.selected_audios.is_empty() {
                                        self.mode = Mode::Audio;
                                    }
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
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter("Video", &formats::VIDEO_FORMATS)
                        .pick_files()
                    {
                        if !paths.is_empty() {
                            self.selected_videos = paths
                                .into_iter()
                                .map(PredVideo::new_simple)
                                .collect();
                            self.video_texture_n = 1;
                            self.mode = Mode::Video;
                            self.load_current_video(ui);
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
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter("Audio", &formats::AUDIO_FORMATS)
                        .pick_files()
                    {
                        self.selected_audios = paths
                            .into_iter()
                            .map(PredAudio::new_simple)
                            .collect();
                        self.audio_texture_n = 1;
                        self.mode = Mode::Audio;
                        self.audio_state.progress_bar = self.selected_audios.get_progress();
                        if let Err(()) = self.load_current_audio() {
                            self.process_error();
                            self.selected_audios.clear();
                        }
                    }
                }
            });

        // Feed url dialog
        if self.dialog == OpenDialog::FeedUrl {
            self.feed_input_dialog(ui);
        }
    }

    /// Load (decode) the AudioData for the currently-selected audio file and
    /// reset all derived view state (mel, texture, view range, playhead). The
    /// returned `Ok(())` / `Err(())` reflects whether the file could be opened.
    pub(super) fn load_current_audio(&mut self) -> Result<(), ()> {
        self.stop_playback();
        self.audio_data = None;
        self.audio_full_mel = None;
        self.audio_mel_meta = None;
        self.audio_tex_dims = None;
        self.audio_state.texture = None;
        self.audio_playhead = None;
        self.audio_play_start = None;
        self.audio_play_start_pos = 0.0;

        let Some(pred) = self
            .selected_audios
            .get(self.audio_texture_n.saturating_sub(1))
        else {
            self.audio_view_range = (0.0, 0.1);
            self.audio_view_range_dirty = true;
            return Ok(());
        };

        match AudioData::from_file(&pred.file_path) {
            Ok(audio) => {
                let mono = audio.to_mono();
                let dur = mono.duration();
                self.audio_data = Some(mono);
                self.audio_view_range = (0.0, dur.max(0.1));
                self.audio_view_range_dirty = true;
                Ok(())
            }
            Err(_) => Err(()),
        }
    }

    pub(super) fn current_video(&self) -> Option<&PredVideo> {
        self.selected_videos.get(self.video_texture_n.saturating_sub(1))
    }

    pub(super) fn current_video_mut(&mut self) -> Option<&mut PredVideo> {
        let idx = self.video_texture_n.saturating_sub(1);
        self.selected_videos.get_mut(idx)
    }

    /// Load (well, "open") the currently-selected video: rebuild the preview
    /// texture from the first frame, drop any cached thumbnails / decoder, and
    /// reset playback state. Like `paint()` for images, this is the seam where
    /// switching videos becomes cheap regardless of how many are loaded.
    pub(super) fn load_current_video(&mut self, ui: &egui::Ui) {
        self.stop_video_playback();
        self.video_thumbnails.clear();
        *self.video_file_processor.lock().unwrap() = None;
        self.video_state.cancel();
        self.video_state.texture = None;
        self.video_last_displayed_frame = None;
        self.video_playhead_frame = Some(0);

        let idx = self.video_texture_n.saturating_sub(1);
        let Some(path_str) = self
            .selected_videos
            .get(idx)
            .map(|pv| pv.file_path.to_string_lossy().to_string())
        else {
            self.video_state.progress_bar = 0.0;
            return;
        };

        if let Ok(probe) = VideofileProcessor::probe(&path_str) {
            if let Some(pv) = self.selected_videos.get_mut(idx) {
                // First open for this video — fill in metadata + frames vec
                // (or load the sidecar if there is one). No-op on subsequent
                // navigations back to this same video.
                pv.hydrate(probe.width, probe.height, probe.fps, probe.n_frames);
            }

            // Cap the preview width before the RGB→RGBA copy + GPU upload, as
            // we did at file-pick time — 4K frames burn ~33 MB at full res.
            const PREVIEW_MAX_W: u32 = 1920;
            let display_rgb = if probe.first_frame.width() > PREVIEW_MAX_W {
                let h = (probe.first_frame.height() as f32
                    * PREVIEW_MAX_W as f32
                    / probe.first_frame.width() as f32) as u32;
                image::imageops::resize(
                    &probe.first_frame,
                    PREVIEW_MAX_W,
                    h.max(1),
                    image::imageops::FilterType::Triangle,
                )
            } else {
                probe.first_frame
            };
            self.video_state.texture = imgbuf_to_texture(
                &image::DynamicImage::ImageRgb8(display_rgb).to_rgba8(),
                ui,
            );
        }

        self.video_state.progress_bar = self
            .selected_videos
            .get(idx)
            .map(|pv| pv.frame_progress())
            .unwrap_or(0.0);
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
            let cond2 = !self.selected_videos.is_empty();
            let cond3 = self.feed_url.is_some();
            let cond4 = !self.selected_audios.is_empty();
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
                    let text = self.t(Key::audio_processing);
                    ui.selectable_value(&mut self.mode, Mode::Audio, text);
                }
            });

            if img_mode || video_mode || feed_mode || audio_mode {
                ui.separator();
            }
            egui::ScrollArea::vertical().show(ui, |ui| match self.mode {
                Mode::Image => {
                    self.ui_image(ui);
                }
                Mode::Video => {
                    self.ui_video(ui);
                }
                Mode::Feed => {
                    self.ui_feed(ui);
                }
                Mode::Audio => {
                    self.ui_audio(ui);
                }
            });
        });
    }
}

// ---------- Multi-file navigation chrome (shared by image/audio/video headers) ----------

/// Prev / next buttons + trailing separator. Mutates `index` in place so each
/// caller observes the post-click value immediately (the per-modality status
/// widgets are drawn after this and need the new index).
pub(super) fn nav_prev_next(
    ui: &mut egui::Ui,
    index: &mut usize,
    n: usize,
    prev_hint: &str,
    next_hint: &str,
) {
    if n <= 1 {
        return;
    }
    ui.add_enabled_ui(*index > 1, |ui| {
        if ui.button("⏮").on_hover_text(prev_hint).clicked() {
            *index = index.saturating_sub(1).max(1);
        }
    });
    ui.add_enabled_ui(*index < n, |ui| {
        if ui.button("⏭").on_hover_text(next_hint).clicked() {
            *index = (*index + 1).min(n);
        }
    });
    ui.separator();
}

/// Filename in strong + " · i / n" counter when more than one file is loaded.
pub(super) fn nav_filename(ui: &mut egui::Ui, name: &str, index: usize, n: usize) {
    ui.label(egui::RichText::new(name).strong());
    if n > 1 {
        ui.label(egui::RichText::new(format!("·  {} / {}", index, n)).weak());
    }
}

/// Wide slider for jumping to an arbitrary file. Caller is responsible for
/// `horizontal_wrapped` containment — this is drawn below it, not inside it.
pub(super) fn nav_slider(ui: &mut egui::Ui, index: &mut usize, n: usize) {
    if n <= 1 {
        return;
    }
    // Leave room for the slider's trailing value editor (~70px) plus padding —
    // without this it gets clipped off the right edge.
    let slider_w = (ui.available_width() - 110.0).max(180.0);
    ui.spacing_mut().slider_width = slider_w;
    ui.add(egui::Slider::new(index, 1..=n).text(""));
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

#[derive(Default, PartialEq)]
enum Mode {
    #[default]
    Image,
    Video,
    Feed,
    Audio,
}

/// Bucket a directory's files into (images, audio, video) by extension. Each
/// file lands in at most one bucket; order matches the source iterator so the
/// resulting UI is stable across folder rescans.
fn partition_media_in_dir(
    entries: fs::ReadDir,
) -> (Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>) {
    let mut images = Vec::new();
    let mut audios = Vec::new();
    let mut videos = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if formats::IMAGE_FORMATS
            .iter()
            .any(|f| ext.eq_ignore_ascii_case(f))
        {
            images.push(path);
        } else if formats::AUDIO_FORMATS
            .iter()
            .any(|f| ext.eq_ignore_ascii_case(f))
        {
            audios.push(path);
        } else if formats::VIDEO_FORMATS
            .iter()
            .any(|f| ext.eq_ignore_ascii_case(f))
        {
            videos.push(path);
        }
    }
    (images, audios, videos)
}
