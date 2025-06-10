#[derive(PartialEq)]
pub enum Lang {
    EN,
    ES,
}

#[allow(non_camel_case_types)]
pub enum Key {
    select_ai,
    select_ep,
    setup,
    deploy,
    deployed_api,
    select_your_data,
    folder,
    image,
    video_file,
    camera_feed,
    about,
    idiom,
    models,
    donate,
    source_code,
    analyze,
    export,
    analysis,
    cancel,
    process_completed,
    done,
    error_ocurred,
    website,
    model_hub,
    api,
    image_processing,
    video_processing,
    feed_processing,
    model_hub_url,
    donate_url,
    website_url,
    export_predictions,
    export_imgs_with_predictions,
    copy_with_classification,
    input_feed_url,
    example,
    every_how_many_frames,
}

pub fn translate(key: Key, lang: &Lang) -> &'static str {
    match key {
        Key::select_ai => match lang {
            Lang::EN => "Select an AI",
            Lang::ES => "Selecciona una IA",
        },
        Key::select_ep => match lang {
            Lang::EN => "Select a processor",
            Lang::ES => "Selecciona un procesador",
        },
        Key::setup => match lang {
            Lang::EN => "Setup",
            Lang::ES => "Configuración",
        },
        Key::deploy => match lang {
            Lang::EN => "Deploy",
            Lang::ES => "Desplegar",
        },
        Key::deployed_api => match lang {
            Lang::EN => "Deployed API",
            Lang::ES => "API desplegada",
        },
        Key::select_your_data => match lang {
            Lang::EN => "Select your data",
            Lang::ES => "Selecciona tus datos",
        },
        Key::folder => match lang {
            Lang::EN => "Folder",
            Lang::ES => "Carpeta",
        },
        Key::image => match lang {
            Lang::EN => "Image",
            Lang::ES => "Imagen",
        },
        Key::video_file => match lang {
            Lang::EN => "Video",
            Lang::ES => "Video",
        },
        Key::camera_feed => match lang {
            Lang::EN => "Feed",
            Lang::ES => "Cámara",
        },
        Key::about => match lang {
            Lang::EN => "About",
            Lang::ES => "Información",
        },
        Key::idiom => match lang {
            Lang::EN => "Language",
            Lang::ES => "Idioma",
        },
        Key::models => match lang {
            Lang::EN => "Models",
            Lang::ES => "Modelos",
        },
        Key::donate => match lang {
            Lang::EN => "Donate",
            Lang::ES => "Donar",
        },
        Key::source_code => match lang {
            Lang::EN => "Source code",
            Lang::ES => "Código fuente",
        },
        Key::analyze => match lang {
            Lang::EN => "Analyze",
            Lang::ES => "Analizar",
        },
        Key::export => match lang {
            Lang::EN => "Export",
            Lang::ES => "Exportar",
        },
        Key::analysis => match lang {
            Lang::EN => "Analysis",
            Lang::ES => "Análisis",
        },
        Key::cancel => match lang {
            Lang::EN => "Cancel",
            Lang::ES => "Cancelar",
        },
        Key::process_completed => match lang {
            Lang::EN => "Process completed",
            Lang::ES => "Proceso completado",
        },
        Key::done => match lang {
            Lang::EN => "✅ Done",
            Lang::ES => "✅ Listo",
        },
        Key::error_ocurred => match lang {
            Lang::EN => "Error ocurred",
            Lang::ES => "Ocurrió un error",
        },
        Key::website => match lang {
            Lang::EN => "Website",
            Lang::ES => "Sitio web",
        },
        Key::model_hub => match lang {
            Lang::EN => "Model HUB",
            Lang::ES => "HUB de Modelos",
        },
        Key::api => match lang {
            _ => "API",
        },
        Key::image_processing => match lang {
            Lang::EN => "Images processing",
            Lang::ES => "Procesamiento de imágenes",
        },
        Key::video_processing => match lang {
            Lang::EN => "Video processing",
            Lang::ES => "Procesamiento de video",
        },
        Key::feed_processing => match lang {
            Lang::EN => "Feed processing",
            Lang::ES => "Procesamiento en vivo",
        },
        // check boquila.org to see all available languages, there's quite a few
        Key::model_hub_url => match lang {
            Lang::EN => "https://boquila.org/hub",
            Lang::ES => "https://boquila.org/es/hub",
        },
        Key::website_url => match lang {
            Lang::EN => "https://boquila.org/en",
            Lang::ES => "https://boquila.org/",
        },
        Key::donate_url => match lang {
            Lang::EN => "https://boquila.org/donate",
            Lang::ES => "https://boquila.org/donar",
        },
        Key::export_predictions => match lang {
            Lang::EN => "Export predictions (.txt)",
            Lang::ES => "Exportar predicciones (.txt)",
        },
        Key::export_imgs_with_predictions => match lang {
            Lang::EN => "Export images with predictions (.jpg)",
            Lang::ES => "Exportar imágenes con predicciones (.jpg)",
        },
        Key::copy_with_classification => match lang {
            Lang::EN => "Copy and separate in folders according to classification",
            Lang::ES => "Copiar y separar en carpetas según clasificación",
        },
        Key::input_feed_url => match lang {
            Lang::EN => "Add the URL",
            Lang::ES => "Ingresa la dirección URL",
        },
        Key::example => match lang {
            Lang::EN => "Example",
            Lang::ES => "Ejemplo",
        },
        // 1 = every single frame, 2 = every second frame, and so on.
        Key::every_how_many_frames => match lang {
            Lang::EN => "Analyze every how many frames?",
            Lang::ES => "¿Cada cuántos frames quiere analizar?",
        },
    }
}
