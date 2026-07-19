use crate::api::{
    bq::{AIMetadata, BQModel, Ep, GlobalBQ, Modality},
    rest::{get_ipv4_address, run_api},
};
use clap::{Args, Parser, Subcommand};
use std::path::Path;
use tokio::fs as tokio_fs;
use tokio::io::AsyncWriteExt;

#[derive(Args)]
pub struct ServeArgs {
    /// Model name to deploy
    #[arg(value_name = "MODEL_PATH", required = true)]
    pub model: String,

    /// Model name to deploy, complementary classification model
    #[arg(long, value_name = "MODEL_CLS_PATH", required = false)]
    pub model_cls: Option<String>,

    /// Port number for the server
    #[arg(long, value_name = "PORT", default_value = "8791")]
    pub port: u16,
}

#[derive(Args)]
pub struct PullArgs {
    /// Model name to pull
    #[arg(value_name = "MODEL_NAME", required = true)]
    pub model: String,
}

#[derive(Subcommand)]
pub enum BqCommands {
    /// Create a new .bq model.
    /// Pass "name" to use "name.json" and "name.onnx" and create "name.bq"
    New { name: String },

    /// Returns the shape of a .bq model
    Shape { name: String },
}

#[derive(Subcommand)]
pub enum Commands {
    /// Deploy and serve a model
    Serve(ServeArgs),

    /// Download a model
    Pull(PullArgs),

    /// Print list of models
    List,

    /// Start the GUI, while keeping the terminal
    Gui,

    /// Start the TUI
    Tui {
        /// Language override (en, es, fr, de, zh, ja, pt, vi)
        #[arg(long)]
        lang: Option<String>,
    },

    /// Utils for .bq models (for devs)
    Bq {
        #[command(subcommand)]
        command: BqCommands,
    },
}

#[derive(Parser)]
#[command(
    name = "BoquilaHUB",
    version = "0.6",
    about = "BoquilaHUB - AIs for Nature"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

impl Cli {
    pub async fn run(self) {
        match self.command.expect("Could not run CLI") {
            Commands::Serve(args) => {
                let ais: Vec<AIMetadata> = BQModel::get_list();
                let model = resolve_model(&args.model, &ais);

                if model.modality == Modality::Audio {
                    panic!(
                        "Audio models cannot be deployed as API. Model '{}' is an audio model.",
                        model.name
                    );
                }

                if let Some(cls_name) = &args.model_cls {
                    let cls = resolve_model(cls_name, &ais);
                    let _ = GlobalBQ::Second.set_model(&cls.get_path(), Ep::gpu(), None);
                }

                let _ = GlobalBQ::First.set_model(&model.get_path(), Ep::gpu(), None);

                println!("{}", ASCII_ART);
                match &args.model_cls {
                    Some(cls) => println!("Model deployed: {} with {}", model.name, cls),
                    None => println!("Model deployed: {}", model.name),
                }
                println!("IP Address: http://{}:8791", get_ipv4_address().unwrap());

                if let Err(e) = run_api(args.port).await {
                    eprintln!("Error running API: {}", e);
                }
            }
            Commands::List => {
                let ais: Vec<AIMetadata> = BQModel::get_list();
                print_ais_table(&ais);
                std::process::exit(0);
            }
            Commands::Pull(args) => match pull(&args.model).await {
                Ok(_) => {}
                Err(e) => eprintln!("❌ Failed to pull model {}: {}", &args.model, e),
            },
            Commands::Gui => {
                let _ = crate::gui::Gui::run();
            }
            Commands::Tui { lang } => {
                let language = crate::localization::Lang::from_optional_str(lang.as_deref());
                let _ = crate::tui::Tui::run(language);
            }
            Commands::Bq { command } => match command {
                BqCommands::Shape { name } => match BQModel::from_file_print_shape(&name) {
                    Ok(_) => {}
                    Err(e) => eprintln!("{}", e),
                },
                BqCommands::New { name } => match BQModel::create_bq_file(name) {
                    Ok(_) => {}
                    Err(e) => eprintln!("{}", e),
                }
            },
        }
    }
}

const ASCII_ART: &'static str = r#"

 /$$$$$$$                                /$$ /$$           /$$   /$$ /$$   /$$ /$$$$$$$
| $$__  $$                              |__/| $$          | $$  | $$| $$  | $$| $$__  $$
| $$  \ $$  /$$$$$$   /$$$$$$  /$$   /$$ /$$| $$  /$$$$$$ | $$  | $$| $$  | $$| $$  \ $$
| $$$$$$$  /$$__  $$ /$$__  $$| $$  | $$| $$| $$ |____  $$| $$$$$$$$| $$  | $$| $$$$$$$
| $$__  $$| $$  \ $$| $$  \ $$| $$  | $$| $$| $$  /$$$$$$$| $$__  $$| $$  | $$| $$__  $$
| $$  \ $$| $$  | $$| $$  | $$| $$  | $$| $$| $$ /$$__  $$| $$  | $$| $$  | $$| $$  \ $$
| $$$$$$$/|  $$$$$$/|  $$$$$$$|  $$$$$$/| $$| $$|  $$$$$$$| $$  | $$|  $$$$$$/| $$$$$$$/
|_______/  \______/  \____  $$ \______/ |__/|__/ \_______/|__/  |__/ \______/ |_______/
                          | $$
                          | $$
                          |__/                                      AI for Biodiversity

"#;

fn resolve_model<'a>(name: &str, ais: &'a [AIMetadata]) -> &'a AIMetadata {
    let clean = name.strip_suffix(".bq").unwrap_or(name);
    ais.iter().find(|ai| ai.name == clean).unwrap_or_else(|| {
        panic!(
            "Model '{0}' (or '{0}.bq') was not found in the 'models/' directory",
            clean
        )
    })
}

pub fn print_ais_table(ais: &[AIMetadata]) {
    if ais.is_empty() {
        println!("No AI models found.");
        return;
    }

    // Calculate column widths
    let name_width = std::cmp::max(4, ais.iter().map(|ai| ai.name.len()).max().unwrap_or(0));
    let task_width = std::cmp::max(4, ais.iter().map(|ai| ai.task.name().len()).max().unwrap_or(0));
    let arch_width = std::cmp::max(
        12,
        ais.iter().map(|ai| ai.architecture.len()).max().unwrap_or(0),
    );
    let classes_width = 8;

    let widths = [name_width, task_width, arch_width, classes_width];

    // Helper to print borders
    let print_border = |left: &str, mid: &str, right: &str| {
        print!("{}", left);
        for (i, &w) in widths.iter().enumerate() {
            print!("{:─<width$}", "", width = w + 2);
            print!("{}", if i < widths.len() - 1 { mid } else { right });
        }
        println!();
    };

    // Table
    print_border("┌", "┬", "┐");
    println!(
        "│ {:^name_width$} │ {:^task_width$} │ {:^arch_width$} │ {:^classes_width$} │",
        "Name", "Task", "Architecture", "Classes"
    );
    print_border("├", "┼", "┤");

    for ai in ais {
        println!(
            "│ {:name_width$} │ {:task_width$} │ {:arch_width$} │ {:>classes_width$} │",
            ai.name,
            ai.task.name(),
            ai.architecture,
            ai.classes.len()
        );
    }

    print_border("└", "┴", "┘");
}

async fn pull(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Searching for model '{}'...", model_name);

    // Fetch the JSON index
    let models = BQModel::get_list_from_api()
        .await
        .map_err(|e| format!("Failed to fetch model index: {}", e))?;

    // Find the requested model
    let model = models
        .into_iter()
        .find(|m| m.name == model_name)
        .ok_or_else(|| format!("Model '{}' not found in the registry", model_name))?;

    println!("Model found, starting download...");
    println!("Downloading from: {}", model.download_link);

    // Ensure models/ directory exists
    tokio_fs::create_dir_all("models")
        .await
        .map_err(|e| format!("Failed to create models directory: {}", e))?;

    // Extract filename from URL
    let filename = Path::new(&model.download_link)
        .file_name()
        .ok_or("Invalid download URL: cannot extract filename")?
        .to_string_lossy();
    let file_path = format!("models/{}", filename);

    // Download the file
    println!("Downloading to '{}'...", file_path);
    let response = reqwest::get(&model.download_link)
        .await
        .map_err(|e| format!("Failed to download model: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Download failed with status: {} ({})",
            response.status().as_u16(),
            response
                .status()
                .canonical_reason()
                .unwrap_or("Unknown error")
        )
        .into());
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read downloaded content: {}", e))?;

    // Save the file
    let mut file = tokio_fs::File::create(&file_path)
        .await
        .map_err(|e| format!("Failed to create file '{}': {}", file_path, e))?;

    file.write_all(&bytes)
        .await
        .map_err(|e| format!("Failed to write to file '{}': {}", file_path, e))?;

    println!("File size: {:.2} MB", bytes.len() as f64 / 1_048_576.0);

    Ok(())
}
