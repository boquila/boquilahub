use crate::api::{
    abstractions::AI,
    bq::get_bqs,
    eps::LIST_EPS,
    inference::{set_model, set_model2},
    rest::{get_ipv4_address, run_api},
};
use clap::{Args, Parser, Subcommand};
use std::path::Path;
use tokio::fs as tokio_fs;
use tokio::io::AsyncWriteExt;

#[derive(Args)]
pub struct ServeArgs {
    /// Model name to deploy
    #[arg(value_name = "MODEL_NAME", required = true)]
    pub model: String,

    /// Model name to deploy, complementary classification model
    #[arg(long, value_name = "MODEL_CLS_NAME", required = false)]
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
pub enum Commands {
    /// Deploy and serve a model
    Serve(ServeArgs),

    /// Download a model
    Pull(PullArgs),

    /// Print list of models
    List,
    
    /// Start the GUI, but keeping the terminal
    Gui,
}

#[derive(Parser)]
#[command(
    name = "BoquilaHUB",
    version = "0.3",
    about = "BoquilaHUB - GUI and CLI tool"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

pub async fn run_cli(command: Commands) {
    match command {
        Commands::Serve(args) => {
            let model_name = &args.model;
            let model_name_clean = model_name.strip_suffix(".bq").unwrap_or(model_name);
            let model_path = format!("models/{}.bq", model_name_clean);

            let ais: Vec<AI> = get_bqs();

            if let Some(model_cls_name) = &args.model_cls {
                let model_cls_name_clean =
                    model_cls_name.strip_suffix(".bq").unwrap_or(model_cls_name);
                let model_cls_path = format!("models/{}.bq", model_cls_name_clean);

                let cls_found = ais.iter().any(|ai| ai.get_path().contains(&model_cls_path));
                if !cls_found {
                    panic!(
                        "Model class path '{}' was not found in any of the registered AI paths.\n\
            Make sure that the model '{}' (or '{}.bq') exists in the 'models/' directory",
                        model_cls_path, model_cls_name_clean, model_cls_name_clean
                    );
                }

                let _ = set_model2(&model_cls_path, &LIST_EPS[1], None);
            }

            let port = args.port;

            let found = ais.iter().any(|ai| ai.get_path().contains(&model_path));
            if !found {
                panic!(
                    "Model path '{}' was not found in any of the registered AI paths.\n\
        Make sure that the model '{}' (or '{}.bq') exists in the 'models/' directory",
                    model_path, model_name_clean, model_name_clean
                );
            }

            let _ = set_model(&model_path, &LIST_EPS[1], None);

            let ip_text = format!("http://{}:8791", get_ipv4_address().unwrap());
            println!("{}", ASCII_ART);

            if let Some(model_cls_name) = &args.model_cls {
                println!("Model deployed: {} with {}", model_name, model_cls_name);
            } else {
                println!("Model deployed: {}", model_name);
            }

            println!("IP Address: {}", ip_text);

            let result = run_api(port).await;
            if let Err(e) = result {
                eprintln!("Error running API: {}", e);
            }
        }
        Commands::List => {
            let ais: Vec<AI> = get_bqs();
            print_ais_table(&ais);
            std::process::exit(0);
        }
        Commands::Pull(args) => match pull(&args.model).await {
            Ok(_) => {}
            Err(e) => eprintln!("❌ Failed to pull model {}: {}", &args.model, e),
        },
        Commands::Gui => {
            let _ = crate::gui::run_gui();
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

pub fn print_ais_table(ais: &Vec<AI>) {
    if ais.is_empty() {
        println!("No AI models found.");
        return;
    }

    // Calculate column widths
    let name_width = std::cmp::max(4, ais.iter().map(|ai| ai.name.len()).max().unwrap_or(0));
    let task_width = std::cmp::max(4, ais.iter().map(|ai| ai.task.len()).max().unwrap_or(0));
    let arch_width = std::cmp::max(
        12,
        ais.iter()
            .map(|ai| ai.architecture.as_ref().map_or(0, |s| s.len()))
            .max()
            .unwrap_or(0),
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
            ai.task,
            ai.architecture.as_deref().unwrap_or(""),
            ai.classes.len()
        );
    }

    print_border("└", "┴", "┘");
}

async fn pull(model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Searching for model '{}'...", model_name);

    // Fetch the JSON index
    let models = crate::api::pull::get_list()
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
