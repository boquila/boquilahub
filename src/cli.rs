use crate::api::{
    abstractions::AI,
    bq::get_bqs,
    eps::LIST_EPS,
    inference::{set_model, set_model2},
    rest::{get_ipv4_address, run_api},
};
use clap::{Args, Parser, Subcommand};
use std::process::exit;

#[derive(Args)]
pub struct ServeArgs {
    /// Model name to deploy
    #[arg(long, value_name = "MODEL_NAME", required = true)]
    pub model: String,

    /// Model name to deploy, complementary classification model
    #[arg(long, value_name = "MODEL_CLS_NAME", required = false)]
    pub model_cls: Option<String>,

    /// Port number for the server
    #[arg(long, value_name = "PORT", default_value = "8791")]
    pub port: u16,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Deploy and serve a model
    Serve(ServeArgs),

    /// Print list of models
    List,
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

                set_model2(&model_cls_path, &LIST_EPS[1]);
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

            set_model(&model_path, &LIST_EPS[1]);

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
            exit(0);
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
    use std::cmp;

    if ais.is_empty() {
        println!("No AI models found.");
        return;
    }

    // Calculate column widths
    let name_width = cmp::max(4, ais.iter().map(|ai| ai.name.len()).max().unwrap_or(0));
    let task_width = cmp::max(4, ais.iter().map(|ai| ai.task.len()).max().unwrap_or(0));
    let arch_width = cmp::max(
        12,
        ais.iter()
            .map(|ai| ai.architecture.as_ref().map_or(0, |s| s.len()))
            .max()
            .unwrap_or(0),
    );
    let classes_width = 8;

    // Header
    println!(
        "┌─{:─<width1$}─┬─{:─<width2$}─┬─{:─<width3$}─┬─{:─<width4$}─┐",
        "",
        "",
        "",
        "",
        width1 = name_width,
        width2 = task_width,
        width3 = arch_width,
        width4 = classes_width
    );

    println!(
        "│ {:^width1$} │ {:^width2$} │ {:^width3$} │ {:^width4$} │",
        "Name",
        "Task",
        "Architecture",
        "Classes",
        width1 = name_width,
        width2 = task_width,
        width3 = arch_width,
        width4 = classes_width
    );

    println!(
        "├─{:─<width1$}─┼─{:─<width2$}─┼─{:─<width3$}─┼─{:─<width4$}─┤",
        "",
        "",
        "",
        "",
        width1 = name_width,
        width2 = task_width,
        width3 = arch_width,
        width4 = classes_width
    );

    // Rows
    for ai in ais {
        let classes_count = ai.classes.len().to_string();
        println!(
            "│ {:width1$} │ {:width2$} │ {:width3$} │ {:>width4$} │",
            ai.name,
            ai.task,
            ai.architecture.as_deref().unwrap_or(""),
            classes_count,
            width1 = name_width,
            width2 = task_width,
            width3 = arch_width,
            width4 = classes_width
        );
    }

    println!(
        "└─{:─<width1$}─┴─{:─<width2$}─┴─{:─<width3$}─┴─{:─<width4$}─┘",
        "",
        "",
        "",
        "",
        width1 = name_width,
        width2 = task_width,
        width3 = arch_width,
        width4 = classes_width
    );
}
