use crate::api::{
    abstractions::AI,
    bq::get_bqs,
    eps::LIST_EPS,
    inference::{set_model, set_model2},
    rest::{get_ipv4_address, run_api},
};
use clap::{Arg, Command};

pub async fn run_cli() {
    let matches = Command::new("BoquilaHUB")
        .version("0.3")
        .about("BoquilaHUB - GUI and CLI tool")
        .subcommand(
            Command::new("serve")
                .about("Deploy and serve a model")
                .arg(
                    Arg::new("model")
                        .long("model")
                        .help("Model name to deploy")
                        .value_name("MODEL_NAME")
                        .required(true),
                )
                .arg(
                    Arg::new("model_cls")
                        .long("model_cls")
                        .help("Model name to deploy, complementary classification model")
                        .value_name("MODEL_CLS_NAME")
                        .required(false),
                )
                .arg(
                    Arg::new("port")
                        .long("port")
                        .help("Port number for the server")
                        .value_name("PORT")
                        .default_value("8791")
                        .value_parser(clap::value_parser!(u16)),
                ),
        )
        .subcommand(Command::new("list").about("Print list of models"))
        .get_matches();

    match matches.subcommand() {
        Some(("serve", sub_matches)) => {
            let model_name = sub_matches.get_one::<String>("model").unwrap();
            let model_name_clean = model_name.strip_suffix(".bq").unwrap_or(model_name);
            let model_path = format!("models/{}.bq", model_name_clean);

            let ais: Vec<AI> = get_bqs();

            if let Some(model_cls_name) = sub_matches.get_one::<String>("model_cls") {
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

            let port = *sub_matches.get_one::<u16>("port").unwrap();

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
            if let Some(model_cls_name) = sub_matches.get_one::<String>("model_cls") {
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
        Some(("list", _sub_matches)) => {
            let ais: Vec<AI> = get_bqs();
            print_ais_table(&ais);
            std::process::exit(0);
        }
        _ => {}
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
            .map(|ai| ai.architecture.len())
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
            ai.architecture,
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
