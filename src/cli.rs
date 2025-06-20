use clap::{Arg, Command};

use crate::api::{
    abstractions::AI,
    bq::get_bqs,
    eps::LIST_EPS,
    inference::set_model,
    rest::{get_ipv4_address, run_api},
};

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
                    Arg::new("port")
                        .long("port")
                        .help("Port number for the server")
                        .value_name("PORT")
                        .default_value("8791")
                        .value_parser(clap::value_parser!(u16)),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("serve", _sub_matches)) => {
            let model_name = matches.get_one::<String>("model").unwrap();
            let port = *matches.get_one::<u16>("port").unwrap();

            let model_path = format!(
                "models/{}.bq",
                model_name.strip_suffix(".bq").unwrap_or(model_name)
            );
            let ais: Vec<AI> = get_bqs();
            let found = ais.iter().any(|ai| ai.get_path().contains(&model_path));

            if !found {
                panic!(
                    "Model path '{}' was not found in any of the registered AI paths.\n\
        Make sure that the model '{}' (or '{}.bq') exists in the 'models/' directory",
                    model_path,
                    model_name.strip_suffix(".bq").unwrap_or(model_name),
                    model_name.strip_suffix(".bq").unwrap_or(model_name)
                );
            }

            set_model(&model_path, &LIST_EPS[1]);
            run_api(port).await;
            // CLI mode

            let ip_text = format!("http://{}:8791", get_ipv4_address().unwrap());
            println!("{}", ASCII_ART);
            println!("Model deployed: {}", model_name);
            println!("IP Address: {}", ip_text);
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
