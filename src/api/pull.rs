use anyhow::Result;
use serde::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct AvailableModel {
    pub name: String,
    pub description: String,
    pub download_link: String,
}

pub async fn get_list() -> Result<Vec<AvailableModel>> {
    let url = "https://boquila.org/api/models.json";
    let listmodels: Vec<AvailableModel> = reqwest::get(url).await?.json().await?;
    Ok(listmodels)
}
