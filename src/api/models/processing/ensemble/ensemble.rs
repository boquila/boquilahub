#[derive(Debug)]
pub struct SpeciesRecord {
    pub uuid: String,
    pub class: String,
    pub order: String,
    pub family: String,
    pub genus: String,
    pub species: String,
    pub common_name: String,
}

impl SpeciesRecord {
    pub fn new(line: &str) -> Result<SpeciesRecord, ()> {
        let parts: Vec<&str> = line.trim().split(';').collect();
        if parts.len() < 7 {
            return Err(());
        }
        Ok(SpeciesRecord {
            uuid: parts[0].to_string(),
            class: parts[1].to_string(),
            order: parts[2].to_string(),
            family: parts[3].to_string(),
            genus: parts[4].to_string(),
            species: parts[5].to_string(),
            common_name: parts[6].to_string(),
        })
    }
}

pub fn get_uuid(line: &str) -> String {
    line.split(';').nth(0).unwrap().to_string()
}

pub fn get_class(line: &str) -> String {
    line.split(';').nth(1).unwrap().to_string()
}

pub fn get_order(line: &str) -> String {
    line.split(';').nth(2).unwrap().to_string()
}

pub fn get_family(line: &str) -> String {
    line.split(';').nth(3).unwrap().to_string()
}

pub fn get_genus(line: &str) -> String {
    line.split(';').nth(4).unwrap().to_string()
}

pub fn get_species(line: &str) -> String {
    line.split(';').nth(5).unwrap().to_string()
}

pub fn get_common_name(line: &str) -> String {
    line.split(';').nth(6).unwrap().to_string()
}
