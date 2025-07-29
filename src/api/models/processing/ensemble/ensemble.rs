#[derive(Debug)]
struct SpeciesRecord {
    pub original_line: String, 
    pub uuid: String,
    pub class: String,
    pub order: String,
    pub family: String,
    pub genus: String,
    pub species: String,
    pub common_name: String,
}

impl SpeciesRecord {
    fn new(line: &str) -> Result<SpeciesRecord, ()> {
        let parts: Vec<&str> = line.trim().split(';').collect();
        if parts.len() < 7 {
            return Err(());
        }
        Ok(SpeciesRecord {
            original_line: line.to_string(),
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