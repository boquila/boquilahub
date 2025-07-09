struct SpeciesRecord {
    uuid: String,
    class: Option<String>,
    order: Option<String>,
    family: Option<String>,
    genus: Option<String>,
    species: Option<String>,
    common_name: Option<String>,
}

impl SpeciesRecord {
    fn new(
        uuid: String,
        class_field: Option<String>,
        order: Option<String>,
        family: Option<String>,
        genus: Option<String>,
        species: Option<String>,
        common_name: Option<String>,
    ) -> Self {
        SpeciesRecord {
            uuid,
            class: class_field,
            order,
            family,
            genus,
            species,
            common_name,
        }
    }

    fn parse(line: &str) -> Option<SpeciesRecord> {
        let parts: Vec<&str> = line.split(';').collect();

        if parts.len() != 7 {
            return None;
        }

        let uuid = parts[0].to_string();

        let class_field = Some(parts[1].to_string()).filter(|s| !s.is_empty());
        let order = Some(parts[2].to_string()).filter(|s| !s.is_empty());
        let family = Some(parts[3].to_string()).filter(|s| !s.is_empty());
        let genus = Some(parts[4].to_string()).filter(|s| !s.is_empty());
        let species = Some(parts[5].to_string()).filter(|s| !s.is_empty());
        let common_name = Some(parts[6].to_string()).filter(|s| !s.is_empty());

        Some(SpeciesRecord::new(uuid, class_field, order, family, genus, species, common_name))
    }
}