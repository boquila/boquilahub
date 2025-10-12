#[cfg(test)]
mod tests {
    use boquilahub::api::processing::post::SpeciesRecord;
    
    #[test]
    fn test_species_record_to_taxonomic_string() {
        let record = SpeciesRecord::new("23a6f03b-b3d0-471b-a67d-88f10cb64e59;amphibia;;;;;amphibian").unwrap();        
        assert_eq!(record.to_taxonomic_string(), "amphibia;;;;");

        let record = SpeciesRecord::new("22976d14-d424-4f18-a67a-d8e1689cefcc;mammalia;carnivora;felidae;leopardus;pardalis;ocelot").unwrap();        
        assert_eq!(record.to_taxonomic_string(), "mammalia;carnivora;felidae;leopardus;pardalis");
    }
}