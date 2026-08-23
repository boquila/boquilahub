use std::collections::HashMap;

use boquilahub::api::abstractions::Prob;
use boquilahub::api::processing::post::{apply_geofence_filter, apply_label_rollup, SpeciesRecord};

const OCELOT: &str =
    "22976d14-d424-4f18-a67a-d8e1689cefcc;mammalia;carnivora;felidae;leopardus;pardalis;ocelot";
const INVALID: &str = "not a taxonomy label";

#[test]
fn test_species_record_to_taxonomic_string() {
    let record = SpeciesRecord::new("23a6f03b-b3d0-471b-a67d-88f10cb64e59;amphibia;;;;;amphibian").unwrap();        
    assert_eq!(record.to_taxonomic_string(), "amphibia;;;;");

    let record = SpeciesRecord::new("22976d14-d424-4f18-a67a-d8e1689cefcc;mammalia;carnivora;felidae;leopardus;pardalis;ocelot").unwrap();        
    assert_eq!(record.to_taxonomic_string(), "mammalia;carnivora;felidae;leopardus;pardalis");
}

fn probs(items: &[(&str, f32, u32)]) -> Vec<Prob> {
    items
        .iter()
        .map(|(label, prob, id)| Prob::new(label.to_string(), *prob, *id))
        .collect()
}

#[test]
fn rollup_all_invalid_labels_returns_unchanged() {
    let mut input = probs(&[(INVALID, 0.9, 0), ("still not;valid", 0.8, 1)]);
    let before = input.clone();
    apply_label_rollup(&mut input, 0.5);
    assert_eq!(input.len(), before.len());
    for (a, b) in input.iter().zip(before.iter()) {
        assert_eq!(a.label, b.label);
        assert_eq!(a.prob, b.prob);
        assert_eq!(a.class_id, b.class_id);
    }
}

#[test]
fn rollup_mixed_input_considers_only_valid_records() {
    // The invalid label has the highest raw confidence but must be ignored:
    // the only valid record wins the species decision.
    let mut input = probs(&[(INVALID, 0.95, 1), (OCELOT, 0.6, 0)]);
    apply_label_rollup(&mut input, 0.5);
    assert_eq!(input.len(), 1);
    assert_eq!(input[0].label, "leopardus pardalis (ocelot)");
    assert_eq!(input[0].class_id, 0);
}

#[test]
fn geofence_all_invalid_labels_returns_unchanged() {
    let mut geofence = HashMap::new();
    geofence.insert(
        "mammalia;carnivora;felidae;leopardus;pardalis".to_string(),
        vec!["MX".to_string()],
    );
    let mut input = probs(&[(INVALID, 0.9, 0)]);
    let before = input.clone();
    apply_geofence_filter(&mut input, &geofence, "MX");
    assert_eq!(input[0].label, before[0].label);
}

#[test]
fn geofence_matching_country_keeps_species_level() {
    let mut geofence = HashMap::new();
    geofence.insert(
        "mammalia;carnivora;felidae;leopardus;pardalis".to_string(),
        vec!["MX".to_string()],
    );
    let mut input = probs(&[(OCELOT, 0.9, 0)]);
    apply_geofence_filter(&mut input, &geofence, "MX");
    assert_eq!(input[0].label, OCELOT);
}

#[test]
fn geofence_non_matching_rolls_up_until_match() {
    // Country only matches at the family level: the label must be rolled
    // up (species and genus dropped) until the taxonomic string hits it.
    let mut geofence = HashMap::new();
    geofence.insert(
        "mammalia;carnivora;felidae;;".to_string(),
        vec!["BR".to_string()],
    );
    let mut input = probs(&[(OCELOT, 0.9, 0)]);
    apply_geofence_filter(&mut input, &geofence, "BR");
    assert_eq!(input[0].label, "22976d14-d424-4f18-a67a-d8e1689cefcc;mammalia;carnivora;felidae;;;");
}

#[test]
fn geofence_no_match_anywhere_rolls_up_to_class() {
    let mut geofence = HashMap::new();
    geofence.insert("aves;;;;".to_string(), vec!["BR".to_string()]);
    let mut input = probs(&[(OCELOT, 0.9, 0)]);
    apply_geofence_filter(&mut input, &geofence, "BR");
    // Exhausted rollups: species..order dropped, stops at class.
    assert_eq!(input[0].label, "22976d14-d424-4f18-a67a-d8e1689cefcc;mammalia;;;;;");
}
