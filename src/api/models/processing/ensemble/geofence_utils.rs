use std::collections::HashMap;

type RegionMap = HashMap<String, Vec<String>>;
type GeofenceMap = HashMap<String, HashMap<String, RegionMap>>;

fn get_full_class_string(label: &str) -> String {
    // Placeholder for actual logic to generate full class string.
    label.to_string()
}

fn should_geofence_animal_classification(
    label: &str,
    country: Option<&str>,
    admin1_region: Option<&str>,
    geofence_map: &GeofenceMap,
    enable_geofence: bool,
) -> bool {
    if !enable_geofence {
        return false;
    }

    let country = match country {
        Some(c) => c,
        None => return false,
    };

    let full_class_string = get_full_class_string(label);

    let class_rules = match geofence_map.get(&full_class_string) {
        Some(rules) => rules,
        None => return false,
    };

    // Check "allow" rules
    if let Some(allow_countries) = class_rules.get("allow") {
        if !allow_countries.contains_key(country) {
            return true;
        }

        if let (Some(admin_region), Some(allowed_regions)) =
            (admin1_region, allow_countries.get(country))
        {
            if !allowed_regions.is_empty() && !allowed_regions.contains(&admin_region.to_string()) {
                return true;
            }
        }
    }

    // Check "block" rules
    if let Some(block_countries) = class_rules.get("block") {
        if let Some(blocked_regions) = block_countries.get(country) {
            if blocked_regions.is_empty() {
                return true;
            }

            if let Some(admin_region) = admin1_region {
                if blocked_regions.contains(&admin_region.to_string()) {
                    return true;
                }
            }
        }
    }

    false
}
