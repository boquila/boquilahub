use std::collections::HashMap;

type RegionMap = HashMap<String, Vec<String>>;
type GeofenceMap = HashMap<String, HashMap<String, RegionMap>>;

fn get_full_class_string(label: &str) -> String {
    // Placeholder for actual logic to generate full class string.
    label.to_string()
}

/// Determines whether an animal classification should be geofenced based on
/// provided location and geofencing rules.
///
/// # Arguments
///
/// * `label` - The label of the animal to check geofencing rules for.
/// * `country` - Optional ISO 3166-1 alpha-3 country code where the prediction occurred.
/// * `admin1_region` - Optional ISO 3166-2 first-level administrative region code.
/// * `geofence_map` - A nested map containing geofencing rules:
///     - Outer key: full class string
///     - Inner keys: "allow" and/or "block"
///     - Inner values: map of country codes to allowed or blocked regions.
/// * `enable_geofence` - Boolean indicating whether geofencing is enabled.
///
/// # Returns
///
/// * `true` if the animal classification should be geofenced based on the rules.
/// * `false` otherwise.
///
/// # Behavior
///
/// - Returns `false` immediately if geofencing is disabled or if the country is not provided.
/// - If an "allow" list exists for the class and country is not explicitly allowed, returns `true`.
/// - If admin1_region is provided but not in the allowed list for the country, returns `true`.
/// - If a "block" list exists and the entire country or the specific admin1_region is blocked, returns `true`.
/// - Returns `false` if no rules require geofencing for the given inputs.
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
