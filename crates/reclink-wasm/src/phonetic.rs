//! Phonetic encoding functions exposed to JavaScript.

use reclink_core::phonetic::*;
use wasm_bindgen::prelude::*;

fn algorithm_from_name(name: &str) -> Result<PhoneticAlgorithm, JsError> {
    match name {
        "soundex" => Ok(PhoneticAlgorithm::Soundex(Soundex)),
        "metaphone" => Ok(PhoneticAlgorithm::Metaphone(Metaphone)),
        "double_metaphone" => Ok(PhoneticAlgorithm::DoubleMetaphone(DoubleMetaphone)),
        "nysiis" => Ok(PhoneticAlgorithm::Nysiis(Nysiis)),
        "caverphone" => Ok(PhoneticAlgorithm::Caverphone(Caverphone)),
        "cologne" => Ok(PhoneticAlgorithm::ColognePhonetic(ColognePhonetic)),
        "beider_morse" => Ok(PhoneticAlgorithm::BeiderMorse(BeiderMorse::default())),
        "phonex" => Ok(PhoneticAlgorithm::Phonex(Phonex)),
        "match_rating" => Ok(PhoneticAlgorithm::MatchRatingApproach(MatchRatingApproach)),
        "daitch_mokotoff" => Ok(PhoneticAlgorithm::DaitchMokotoff(DaitchMokotoff)),
        _ => Err(JsError::new(&format!(
            "unknown phonetic algorithm: '{name}'"
        ))),
    }
}

/// Encode a string using the named phonetic algorithm.
#[wasm_bindgen]
pub fn phonetic_encode(s: &str, algorithm: &str) -> Result<String, JsError> {
    let alg = algorithm_from_name(algorithm)?;
    Ok(alg.encode(s))
}

/// Check if two strings match phonetically using the named algorithm.
#[wasm_bindgen]
pub fn phonetic_match(a: &str, b: &str, algorithm: &str) -> Result<bool, JsError> {
    let alg = algorithm_from_name(algorithm)?;
    Ok(alg.is_match(a, b))
}

/// List all available phonetic algorithm names.
#[wasm_bindgen]
pub fn list_phonetic_algorithms() -> JsValue {
    let names = vec![
        "soundex",
        "metaphone",
        "double_metaphone",
        "nysiis",
        "caverphone",
        "cologne",
        "beider_morse",
        "phonex",
        "match_rating",
        "daitch_mokotoff",
    ];
    serde_wasm_bindgen::to_value(&names).unwrap()
}
