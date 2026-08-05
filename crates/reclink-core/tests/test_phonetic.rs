use reclink_core::phonetic::{
    DoubleMetaphone, Metaphone, Nysiis, PhoneticAlgorithm, PhoneticEncoder, Soundex,
};

#[test]
fn phonetic_algorithm_enum_dispatch() {
    let algorithms: Vec<(PhoneticAlgorithm, &str)> = vec![
        (PhoneticAlgorithm::Soundex(Soundex), "Robert"),
        (PhoneticAlgorithm::Metaphone(Metaphone), "Robert"),
        (
            PhoneticAlgorithm::DoubleMetaphone(DoubleMetaphone),
            "Robert",
        ),
        (PhoneticAlgorithm::Nysiis(Nysiis), "Robert"),
    ];
    for (algo, input) in &algorithms {
        let code = algo.encode(input);
        assert!(
            !code.is_empty(),
            "{algo:?} should produce non-empty encoding"
        );
    }
}

#[test]
fn phonetic_encoder_encode_all() {
    let encoder = Soundex;
    let inputs = vec!["Smith", "Smyth", "Jones"];
    let codes = encoder.encode_all(&inputs);
    assert_eq!(codes.len(), 3);
    // Smith and Smyth should have the same Soundex code
    assert_eq!(codes[0], codes[1]);
    // Jones should differ
    assert_ne!(codes[0], codes[2]);
}

#[test]
fn phonetic_encoder_is_match() {
    let encoder = Metaphone;
    // "Smith" and "Smyth" should have the same Metaphone encoding
    assert!(encoder.is_match("Smith", "Smyth"));
    // "Smith" and "Jones" should not
    assert!(!encoder.is_match("Smith", "Jones"));
}
