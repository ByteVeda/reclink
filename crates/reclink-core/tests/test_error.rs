use reclink_core::error::ReclinkError;

#[test]
fn unequal_length_display() {
    let err = ReclinkError::UnequalLength { a: 3, b: 5 };
    assert_eq!(
        err.to_string(),
        "strings must have equal length: got 3 and 5"
    );
}

#[test]
fn missing_field_display() {
    let err = ReclinkError::MissingField("name".into());
    assert_eq!(err.to_string(), "missing field: name");
}

#[test]
fn type_mismatch_display() {
    let err = ReclinkError::TypeMismatch {
        field: "age".into(),
        expected: "Integer".into(),
        got: "Text".into(),
    };
    assert_eq!(
        err.to_string(),
        "type mismatch for field `age`: expected Integer, got Text"
    );
}

#[test]
fn invalid_config_display() {
    let err = ReclinkError::InvalidConfig("bad threshold".into());
    assert_eq!(err.to_string(), "invalid configuration: bad threshold");
}

#[test]
fn pipeline_error_display() {
    let err = ReclinkError::Pipeline("missing blocker".into());
    assert_eq!(err.to_string(), "pipeline error: missing blocker");
}

#[test]
fn empty_input_display() {
    let err = ReclinkError::EmptyInput("no records".into());
    assert_eq!(err.to_string(), "empty input: no records");
}
