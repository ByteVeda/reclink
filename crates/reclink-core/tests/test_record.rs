use reclink_core::record::{FieldValue, Record, RecordBatch};

#[test]
fn field_value_as_text() {
    assert_eq!(FieldValue::Text("hello".into()).as_text(), Some("hello"));
    assert_eq!(FieldValue::Integer(42).as_text(), None);
    assert_eq!(FieldValue::Float(1.5).as_text(), None);
    assert_eq!(FieldValue::Date("2024-01-01".into()).as_text(), None);
    assert_eq!(FieldValue::Null.as_text(), None);
}

#[test]
fn field_value_as_integer() {
    assert_eq!(FieldValue::Integer(42).as_integer(), Some(42));
    assert_eq!(FieldValue::Text("hello".into()).as_integer(), None);
    assert_eq!(FieldValue::Float(1.5).as_integer(), None);
    assert_eq!(FieldValue::Null.as_integer(), None);
}

#[test]
fn field_value_as_float() {
    assert_eq!(FieldValue::Float(3.14).as_float(), Some(3.14));
    assert_eq!(FieldValue::Text("hello".into()).as_float(), None);
    assert_eq!(FieldValue::Integer(42).as_float(), None);
    assert_eq!(FieldValue::Null.as_float(), None);
}

#[test]
fn field_value_is_null() {
    assert!(FieldValue::Null.is_null());
    assert!(!FieldValue::Text("x".into()).is_null());
    assert!(!FieldValue::Integer(0).is_null());
    assert!(!FieldValue::Float(0.0).is_null());
    assert!(!FieldValue::Date("2024-01-01".into()).is_null());
}

#[test]
fn field_value_display() {
    assert_eq!(FieldValue::Text("hello".into()).to_string(), "hello");
    assert_eq!(FieldValue::Integer(42).to_string(), "42");
    assert_eq!(FieldValue::Float(3.14).to_string(), "3.14");
    assert_eq!(
        FieldValue::Date("2024-01-01".into()).to_string(),
        "2024-01-01"
    );
    assert_eq!(FieldValue::Null.to_string(), "NULL");
}

#[test]
fn record_new_with_field_get() {
    let record = Record::new("r1")
        .with_field("name", FieldValue::Text("Alice".into()))
        .with_field("age", FieldValue::Integer(30));

    assert_eq!(record.id, "r1");
    assert_eq!(record.get("name"), Some(&FieldValue::Text("Alice".into())));
    assert_eq!(record.get("age"), Some(&FieldValue::Integer(30)));
    assert_eq!(record.get("missing"), None);
}

#[test]
fn record_get_text() {
    let record = Record::new("r1")
        .with_field("name", FieldValue::Text("Alice".into()))
        .with_field("age", FieldValue::Integer(30));

    assert_eq!(record.get_text("name"), Some("Alice"));
    assert_eq!(record.get_text("age"), None);
    assert_eq!(record.get_text("missing"), None);
}

#[test]
fn record_batch_new_len_is_empty() {
    let empty = RecordBatch::new(vec!["name".into()], vec![]);
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());

    let batch = RecordBatch::new(
        vec!["name".into()],
        vec![
            Record::new("1").with_field("name", FieldValue::Text("Alice".into())),
            Record::new("2").with_field("name", FieldValue::Text("Bob".into())),
        ],
    );
    assert_eq!(batch.len(), 2);
    assert!(!batch.is_empty());
}

#[test]
fn record_batch_field_names() {
    let batch = RecordBatch::new(vec!["first".into(), "last".into()], vec![Record::new("1")]);
    assert_eq!(batch.field_names, vec!["first", "last"]);
}

#[test]
fn field_value_equality() {
    assert_eq!(FieldValue::Text("a".into()), FieldValue::Text("a".into()));
    assert_ne!(FieldValue::Text("a".into()), FieldValue::Text("b".into()));
    assert_ne!(FieldValue::Text("42".into()), FieldValue::Integer(42));
    assert_eq!(FieldValue::Null, FieldValue::Null);
}
