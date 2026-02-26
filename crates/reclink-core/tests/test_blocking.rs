use reclink_core::blocking::BlockingStrategy;
use reclink_core::blocking::{
    CanopyClustering, LshBlocking, PhoneticBlocking, QgramBlocking, SortedNeighborhood,
};
use reclink_core::metrics::Metric;
use reclink_core::record::{FieldValue, Record, RecordBatch};

fn make_batch(names: &[&str]) -> RecordBatch {
    let records: Vec<Record> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            Record::new(format!("r{i}")).with_field("name", FieldValue::Text((*name).into()))
        })
        .collect();
    RecordBatch::new(vec!["name".into()], records)
}

#[test]
fn qgram_link() {
    let left = make_batch(&["Smith", "Jones"]);
    let right = make_batch(&["Smyth", "Brown"]);
    let blocker = QgramBlocking::new("name", 2, 1);
    let pairs = blocker.block_link(&left, &right);
    // Smith and Smyth share "th" bigram at minimum
    assert!(
        pairs.iter().any(|p| p.left == 0 && p.right == 0),
        "Smith(left=0) and Smyth(right=0) should share q-grams"
    );
}

#[test]
fn sorted_neighborhood_link() {
    let left = make_batch(&["Alice", "Bob"]);
    let right = make_batch(&["Alicia", "Zane"]);
    let blocker = SortedNeighborhood::new("name", 3);
    let pairs = blocker.block_link(&left, &right);
    // After sorting: Alice, Alicia, Bob, Zane — Alice and Alicia are neighbors
    assert!(
        pairs.iter().any(|p| p.left == 0 && p.right == 0),
        "Alice(left=0) and Alicia(right=0) should be neighbors"
    );
}

#[test]
fn lsh_link() {
    let left = make_batch(&["Jonathan Smith", "Xyz Abc"]);
    let right = make_batch(&["Jonathon Smith", "Qwerty"]);
    let blocker = LshBlocking::new("name", 100, 20);
    let pairs = blocker.block_link(&left, &right);
    // "Jonathan Smith" and "Jonathon Smith" are very similar
    assert!(
        pairs.iter().any(|p| p.left == 0 && p.right == 0),
        "Jonathan Smith and Jonathon Smith should share LSH bands"
    );
}

#[test]
fn canopy_link() {
    let left = make_batch(&["Smith", "Jones"]);
    let right = make_batch(&["Smyth", "Brown"]);
    let blocker = CanopyClustering::new("name", 0.9, 0.5, Metric::default());
    let pairs = blocker.block_link(&left, &right);
    // Smith and Smyth should be within the loose threshold
    assert!(
        pairs.iter().any(|p| p.left == 0 && p.right == 0),
        "Smith(left=0) and Smyth(right=0) should be within canopy threshold"
    );
}

#[test]
fn phonetic_blocking_link() {
    let left = make_batch(&["Smith", "Jones"]);
    let right = make_batch(&["Smyth", "Brown"]);
    let blocker = PhoneticBlocking::soundex("name");
    let pairs = blocker.block_link(&left, &right);
    // Smith and Smyth have the same Soundex code (S530)
    assert!(
        pairs.iter().any(|p| p.left == 0 && p.right == 0),
        "Smith and Smyth should have the same Soundex code"
    );
}
