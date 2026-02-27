use criterion::{black_box, criterion_group, criterion_main, Criterion};
use reclink_core::metrics::damerau_levenshtein::{
    damerau_levenshtein_distance, damerau_levenshtein_distance_threshold,
};
use reclink_core::metrics::hamming::hamming_distance;
use reclink_core::metrics::lcs::lcs_length;
use reclink_core::metrics::levenshtein::{levenshtein_distance, levenshtein_distance_threshold};
use reclink_core::metrics::longest_common_substring::longest_common_substring_length;
use reclink_core::metrics::smith_waterman::smith_waterman_score;
use reclink_core::metrics::{
    Cosine, DamerauLevenshtein, DistanceMetric, Jaccard, Jaro, JaroWinkler, Lcs, Levenshtein,
    LongestCommonSubstring, NgramSimilarity, PartialRatio, PhoneticHybrid, SimilarityMetric,
    SmithWaterman, SorensenDice, TokenSet, TokenSort, WeightedLevenshtein,
};

/// Generate a pair of strings of given length with `num_changes` substitutions.
fn make_pair(len: usize, num_changes: usize) -> (String, String) {
    let a: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
    let b: String = a
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if num_changes > 0 && i % (len / num_changes) == 0 {
                'Z'
            } else {
                c
            }
        })
        .collect();
    (a, b)
}

fn bench_levenshtein(c: &mut Criterion) {
    let m = Levenshtein;
    c.bench_function("levenshtein_short_7x7", |b| {
        b.iter(|| m.distance(black_box("kitten"), black_box("sitting")))
    });
}

fn bench_levenshtein_medium(c: &mut Criterion) {
    let a = "the quick brown fox jumps over the lazy dog";
    let b = "the quack brawn fix jumped over a lazy cat";
    c.bench_function("levenshtein_medium_43x43", |b_| {
        b_.iter(|| levenshtein_distance(black_box(a), black_box(b)))
    });
}

fn bench_levenshtein_boundary(c: &mut Criterion) {
    let a: String = (0..64).map(|i| (b'a' + (i % 26)) as char).collect();
    let b: String = a
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if i == 10 || i == 30 || i == 50 {
                'Z'
            } else {
                c
            }
        })
        .collect();
    c.bench_function("levenshtein_boundary_64x64", |b_| {
        b_.iter(|| levenshtein_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_levenshtein_long(c: &mut Criterion) {
    let (a, b) = make_pair(100, 3);
    c.bench_function("levenshtein_long_100x100", |b_| {
        b_.iter(|| levenshtein_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_levenshtein_200(c: &mut Criterion) {
    let (a, b) = make_pair(200, 5);
    c.bench_function("levenshtein_200x200", |b_| {
        b_.iter(|| levenshtein_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_levenshtein_500(c: &mut Criterion) {
    let (a, b) = make_pair(500, 10);
    c.bench_function("levenshtein_500x500", |b_| {
        b_.iter(|| levenshtein_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_damerau_levenshtein(c: &mut Criterion) {
    let m = DamerauLevenshtein;
    c.bench_function("damerau_levenshtein_short_7x7", |b| {
        b.iter(|| m.distance(black_box("kitten"), black_box("sitting")))
    });
}

fn bench_damerau_levenshtein_43(c: &mut Criterion) {
    let a = "the quick brown fox jumps over the lazy dog";
    let b = "the quack brawn fix jumped over a lazy cat";
    c.bench_function("damerau_levenshtein_43x43", |b_| {
        b_.iter(|| damerau_levenshtein_distance(black_box(a), black_box(b)))
    });
}

fn bench_damerau_levenshtein_64(c: &mut Criterion) {
    let (a, b) = make_pair(64, 3);
    c.bench_function("damerau_levenshtein_64x64", |b_| {
        b_.iter(|| damerau_levenshtein_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_damerau_levenshtein_100(c: &mut Criterion) {
    let (a, b) = make_pair(100, 3);
    c.bench_function("damerau_levenshtein_100x100", |b_| {
        b_.iter(|| damerau_levenshtein_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_jaro(c: &mut Criterion) {
    let m = Jaro;
    c.bench_function("jaro_short_6x6", |b| {
        b.iter(|| m.similarity(black_box("martha"), black_box("marhta")))
    });
}

fn bench_jaro_medium(c: &mut Criterion) {
    let a = "the quick brown fox jumps over the lazy dog";
    let b = "the quack brawn fix jumped over a lazy cat";
    let m = Jaro;
    c.bench_function("jaro_medium_43x43", |b_| {
        b_.iter(|| m.similarity(black_box(a), black_box(b)))
    });
}

fn bench_jaro_100(c: &mut Criterion) {
    let (a, b) = make_pair(100, 3);
    let m = Jaro;
    c.bench_function("jaro_100x100", |b_| {
        b_.iter(|| m.similarity(black_box(&a), black_box(&b)))
    });
}

fn bench_jaro_winkler(c: &mut Criterion) {
    let m = JaroWinkler::default();
    c.bench_function("jaro_winkler", |b| {
        b.iter(|| m.similarity(black_box("martha"), black_box("marhta")))
    });
}

fn bench_jaro_winkler_medium(c: &mut Criterion) {
    let a = "the quick brown fox jumps over the lazy dog";
    let b = "the quack brawn fix jumped over a lazy cat";
    let m = JaroWinkler::default();
    c.bench_function("jaro_winkler_medium_43x43", |b_| {
        b_.iter(|| m.similarity(black_box(a), black_box(b)))
    });
}

fn bench_cosine(c: &mut Criterion) {
    let m = Cosine::default();
    c.bench_function("cosine_bigram", |b| {
        b.iter(|| m.similarity(black_box("night"), black_box("nacht")))
    });
}

fn bench_jaccard(c: &mut Criterion) {
    let m = Jaccard;
    c.bench_function("jaccard", |b| {
        b.iter(|| m.similarity(black_box("hello world foo"), black_box("hello bar baz")))
    });
}

fn bench_sorensen_dice(c: &mut Criterion) {
    let m = SorensenDice;
    c.bench_function("sorensen_dice", |b| {
        b.iter(|| m.similarity(black_box("night"), black_box("nacht")))
    });
}

fn bench_weighted_levenshtein(c: &mut Criterion) {
    let m = WeightedLevenshtein::default();
    c.bench_function("weighted_levenshtein", |b| {
        b.iter(|| m.similarity(black_box("kitten"), black_box("sitting")))
    });
}

fn bench_token_sort(c: &mut Criterion) {
    let m = TokenSort;
    c.bench_function("token_sort", |b| {
        b.iter(|| m.similarity(black_box("John Smith"), black_box("Smith John")))
    });
}

fn bench_token_set(c: &mut Criterion) {
    let m = TokenSet;
    c.bench_function("token_set", |b| {
        b.iter(|| m.similarity(black_box("New York City"), black_box("New York")))
    });
}

fn bench_partial_ratio(c: &mut Criterion) {
    let m = PartialRatio;
    c.bench_function("partial_ratio", |b| {
        b.iter(|| m.similarity(black_box("test"), black_box("this is a test")))
    });
}

fn bench_lcs(c: &mut Criterion) {
    let m = Lcs;
    c.bench_function("lcs_short_7x7", |b| {
        b.iter(|| m.similarity(black_box("kitten"), black_box("sitting")))
    });
}

fn bench_lcs_43(c: &mut Criterion) {
    let a = "the quick brown fox jumps over the lazy dog";
    let b = "the quack brawn fix jumped over a lazy cat";
    c.bench_function("lcs_43x43", |b_| {
        b_.iter(|| lcs_length(black_box(a), black_box(b)))
    });
}

fn bench_lcs_100(c: &mut Criterion) {
    let (a, b) = make_pair(100, 3);
    c.bench_function("lcs_100x100", |b_| {
        b_.iter(|| lcs_length(black_box(&a), black_box(&b)))
    });
}

fn bench_longest_common_substring(c: &mut Criterion) {
    let m = LongestCommonSubstring;
    c.bench_function("longest_common_substring_short", |b| {
        b.iter(|| m.similarity(black_box("abcxyz"), black_box("xyzabc")))
    });
}

fn bench_longest_common_substring_43(c: &mut Criterion) {
    let a = "the quick brown fox jumps over the lazy dog";
    let b = "the quack brawn fix jumped over a lazy cat";
    c.bench_function("longest_common_substring_43x43", |b_| {
        b_.iter(|| longest_common_substring_length(black_box(a), black_box(b)))
    });
}

fn bench_longest_common_substring_100(c: &mut Criterion) {
    let (a, b) = make_pair(100, 3);
    c.bench_function("longest_common_substring_100x100", |b_| {
        b_.iter(|| longest_common_substring_length(black_box(&a), black_box(&b)))
    });
}

fn bench_ngram_similarity(c: &mut Criterion) {
    let m = NgramSimilarity::default();
    c.bench_function("ngram_similarity", |b| {
        b.iter(|| m.similarity(black_box("night"), black_box("nacht")))
    });
}

fn bench_smith_waterman(c: &mut Criterion) {
    let m = SmithWaterman::default();
    c.bench_function("smith_waterman_short", |b| {
        b.iter(|| m.similarity(black_box("ACBDE"), black_box("XACBDEY")))
    });
}

fn bench_smith_waterman_43(c: &mut Criterion) {
    let a = "the quick brown fox jumps over the lazy dog";
    let b = "the quack brawn fix jumped over a lazy cat";
    c.bench_function("smith_waterman_43x43", |b_| {
        b_.iter(|| smith_waterman_score(black_box(a), black_box(b), 2.0, -1.0, -1.0))
    });
}

fn bench_smith_waterman_100(c: &mut Criterion) {
    let (a, b) = make_pair(100, 3);
    c.bench_function("smith_waterman_100x100", |b_| {
        b_.iter(|| smith_waterman_score(black_box(&a), black_box(&b), 2.0, -1.0, -1.0))
    });
}

fn bench_hamming_100(c: &mut Criterion) {
    let (a, b) = make_pair(100, 3);
    c.bench_function("hamming_100x100", |b_| {
        b_.iter(|| hamming_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_hamming_500(c: &mut Criterion) {
    let (a, b) = make_pair(500, 10);
    c.bench_function("hamming_500x500", |b_| {
        b_.iter(|| hamming_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_phonetic_hybrid(c: &mut Criterion) {
    let m = PhoneticHybrid::default();
    c.bench_function("phonetic_hybrid", |b| {
        b.iter(|| m.similarity(black_box("Smith"), black_box("Smyth")))
    });
}

fn bench_levenshtein_threshold(c: &mut Criterion) {
    c.bench_function("levenshtein_threshold", |b| {
        b.iter(|| {
            levenshtein_distance_threshold(
                black_box("kitten sitting on a mat"),
                black_box("sitting kitten on the mat"),
                black_box(5),
            )
        })
    });
}

fn bench_damerau_levenshtein_threshold(c: &mut Criterion) {
    c.bench_function("damerau_levenshtein_threshold", |b| {
        b.iter(|| {
            damerau_levenshtein_distance_threshold(
                black_box("kitten sitting on a mat"),
                black_box("sitting kitten on the mat"),
                black_box(5),
            )
        })
    });
}

criterion_group!(
    benches,
    // Levenshtein
    bench_levenshtein,
    bench_levenshtein_medium,
    bench_levenshtein_boundary,
    bench_levenshtein_long,
    bench_levenshtein_200,
    bench_levenshtein_500,
    // Damerau-Levenshtein
    bench_damerau_levenshtein,
    bench_damerau_levenshtein_43,
    bench_damerau_levenshtein_64,
    bench_damerau_levenshtein_100,
    // Jaro / Jaro-Winkler
    bench_jaro,
    bench_jaro_medium,
    bench_jaro_100,
    bench_jaro_winkler,
    bench_jaro_winkler_medium,
    // Cosine
    bench_cosine,
    // Set-based
    bench_jaccard,
    bench_sorensen_dice,
    // Weighted
    bench_weighted_levenshtein,
    // Token-based
    bench_token_sort,
    bench_token_set,
    bench_partial_ratio,
    // LCS
    bench_lcs,
    bench_lcs_43,
    bench_lcs_100,
    // Longest Common Substring
    bench_longest_common_substring,
    bench_longest_common_substring_43,
    bench_longest_common_substring_100,
    // N-gram
    bench_ngram_similarity,
    // Smith-Waterman
    bench_smith_waterman,
    bench_smith_waterman_43,
    bench_smith_waterman_100,
    // Hamming
    bench_hamming_100,
    bench_hamming_500,
    // Phonetic
    bench_phonetic_hybrid,
    // Threshold variants
    bench_levenshtein_threshold,
    bench_damerau_levenshtein_threshold,
);
criterion_main!(benches);
