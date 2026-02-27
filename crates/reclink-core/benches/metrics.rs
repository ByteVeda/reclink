use criterion::{black_box, criterion_group, criterion_main, Criterion};
use reclink_core::metrics::damerau_levenshtein::damerau_levenshtein_distance_threshold;
use reclink_core::metrics::levenshtein::{levenshtein_distance, levenshtein_distance_threshold};
use reclink_core::metrics::{
    Cosine, DamerauLevenshtein, DistanceMetric, Jaccard, Jaro, JaroWinkler, Lcs,
    Levenshtein, LongestCommonSubstring, NgramSimilarity, PartialRatio, PhoneticHybrid,
    SimilarityMetric, SmithWaterman, SorensenDice, TokenSet, TokenSort, WeightedLevenshtein,
};

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
        .map(|(i, c)| if i == 10 || i == 30 || i == 50 { 'Z' } else { c })
        .collect();
    c.bench_function("levenshtein_boundary_64x64", |b_| {
        b_.iter(|| levenshtein_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_levenshtein_long(c: &mut Criterion) {
    let a: String = (0..100).map(|i| (b'a' + (i % 26)) as char).collect();
    let b: String = a
        .chars()
        .enumerate()
        .map(|(i, c)| if i == 20 || i == 50 || i == 80 { 'Z' } else { c })
        .collect();
    c.bench_function("levenshtein_long_100x100", |b_| {
        b_.iter(|| levenshtein_distance(black_box(&a), black_box(&b)))
    });
}

fn bench_damerau_levenshtein(c: &mut Criterion) {
    let m = DamerauLevenshtein;
    c.bench_function("damerau_levenshtein", |b| {
        b.iter(|| m.distance(black_box("kitten"), black_box("sitting")))
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
    c.bench_function("lcs", |b| {
        b.iter(|| m.similarity(black_box("kitten"), black_box("sitting")))
    });
}

fn bench_longest_common_substring(c: &mut Criterion) {
    let m = LongestCommonSubstring;
    c.bench_function("longest_common_substring", |b| {
        b.iter(|| m.similarity(black_box("abcxyz"), black_box("xyzabc")))
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
    c.bench_function("smith_waterman", |b| {
        b.iter(|| m.similarity(black_box("ACBDE"), black_box("XACBDEY")))
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
    bench_levenshtein,
    bench_levenshtein_medium,
    bench_levenshtein_boundary,
    bench_levenshtein_long,
    bench_damerau_levenshtein,
    bench_jaro,
    bench_jaro_medium,
    bench_jaro_winkler,
    bench_jaro_winkler_medium,
    bench_cosine,
    bench_jaccard,
    bench_sorensen_dice,
    bench_weighted_levenshtein,
    bench_token_sort,
    bench_token_set,
    bench_partial_ratio,
    bench_lcs,
    bench_longest_common_substring,
    bench_ngram_similarity,
    bench_smith_waterman,
    bench_phonetic_hybrid,
    bench_levenshtein_threshold,
    bench_damerau_levenshtein_threshold,
);
criterion_main!(benches);
