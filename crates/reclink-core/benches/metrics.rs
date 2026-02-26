use criterion::{black_box, criterion_group, criterion_main, Criterion};
use reclink_core::metrics::{
    Cosine, DamerauLevenshtein, DistanceMetric, Hamming, Jaccard, Jaro, JaroWinkler, Levenshtein,
    SimilarityMetric, SorensenDice,
};

fn bench_levenshtein(c: &mut Criterion) {
    let m = Levenshtein;
    c.bench_function("levenshtein", |b| {
        b.iter(|| m.distance(black_box("kitten"), black_box("sitting")))
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
    c.bench_function("jaro", |b| {
        b.iter(|| m.similarity(black_box("martha"), black_box("marhta")))
    });
}

fn bench_jaro_winkler(c: &mut Criterion) {
    let m = JaroWinkler::default();
    c.bench_function("jaro_winkler", |b| {
        b.iter(|| m.similarity(black_box("martha"), black_box("marhta")))
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

criterion_group!(
    benches,
    bench_levenshtein,
    bench_damerau_levenshtein,
    bench_jaro,
    bench_jaro_winkler,
    bench_cosine,
    bench_jaccard,
    bench_sorensen_dice,
);
criterion_main!(benches);
