"""Tests for Tier 4 features: transliteration, alignment, benchmark,
pipeline serialization, language detection, profiling, MinHash index."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import reclink
from reclink import MinHashIndex
from reclink.pipeline import ReclinkPipeline

# ---------------------------------------------------------------------------
# T4-1: Transliteration
# ---------------------------------------------------------------------------


class TestTransliterateCyrillic:
    def test_basic(self) -> None:
        assert reclink.transliterate_cyrillic("Москва") == "Moskva"

    def test_name(self) -> None:
        assert reclink.transliterate_cyrillic("Иванов") == "Ivanov"

    def test_mixed_script(self) -> None:
        assert reclink.transliterate_cyrillic("Привет world") == "Privet world"

    def test_empty(self) -> None:
        assert reclink.transliterate_cyrillic("") == ""

    def test_latin_passthrough(self) -> None:
        assert reclink.transliterate_cyrillic("hello") == "hello"


class TestTransliterateGreek:
    def test_basic(self) -> None:
        assert reclink.transliterate_greek("Αθήνα") == "Athina"

    def test_name(self) -> None:
        result = reclink.transliterate_greek("Παπαδόπουλος")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mixed_script(self) -> None:
        assert reclink.transliterate_greek("Ελλάδα Greece") == "Ellada Greece"

    def test_empty(self) -> None:
        assert reclink.transliterate_greek("") == ""

    def test_latin_passthrough(self) -> None:
        assert reclink.transliterate_greek("hello") == "hello"


# ---------------------------------------------------------------------------
# T4-2: Levenshtein Alignment
# ---------------------------------------------------------------------------


class TestLevenshteinAlign:
    def test_basic_structure(self) -> None:
        result = reclink.levenshtein_align("cat", "car")
        assert "ops" in result
        assert "distance" in result
        assert "visual" in result

    def test_identical(self) -> None:
        result = reclink.levenshtein_align("abc", "abc")
        assert result["distance"] == 0
        ops: list[object] = list(result["ops"])  # type: ignore[call-overload]
        assert all(str(op).startswith("match:") for op in ops)

    def test_known_distance(self) -> None:
        result = reclink.levenshtein_align("kitten", "sitting")
        assert result["distance"] == 3

    def test_empty_strings(self) -> None:
        result = reclink.levenshtein_align("", "")
        assert result["distance"] == 0
        assert result["ops"] == []

    def test_one_empty(self) -> None:
        result = reclink.levenshtein_align("abc", "")
        assert result["distance"] == 3

    def test_visual_has_three_lines(self) -> None:
        result = reclink.levenshtein_align("Smith", "Smyth")
        visual = str(result["visual"])
        assert visual.count("\n") == 2  # 3 lines = 2 newlines

    def test_substitution_in_ops(self) -> None:
        result = reclink.levenshtein_align("cat", "car")
        assert result["distance"] == 1
        ops: list[object] = list(result["ops"])  # type: ignore[call-overload]
        assert any("sub:" in str(op) for op in ops)


# ---------------------------------------------------------------------------
# T4-3: Benchmark Utility
# ---------------------------------------------------------------------------


class TestBenchmarkMetrics:
    def test_basic(self) -> None:
        pairs = [("hello", "hallo"), ("john", "jon")]
        result = reclink.benchmark.benchmark_metrics(pairs, n=10)
        assert "results" in result
        assert "n_iterations" in result
        assert "n_pairs" in result
        assert result["n_iterations"] == 10
        assert result["n_pairs"] == 2
        assert len(result["results"]) > 0

    def test_specific_metrics(self) -> None:
        pairs = [("hello", "world")]
        result = reclink.benchmark.benchmark_metrics(pairs, metrics=["jaro", "jaro_winkler"], n=5)
        names = [r["metric"] for r in result["results"]]
        assert "jaro" in names
        assert "jaro_winkler" in names

    def test_empty_pairs(self) -> None:
        result = reclink.benchmark.benchmark_metrics([], n=10)
        assert result["results"] == []
        assert result["n_pairs"] == 0

    def test_result_structure(self) -> None:
        pairs = [("a", "b")]
        result = reclink.benchmark.benchmark_metrics(pairs, metrics=["jaro"], n=5)
        entry = result["results"][0]
        assert "metric" in entry
        assert "total_ns" in entry
        assert "per_pair_ns" in entry
        assert "pairs_per_sec" in entry


class TestBenchmarkPipeline:
    def test_basic(self) -> None:
        pipeline = (
            ReclinkPipeline.builder()
            .block_sorted_neighborhood("first_name", window=5)
            .compare_string("first_name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        records = [
            {"id": "1", "first_name": "John"},
            {"id": "2", "first_name": "Jon"},
        ]
        result = reclink.benchmark.benchmark_pipeline(pipeline, records, n=2)
        assert "total_ns" in result
        assert "per_run_ns" in result
        assert "runs_per_sec" in result
        assert "n_iterations" in result
        assert result["n_iterations"] == 2


# ---------------------------------------------------------------------------
# T4-4: Pipeline Serialization
# ---------------------------------------------------------------------------


class TestPipelineSerialization:
    def _build_pipeline(self) -> ReclinkPipeline:
        return (
            ReclinkPipeline.builder()
            .preprocess("first_name", ["fold_case"])
            .compare_string("first_name", metric="jaro_winkler")
            .classify_threshold(0.8)
            .build()
        )

    def test_to_json_returns_string(self) -> None:
        pipeline = self._build_pipeline()
        result = pipeline.to_json()
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_roundtrip_json(self) -> None:
        pipeline = self._build_pipeline()
        json_str = pipeline.to_json()
        restored = ReclinkPipeline.from_json(json_str)
        assert isinstance(restored, ReclinkPipeline)

    def test_roundtrip_produces_same_results(self) -> None:
        pipeline = (
            ReclinkPipeline.builder()
            .preprocess("first_name", ["fold_case"])
            .block_sorted_neighborhood("first_name", window=5)
            .compare_string("first_name", metric="jaro_winkler")
            .classify_threshold(0.8)
            .build()
        )
        records = [
            {"id": "1", "first_name": "John"},
            {"id": "2", "first_name": "Jon"},
        ]
        original_results = pipeline.dedup(records)

        json_str = pipeline.to_json()
        restored = ReclinkPipeline.from_json(json_str)
        restored_results = restored.dedup(records)

        assert len(original_results) == len(restored_results)

    def test_to_file_from_file(self) -> None:
        pipeline = self._build_pipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pipeline.json")
            pipeline.to_file(path)
            assert Path(path).exists()

            restored = ReclinkPipeline.from_file(path)
            assert isinstance(restored, ReclinkPipeline)


# ---------------------------------------------------------------------------
# T4-5: Language Detection
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_english(self) -> None:
        result = reclink.detect_language("Smith")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string(self) -> None:
        result = reclink.detect_language("Müller")
        assert isinstance(result, str)

    def test_empty(self) -> None:
        result = reclink.detect_language("")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# T4-6: Pipeline Profiling
# ---------------------------------------------------------------------------


class TestPipelineProfiling:
    def test_profiling_disabled_by_default(self) -> None:
        pipeline = (
            ReclinkPipeline.builder()
            .block_sorted_neighborhood("first_name", window=5)
            .compare_string("first_name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        records = [
            {"id": "1", "first_name": "John"},
            {"id": "2", "first_name": "Jon"},
        ]
        pipeline.dedup(records)
        stats = pipeline.profiling_stats
        assert isinstance(stats, dict)
        assert len(stats) == 0

    def test_profiling_enabled(self) -> None:
        pipeline = (
            ReclinkPipeline.builder()
            .block_sorted_neighborhood("first_name", window=5)
            .compare_string("first_name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
            .with_profiling()
        )
        records = [
            {"id": "1", "first_name": "John"},
            {"id": "2", "first_name": "Jon"},
        ]
        pipeline.dedup(records)
        stats = pipeline.profiling_stats
        assert isinstance(stats, dict)
        assert len(stats) > 0
        # All values should be non-negative nanoseconds
        for value in stats.values():
            assert isinstance(value, int)
            assert value >= 0

    def test_with_profiling_returns_self(self) -> None:
        pipeline = (
            ReclinkPipeline.builder()
            .block_sorted_neighborhood("first_name", window=5)
            .compare_string("first_name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        result = pipeline.with_profiling()
        assert result is pipeline


# ---------------------------------------------------------------------------
# T4-7: MinHash Index
# ---------------------------------------------------------------------------


class TestMinHashIndex:
    def test_build_and_query(self) -> None:
        strings = ["Jonathan Smith", "Jonathon Smith", "Xyz Abc"]
        index = MinHashIndex.build(strings, num_hashes=100, num_bands=20)
        assert len(index) == 3

        results = index.query("Jonathan Smith", threshold=0.3)
        assert len(results) > 0
        found = [r[1] for r in results]
        assert "Jonathan Smith" in found

    def test_insert(self) -> None:
        index = MinHashIndex.build(["hello world"], num_hashes=50, num_bands=10)
        assert len(index) == 1
        idx = index.insert("hello worl")
        assert idx == 1
        assert len(index) == 2

    def test_remove(self) -> None:
        strings = ["abc def", "abc deg", "xyz"]
        index = MinHashIndex.build(strings, num_hashes=50, num_bands=10)
        assert len(index) == 3
        assert index.remove(0)
        assert len(index) == 2

    def test_empty_index(self) -> None:
        index = MinHashIndex.build([], num_hashes=50, num_bands=10)
        assert len(index) == 0
        results = index.query("anything", threshold=0.0)
        assert results == []

    def test_save_load(self) -> None:
        strings = ["hello", "world", "test"]
        index = MinHashIndex.build(strings, num_hashes=50, num_bands=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "minhash.bin")
            index.save(path)
            loaded = MinHashIndex.load(path)
            assert len(loaded) == 3

    def test_similar_strings_found(self) -> None:
        strings = ["Jonathan Smith", "Jonathon Smith", "Xyz Abc"]
        index = MinHashIndex.build(strings, num_hashes=100, num_bands=20)
        results = index.query("Jonathan Smith", threshold=0.3)
        found_names = [r[1] for r in results]
        assert "Jonathan Smith" in found_names
        assert "Jonathon Smith" in found_names


# ---------------------------------------------------------------------------
# T5-2: CJK Tokenization
# ---------------------------------------------------------------------------


class TestCjkTokenization:
    def test_character_tokenize_cjk(self) -> None:
        tokens = reclink.character_tokenize("東京タワー")
        assert tokens == ["東", "京", "タ", "ワ", "ー"]

    def test_character_tokenize_latin(self) -> None:
        tokens = reclink.character_tokenize("hello")
        assert tokens == ["h", "e", "l", "l", "o"]

    def test_character_tokenize_strips_whitespace(self) -> None:
        tokens = reclink.character_tokenize("a b")
        assert tokens == ["a", "b"]

    def test_character_tokenize_empty(self) -> None:
        assert reclink.character_tokenize("") == []

    def test_smart_tokenize_chinese(self) -> None:
        tokens = reclink.smart_tokenize("你好世界")
        assert tokens == ["你", "好", "世", "界"]

    def test_smart_tokenize_mixed(self) -> None:
        tokens = reclink.smart_tokenize("Hello 世界")
        assert tokens == ["Hello", "世", "界"]

    def test_smart_tokenize_latin(self) -> None:
        tokens = reclink.smart_tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_smart_tokenize_japanese(self) -> None:
        tokens = reclink.smart_tokenize("東京タワー")
        assert len(tokens) == 5

    def test_smart_tokenize_empty(self) -> None:
        assert reclink.smart_tokenize("") == []

    def test_character_tokenize_batch(self) -> None:
        results = reclink.character_tokenize_batch(["ab", "你好"])
        assert results == [["a", "b"], ["你", "好"]]

    def test_smart_tokenize_batch(self) -> None:
        results = reclink.smart_tokenize_batch(["Hello 世界", "test"])
        assert results == [["Hello", "世", "界"], ["test"]]


# ---------------------------------------------------------------------------
# T5-4: Index Memory Usage Reporting
# ---------------------------------------------------------------------------


class TestIndexMemoryUsage:
    def test_bk_tree_memory_usage(self) -> None:
        tree = reclink.BkTree.build(["hello", "world", "test"], "levenshtein")
        usage = tree.memory_usage()
        assert isinstance(usage, int)
        assert usage > 0

    def test_bk_tree_memory_usage_human(self) -> None:
        tree = reclink.BkTree.build(["hello", "world"], "levenshtein")
        human = tree.memory_usage_human()
        assert isinstance(human, str)
        assert any(unit in human for unit in ["B", "KB", "MB", "GB"])

    def test_vp_tree_memory_usage(self) -> None:
        tree = reclink.VpTree.build(["hello", "world", "test"], "jaro_winkler")
        usage = tree.memory_usage()
        assert isinstance(usage, int)
        assert usage > 0

    def test_vp_tree_memory_usage_human(self) -> None:
        tree = reclink.VpTree.build(["hello", "world"], "jaro_winkler")
        human = tree.memory_usage_human()
        assert isinstance(human, str)

    def test_ngram_index_memory_usage(self) -> None:
        index = reclink.NgramIndex.build(["hello", "world", "test"], 2)
        usage = index.memory_usage()
        assert isinstance(usage, int)
        assert usage > 0

    def test_ngram_index_memory_usage_human(self) -> None:
        index = reclink.NgramIndex.build(["hello", "world"], 2)
        human = index.memory_usage_human()
        assert isinstance(human, str)

    def test_minhash_index_memory_usage(self) -> None:
        index = MinHashIndex.build(["hello", "world", "test"], num_hashes=50, num_bands=10)
        usage = index.memory_usage()
        assert isinstance(usage, int)
        assert usage > 0

    def test_minhash_index_memory_usage_human(self) -> None:
        index = MinHashIndex.build(["hello", "world"], num_hashes=50, num_bands=10)
        human = index.memory_usage_human()
        assert isinstance(human, str)

    def test_memory_grows_with_data(self) -> None:
        small = reclink.NgramIndex.build(["a", "b"], 2)
        large = reclink.NgramIndex.build([f"string_{i}" for i in range(100)], 2)
        assert large.memory_usage() > small.memory_usage()
