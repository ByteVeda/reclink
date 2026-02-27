"""Tests for string similarity metrics."""

import numpy as np
import pytest

import reclink


class TestLevenshtein:
    def test_identical(self) -> None:
        assert reclink.levenshtein("hello", "hello") == 0

    def test_known_values(self) -> None:
        assert reclink.levenshtein("kitten", "sitting") == 3
        assert reclink.levenshtein("saturday", "sunday") == 3

    def test_empty(self) -> None:
        assert reclink.levenshtein("", "") == 0
        assert reclink.levenshtein("abc", "") == 3

    def test_similarity(self) -> None:
        sim = reclink.levenshtein_similarity("kitten", "sitting")
        assert 0.5 < sim < 0.7


class TestLevenshteinThreshold:
    def test_within_threshold(self) -> None:
        assert reclink.levenshtein_threshold("kitten", "sitting", 3) == 3
        assert reclink.levenshtein_threshold("kitten", "sitting", 5) == 3

    def test_exceeds_threshold(self) -> None:
        assert reclink.levenshtein_threshold("kitten", "sitting", 2) is None

    def test_identical(self) -> None:
        assert reclink.levenshtein_threshold("hello", "hello", 0) == 0

    def test_empty(self) -> None:
        assert reclink.levenshtein_threshold("", "", 0) == 0


class TestDamerauLevenshteinThreshold:
    def test_within_threshold(self) -> None:
        assert reclink.damerau_levenshtein_threshold("ab", "ba", 1) == 1

    def test_exceeds_threshold(self) -> None:
        assert reclink.damerau_levenshtein_threshold("ca", "abc", 2) is None

    def test_identical(self) -> None:
        assert reclink.damerau_levenshtein_threshold("hello", "hello", 0) == 0


class TestDamerauLevenshtein:
    def test_transposition(self) -> None:
        assert reclink.damerau_levenshtein("ab", "ba") == 1

    def test_known_values(self) -> None:
        assert reclink.damerau_levenshtein("ca", "abc") == 3


class TestHamming:
    def test_known_values(self) -> None:
        assert reclink.hamming("karolin", "kathrin") == 3

    def test_unequal_length(self) -> None:
        with pytest.raises(ValueError):
            reclink.hamming("abc", "ab")


class TestJaro:
    def test_known_values(self) -> None:
        assert abs(reclink.jaro("martha", "marhta") - 0.9444) < 0.001

    def test_identical(self) -> None:
        assert reclink.jaro("hello", "hello") == 1.0


class TestJaroWinkler:
    def test_known_values(self) -> None:
        assert abs(reclink.jaro_winkler("martha", "marhta") - 0.9611) < 0.001

    def test_prefix_boost(self) -> None:
        jw = reclink.jaro_winkler("abc", "abx")
        j = reclink.jaro("abc", "abx")
        assert jw >= j


class TestCosine:
    def test_identical(self) -> None:
        assert abs(reclink.cosine("hello", "hello") - 1.0) < 0.001

    def test_different(self) -> None:
        assert abs(reclink.cosine("ab", "cd")) < 0.001


class TestJaccard:
    def test_identical(self) -> None:
        assert abs(reclink.jaccard("hello world", "hello world") - 1.0) < 0.001

    def test_partial(self) -> None:
        sim = reclink.jaccard("cat dog", "cat bird")
        assert abs(sim - 1 / 3) < 0.001


class TestSorensenDice:
    def test_known_values(self) -> None:
        assert abs(reclink.sorensen_dice("night", "nacht") - 0.25) < 0.001


class TestWeightedLevenshtein:
    def test_uniform_costs(self) -> None:
        # Uniform costs of 1.0 should match regular Levenshtein distance
        assert abs(reclink.weighted_levenshtein("kitten", "sitting") - 3.0) < 0.001

    def test_asymmetric_costs(self) -> None:
        # Insert cost = 2.0, so "abc" from "" costs 6.0
        assert abs(reclink.weighted_levenshtein("", "abc", insert_cost=2.0) - 6.0) < 0.001

    def test_transpose(self) -> None:
        # "ab" -> "ba" with transpose cost 0.5
        dist = reclink.weighted_levenshtein("ab", "ba", transpose_cost=0.5)
        assert abs(dist - 0.5) < 0.001

    def test_similarity(self) -> None:
        sim = reclink.weighted_levenshtein_similarity("hello", "hello")
        assert abs(sim - 1.0) < 0.001


class TestTokenSortRatio:
    def test_reordered(self) -> None:
        assert abs(reclink.token_sort_ratio("John Smith", "Smith John") - 1.0) < 0.001

    def test_different(self) -> None:
        sim = reclink.token_sort_ratio("John Smith", "Jane Doe")
        assert sim < 0.5


class TestTokenSetRatio:
    def test_subset(self) -> None:
        sim = reclink.token_set_ratio("New York", "New York City")
        assert sim > 0.8

    def test_identical(self) -> None:
        assert abs(reclink.token_set_ratio("hello world", "hello world") - 1.0) < 0.001


class TestPartialRatio:
    def test_substring(self) -> None:
        assert abs(reclink.partial_ratio("test", "this is a test") - 1.0) < 0.001

    def test_identical(self) -> None:
        assert abs(reclink.partial_ratio("hello", "hello") - 1.0) < 0.001


class TestLcs:
    def test_known_values(self) -> None:
        assert reclink.lcs_length("abcde", "ace") == 3

    def test_similarity(self) -> None:
        sim = reclink.lcs_similarity("abcde", "ace")
        assert abs(sim - 0.75) < 0.001

    def test_empty(self) -> None:
        assert reclink.lcs_length("", "") == 0


class TestLongestCommonSubstring:
    def test_known_values(self) -> None:
        assert reclink.longest_common_substring_length("abcxyz", "xyzabc") == 3

    def test_similarity(self) -> None:
        sim = reclink.longest_common_substring_similarity("abcxyz", "xyzabc")
        assert abs(sim - 0.5) < 0.001

    def test_no_common(self) -> None:
        assert reclink.longest_common_substring_length("abc", "xyz") == 0


class TestNgramSimilarity:
    def test_identical(self) -> None:
        assert abs(reclink.ngram_similarity("hello", "hello") - 1.0) < 0.001

    def test_configurable_n(self) -> None:
        sim = reclink.ngram_similarity("abc", "bcd", n=1)
        assert abs(sim - 0.5) < 0.001

    def test_known_values(self) -> None:
        sim = reclink.ngram_similarity("night", "nacht")
        assert abs(sim - 1.0 / 7.0) < 0.001


class TestSmithWaterman:
    def test_identical(self) -> None:
        assert abs(reclink.smith_waterman_similarity("hello", "hello") - 1.0) < 0.001

    def test_local_alignment(self) -> None:
        sim = reclink.smith_waterman_similarity("ACBDE", "XACBDEY")
        assert abs(sim - 1.0) < 0.001

    def test_empty(self) -> None:
        assert abs(reclink.smith_waterman_similarity("", "") - 1.0) < 0.001

    def test_raw_score(self) -> None:
        score = reclink.smith_waterman("ACGT", "ACGT")
        assert abs(score - 8.0) < 0.001


class TestCaverphone:
    def test_known_values(self) -> None:
        assert reclink.caverphone("Lee") == "L111111111"

    def test_length(self) -> None:
        assert len(reclink.caverphone("test")) == 10


class TestColognePhonetic:
    def test_known_values(self) -> None:
        assert reclink.cologne_phonetic("Muller") == "657"

    def test_matching(self) -> None:
        assert reclink.cologne_phonetic("Muller") == reclink.cologne_phonetic("Mueller")


class TestPhoneticHybrid:
    def test_combined_score(self) -> None:
        # Smith and Smyth have the same Soundex code
        sim = reclink.phonetic_hybrid("Smith", "Smyth")
        edit_only = reclink.jaro_winkler("Smith", "Smyth")
        assert sim > edit_only * 0.7  # phonetic match should boost score

    def test_identical(self) -> None:
        sim = reclink.phonetic_hybrid("hello", "hello")
        assert abs(sim - 1.0) < 0.001


class TestMatchBest:
    def test_identical_match(self) -> None:
        result = reclink.match_best("hello", ["hello", "world", "help"])
        assert result is not None
        assert result[0] == "hello"
        assert abs(result[1] - 1.0) < 0.001
        assert result[2] == 0

    def test_threshold_filters(self) -> None:
        result = reclink.match_best("hello", ["xyz", "abc"], threshold=0.9)
        assert result is None

    def test_empty_candidates(self) -> None:
        result = reclink.match_best("hello", [])
        assert result is None

    def test_with_scorer(self) -> None:
        result = reclink.match_best("hello", ["hallo", "world"], scorer="levenshtein")
        assert result is not None
        assert result[0] == "hallo"


class TestMatchBatch:
    def test_sorted_output(self) -> None:
        results = reclink.match_batch("hello", ["world", "hello", "help"])
        assert len(results) == 3
        # Results should be sorted by descending score
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]
        # Best match should be "hello"
        assert results[0][0] == "hello"

    def test_limit(self) -> None:
        results = reclink.match_batch("hello", ["world", "hello", "help"], limit=2)
        assert len(results) == 2

    def test_threshold(self) -> None:
        results = reclink.match_batch("hello", ["hello", "xyz", "abc"], threshold=0.5)
        assert all(r[1] >= 0.5 for r in results)

    def test_multiple_scorers(self) -> None:
        for scorer in ["jaro", "cosine", "jaccard", "levenshtein"]:
            results = reclink.match_batch("hello", ["hallo", "world"], scorer=scorer)
            assert len(results) == 2


class TestExplain:
    def test_all_algorithms(self) -> None:
        result = reclink.explain("kitten", "sitting")
        assert "levenshtein" in result
        assert "jaro_winkler" in result
        assert 0 <= result["levenshtein"] <= 1

    def test_specific_algorithms(self) -> None:
        result = reclink.explain("hello", "world", algorithms=["jaro", "cosine"])
        assert len(result) == 2
        assert "jaro" in result
        assert "cosine" in result

    def test_identical(self) -> None:
        result = reclink.explain("hello", "hello")
        for name, score in result.items():
            if name != "hamming":
                assert score > 0.9, f"{name} returned {score} for identical strings"


class TestBkTree:
    def test_build_and_query(self) -> None:
        tree = reclink.BkTree.build(["hello", "hallo", "world", "help"], metric="levenshtein")
        results = tree.find_within("hello", max_distance=1)
        values = {r[0] for r in results}
        assert "hello" in values
        assert "hallo" in values
        assert "world" not in values

    def test_find_nearest(self) -> None:
        tree = reclink.BkTree.build(["apple", "apply", "ape", "banana"], metric="levenshtein")
        results = tree.find_nearest("appel", k=2)
        assert len(results) == 2
        assert results[0][0] == "apple"

    def test_len(self) -> None:
        tree = reclink.BkTree.build(["a", "b", "c"], metric="levenshtein")
        assert len(tree) == 3

    def test_invalid_metric(self) -> None:
        with pytest.raises(ValueError):
            reclink.BkTree.build(["hello"], metric="jaro")

    def test_empty(self) -> None:
        tree = reclink.BkTree.build([], metric="levenshtein")
        assert len(tree) == 0
        assert tree.find_within("hello", max_distance=1) == []


class TestTfIdfMatcher:
    def test_fit_and_similarity(self) -> None:
        corpus = ["apple inc", "banana corp", "cherry inc"]
        matcher = reclink.TfIdfMatcher.fit(corpus)
        sim = matcher.similarity("apple inc", "apple corp")
        assert sim > 0.0

    def test_common_tokens_downweighted(self) -> None:
        corpus = ["the cat", "the dog", "the bird", "unique phrase"]
        matcher = reclink.TfIdfMatcher.fit(corpus)
        sim = matcher.similarity("the cat", "the dog")
        assert sim < 0.5

    def test_identical(self) -> None:
        corpus = ["hello world", "foo bar"]
        matcher = reclink.TfIdfMatcher.fit(corpus)
        sim = matcher.similarity("hello world", "hello world")
        assert abs(sim - 1.0) < 0.001

    def test_match_batch(self) -> None:
        corpus = ["apple inc", "apple corp", "banana inc"]
        matcher = reclink.TfIdfMatcher.fit(corpus)
        results = matcher.match_batch("apple inc", ["apple inc", "banana inc", "cherry co"])
        assert len(results) == 3
        assert results[0][0] == "apple inc"

    def test_len(self) -> None:
        corpus = ["a", "b", "c"]
        matcher = reclink.TfIdfMatcher.fit(corpus)
        assert len(matcher) == 3


class TestRemoveStopWords:
    def test_basic(self) -> None:
        result = reclink.remove_stop_words("the cat and the dog")
        assert result == "cat dog"

    def test_no_stop_words(self) -> None:
        result = reclink.remove_stop_words("hello world")
        assert result == "hello world"

    def test_all_stop_words(self) -> None:
        result = reclink.remove_stop_words("the a an")
        assert result == ""


class TestExpandAbbreviations:
    def test_address(self) -> None:
        result = reclink.expand_abbreviations("123 Main St.")
        assert result == "123 Main street"

    def test_company(self) -> None:
        result = reclink.expand_abbreviations("Acme Inc.")
        assert result == "Acme incorporated"

    def test_no_abbreviations(self) -> None:
        result = reclink.expand_abbreviations("hello world")
        assert result == "hello world"


class TestCdist:
    def test_basic(self) -> None:
        a = ["Jon", "Jane"]
        b = ["John", "Janet"]
        matrix = reclink.cdist(a, b, scorer="jaro_winkler")
        assert matrix.shape == (2, 2)
        assert matrix.dtype == np.float64
        assert 0 <= matrix[0, 0] <= 1
        assert 0 <= matrix[1, 1] <= 1

    def test_different_scorers(self) -> None:
        a = ["hello"]
        b = ["hello", "world"]
        for scorer in [
            "jaro",
            "jaro_winkler",
            "cosine",
            "jaccard",
            "sorensen_dice",
            "token_sort",
            "token_set",
            "partial_ratio",
            "lcs",
            "longest_common_substring",
            "ngram_similarity",
            "smith_waterman",
        ]:
            matrix = reclink.cdist(a, b, scorer=scorer)
            assert matrix.shape == (1, 2)


class TestStripDiacritics:
    def test_cafe(self) -> None:
        assert reclink.strip_diacritics("café") == "cafe"

    def test_naive(self) -> None:
        assert reclink.strip_diacritics("naïve") == "naive"

    def test_ascii_passthrough(self) -> None:
        assert reclink.strip_diacritics("hello") == "hello"

    def test_empty(self) -> None:
        assert reclink.strip_diacritics("") == ""

    def test_uber(self) -> None:
        assert reclink.strip_diacritics("über") == "uber"


class TestRegexReplace:
    def test_simple(self) -> None:
        assert reclink.regex_replace("hello 123 world", r"\d+", "") == "hello  world"

    def test_replace_dashes(self) -> None:
        assert reclink.regex_replace("foo-bar-baz", r"-", " ") == "foo bar baz"

    def test_invalid_regex(self) -> None:
        with pytest.raises(ValueError):
            reclink.regex_replace("test", r"[invalid", "")


class TestCompositeScorer:
    def test_single_metric(self) -> None:
        scorer = reclink.CompositeScorer([("jaro_winkler", 1.0)])
        sim = scorer.similarity("hello", "hello")
        assert abs(sim - 1.0) < 0.001

    def test_uniform_weights_average(self) -> None:
        scorer = reclink.CompositeScorer([("jaro_winkler", 1.0), ("cosine", 1.0)])
        jw = reclink.jaro_winkler("hello", "hallo")
        cos = reclink.cosine("hello", "hallo")
        expected = (jw + cos) / 2.0
        assert abs(scorer.similarity("hello", "hallo") - expected) < 0.001

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            reclink.CompositeScorer([])

    def test_match_best(self) -> None:
        scorer = reclink.CompositeScorer([("jaro_winkler", 1.0)])
        result = scorer.match_best("hello", ["hello", "world"])
        assert result is not None
        assert result[0] == "hello"

    def test_match_batch(self) -> None:
        scorer = reclink.CompositeScorer([("jaro_winkler", 1.0)])
        results = scorer.match_batch("hello", ["hello", "world", "help"], limit=2)
        assert len(results) == 2
        assert results[0][0] == "hello"


class TestCompositeScorerPresets:
    def test_name_matching_preset(self) -> None:
        scorer = reclink.CompositeScorer.preset("name_matching")
        assert abs(scorer.similarity("John Smith", "John Smith") - 1.0) < 0.01

    def test_address_matching_preset(self) -> None:
        scorer = reclink.CompositeScorer.preset("address_matching")
        assert abs(scorer.similarity("123 Main St", "123 Main St") - 1.0) < 0.01

    def test_general_purpose_preset(self) -> None:
        scorer = reclink.CompositeScorer.preset("general_purpose")
        assert abs(scorer.similarity("hello", "hello") - 1.0) < 0.01

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError):
            reclink.CompositeScorer.preset("nonexistent")

    def test_preset_match_best(self) -> None:
        scorer = reclink.CompositeScorer.preset("name_matching")
        result = scorer.match_best("John Smith", ["John Smith", "Jane Doe"])
        assert result is not None
        assert result[0] == "John Smith"


class TestPresetsModule:
    def test_name_matching(self) -> None:
        from reclink.presets import name_matching

        scorer = name_matching()
        assert abs(scorer.similarity("John Smith", "John Smith") - 1.0) < 0.01

    def test_address_matching(self) -> None:
        from reclink.presets import address_matching

        scorer = address_matching()
        assert abs(scorer.similarity("123 Main St", "123 Main St") - 1.0) < 0.01

    def test_general_purpose(self) -> None:
        from reclink.presets import general_purpose

        scorer = general_purpose()
        assert abs(scorer.similarity("hello", "hello") - 1.0) < 0.01


class TestVpTree:
    def test_build_and_query(self) -> None:
        tree = reclink.VpTree.build(["hello", "hallo", "world", "help"], metric="jaro_winkler")
        results = tree.find_within("hello", max_distance=0.2)
        values = {r[0] for r in results}
        assert "hello" in values
        assert "hallo" in values

    def test_find_nearest(self) -> None:
        tree = reclink.VpTree.build(["apple", "apply", "ape", "banana"], metric="jaro_winkler")
        results = tree.find_nearest("apple", k=2)
        assert len(results) == 2
        assert results[0][0] == "apple"

    def test_len(self) -> None:
        tree = reclink.VpTree.build(["a", "b", "c"], metric="jaro_winkler")
        assert len(tree) == 3

    def test_empty(self) -> None:
        tree = reclink.VpTree.build([], metric="jaro_winkler")
        assert len(tree) == 0
        assert tree.find_within("hello", max_distance=0.5) == []


class TestNgramIndex:
    def test_build_and_search(self) -> None:
        index = reclink.NgramIndex.build(["hello", "help", "world"], n=2)
        results = index.search("hello", threshold=2)
        values = {r[0] for r in results}
        assert "hello" in values
        assert "help" in values

    def test_search_top_k(self) -> None:
        index = reclink.NgramIndex.build(["hello", "help", "world", "held"], n=2)
        results = index.search_top_k("hello", k=2)
        assert len(results) == 2
        assert results[0][0] == "hello"

    def test_len(self) -> None:
        index = reclink.NgramIndex.build(["a", "b", "c"], n=2)
        assert len(index) == 3

    def test_empty(self) -> None:
        index = reclink.NgramIndex.build([], n=2)
        assert len(index) == 0
        assert index.search("hello", threshold=1) == []


class TestIndexPersistence:
    def test_bk_tree_save_load(self, tmp_path: pytest.TempPathFactory) -> None:
        path = str(tmp_path / "bk.bin")  # type: ignore[operator]
        tree = reclink.BkTree.build(["hello", "hallo", "world"], metric="levenshtein")
        tree.save(path)
        loaded = reclink.BkTree.load(path)
        assert len(loaded) == 3
        results = loaded.find_within("hello", max_distance=1)
        values = {r[0] for r in results}
        assert "hello" in values
        assert "hallo" in values

    def test_vp_tree_save_load(self, tmp_path: pytest.TempPathFactory) -> None:
        path = str(tmp_path / "vp.bin")  # type: ignore[operator]
        tree = reclink.VpTree.build(["hello", "hallo", "world"], metric="jaro_winkler")
        tree.save(path)
        loaded = reclink.VpTree.load(path)
        assert len(loaded) == 3

    def test_ngram_index_save_load(self, tmp_path: pytest.TempPathFactory) -> None:
        path = str(tmp_path / "ngram.bin")  # type: ignore[operator]
        index = reclink.NgramIndex.build(["hello", "help", "world"], n=2)
        index.save(path)
        loaded = reclink.NgramIndex.load(path)
        assert len(loaded) == 3


class TestStreamingMatcher:
    def test_score_no_threshold(self) -> None:
        matcher = reclink.StreamingMatcher("hello")
        score = matcher.score("hello")
        assert score is not None
        assert abs(score - 1.0) < 0.001

    def test_score_with_threshold(self) -> None:
        matcher = reclink.StreamingMatcher("hello", threshold=0.9)
        assert matcher.score("hello") is not None
        assert matcher.score("xyz") is None

    def test_score_chunk(self) -> None:
        matcher = reclink.StreamingMatcher("hello", threshold=0.5)
        results = matcher.score_chunk(["hello", "xyz", "help"])
        assert len(results) >= 2

    def test_match_stream(self) -> None:
        from reclink.streaming import match_stream

        results = list(match_stream("hello", ["hello", "world", "help"], threshold=0.5))
        assert len(results) >= 2
        values = {r[0] for r in results}
        assert "hello" in values

    def test_match_stream_generator(self) -> None:
        from collections.abc import Generator

        from reclink.streaming import match_stream

        def gen() -> Generator[str, None, None]:
            yield "hello"
            yield "world"
            yield "help"

        results = list(match_stream("hello", gen(), threshold=0.5))
        assert len(results) >= 2
