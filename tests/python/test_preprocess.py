"""Tests for preprocessing functions.

Organized into sections:
- TestBasicPreprocess: Existing scalar functions
- TestTokenization: ngram and whitespace tokenizers
- TestUnicodeNormalization: Unicode normalization
- TestBatchPreprocess: Parallel batch operations
"""

import reclink

# ---------------------------------------------------------------------------
# Basic preprocessing
# ---------------------------------------------------------------------------


class TestBasicPreprocess:
    def test_fold_case(self) -> None:
        assert reclink.fold_case("Hello WORLD") == "hello world"

    def test_normalize_whitespace(self) -> None:
        assert reclink.normalize_whitespace("  hello   world  ") == "hello world"

    def test_strip_punctuation(self) -> None:
        assert reclink.strip_punctuation("hello, world!") == "hello world"

    def test_standardize_name(self) -> None:
        assert reclink.standardize_name("St. Louis") == "saint louis"
        assert reclink.standardize_name("Dr. Smith") == "doctor smith"


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


class TestTokenization:
    def test_ngram_tokenize_bigrams(self) -> None:
        result = reclink.ngram_tokenize("hello", 2)
        assert result == ["he", "el", "ll", "lo"]

    def test_ngram_tokenize_trigrams(self) -> None:
        result = reclink.ngram_tokenize("hello", 3)
        assert result == ["hel", "ell", "llo"]

    def test_ngram_tokenize_too_short(self) -> None:
        result = reclink.ngram_tokenize("a", 2)
        assert result == []

    def test_ngram_tokenize_default_n(self) -> None:
        result = reclink.ngram_tokenize("abc")
        assert result == ["ab", "bc"]

    def test_whitespace_tokenize(self) -> None:
        result = reclink.whitespace_tokenize("hello  beautiful world")
        assert result == ["hello", "beautiful", "world"]

    def test_whitespace_tokenize_empty(self) -> None:
        result = reclink.whitespace_tokenize("")
        assert result == []

    def test_whitespace_tokenize_single_word(self) -> None:
        result = reclink.whitespace_tokenize("hello")
        assert result == ["hello"]


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------


class TestUnicodeNormalization:
    def test_nfkc_default(self) -> None:
        result = reclink.normalize_unicode("café")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_nfc(self) -> None:
        # Composed form: é = U+00E9
        result = reclink.normalize_unicode("caf\u0065\u0301", "nfc")
        assert result == "caf\u00e9"

    def test_nfd(self) -> None:
        result = reclink.normalize_unicode("caf\u00e9", "nfd")
        # NFD decomposes: é → e + combining accent
        assert len(result) == 5

    def test_nfkd(self) -> None:
        result = reclink.normalize_unicode("café", "nfkd")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Batch preprocessing
# ---------------------------------------------------------------------------


class TestBatchPreprocess:
    def test_preprocess_batch_fold_case(self) -> None:
        result = reclink.preprocess_batch(["Hello", "WORLD"], ["fold_case"])
        assert result == ["hello", "world"]

    def test_preprocess_batch_multiple_ops(self) -> None:
        result = reclink.preprocess_batch(
            ["  Hello,  WORLD!  "],
            ["fold_case", "strip_punctuation", "normalize_whitespace"],
        )
        assert result == ["hello world"]

    def test_preprocess_batch_empty(self) -> None:
        result = reclink.preprocess_batch([], ["fold_case"])
        assert result == []

    def test_preprocess_batch_unicode(self) -> None:
        result = reclink.preprocess_batch(["CAF\u00c9"], ["normalize_unicode_nfkc", "fold_case"])
        assert result == ["café"]

    def test_ngram_tokenize_batch(self) -> None:
        result = reclink.ngram_tokenize_batch(["hello", "world"], 2)
        assert result == [["he", "el", "ll", "lo"], ["wo", "or", "rl", "ld"]]

    def test_ngram_tokenize_batch_default_n(self) -> None:
        result = reclink.ngram_tokenize_batch(["ab", "cd"])
        assert result == [["ab"], ["cd"]]

    def test_whitespace_tokenize_batch(self) -> None:
        result = reclink.whitespace_tokenize_batch(["hello world", "foo bar baz"])
        assert result == [["hello", "world"], ["foo", "bar", "baz"]]

    def test_whitespace_tokenize_batch_empty(self) -> None:
        result = reclink.whitespace_tokenize_batch([])
        assert result == []
