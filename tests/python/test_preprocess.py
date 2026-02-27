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

    def test_preprocess_batch_clean_name(self) -> None:
        result = reclink.preprocess_batch(["Mr. John Smith Jr."], ["clean_name"])
        assert result == ["john smith"]

    def test_preprocess_batch_clean_address(self) -> None:
        result = reclink.preprocess_batch(["123 Main St."], ["clean_address"])
        assert result == ["123 main street"]

    def test_preprocess_batch_clean_company(self) -> None:
        result = reclink.preprocess_batch(["Acme Inc."], ["clean_company"])
        assert result == ["acme"]

    def test_preprocess_batch_normalize_email(self) -> None:
        result = reclink.preprocess_batch(["User.Name+tag@Gmail.COM"], ["normalize_email"])
        assert result == ["username@gmail.com"]

    def test_preprocess_batch_normalize_url(self) -> None:
        result = reclink.preprocess_batch(["HTTP://WWW.Example.COM/path/"], ["normalize_url"])
        assert result == ["http://example.com/path"]

    def test_preprocess_batch_synonym_expand(self) -> None:
        result = reclink.preprocess_batch(
            ["the big cat"],
            ['synonym_expand:{"big":"large","cat":"feline"}'],
        )
        assert result == ["the large feline"]


# ---------------------------------------------------------------------------
# Domain preprocessors
# ---------------------------------------------------------------------------


class TestCleanName:
    def test_basic(self) -> None:
        assert reclink.clean_name("Mr. John Smith Jr.") == "john smith"

    def test_comma_reorder(self) -> None:
        assert reclink.clean_name("Smith, John") == "john smith"

    def test_comma_reorder_middle(self) -> None:
        assert reclink.clean_name("Doe, Jane Marie") == "jane marie doe"

    def test_preserves_hyphens(self) -> None:
        assert reclink.clean_name("Mary-Jane Watson") == "mary-jane watson"

    def test_multiple_titles(self) -> None:
        assert reclink.clean_name("Dr. Prof. John Smith") == "john smith"

    def test_suffixes(self) -> None:
        assert reclink.clean_name("John Smith III") == "john smith"
        assert reclink.clean_name("Robert Jones Sr.") == "robert jones"

    def test_phd(self) -> None:
        assert reclink.clean_name("Jane Doe PhD") == "jane doe"


class TestCleanAddress:
    def test_street_expansion(self) -> None:
        assert reclink.clean_address("123 Main St.") == "123 main street"

    def test_avenue_expansion(self) -> None:
        assert reclink.clean_address("456 Oak Ave") == "456 oak avenue"

    def test_directionals(self) -> None:
        assert reclink.clean_address("100 N Main St") == "100 north main street"

    def test_northwest(self) -> None:
        result = reclink.clean_address("200 NW Elm Blvd")
        assert result == "200 northwest elm boulevard"

    def test_suite(self) -> None:
        result = reclink.clean_address("300 First St Ste 100")
        assert result == "300 first street suite 100"


class TestCleanCompany:
    def test_inc(self) -> None:
        assert reclink.clean_company("Acme Inc.") == "acme"

    def test_corporation(self) -> None:
        assert reclink.clean_company("Globex Corporation") == "globex"

    def test_ampersand(self) -> None:
        assert reclink.clean_company("Ben & Jerry's") == "ben and jerry's"

    def test_plus(self) -> None:
        assert reclink.clean_company("A + B Corp") == "a and b"

    def test_llc(self) -> None:
        assert reclink.clean_company("Foo LLC") == "foo"

    def test_gmbh(self) -> None:
        assert reclink.clean_company("Bar GmbH") == "bar"


# ---------------------------------------------------------------------------
# Email / URL normalization
# ---------------------------------------------------------------------------


class TestNormalizeEmail:
    def test_gmail_plus_and_dots(self) -> None:
        assert reclink.normalize_email("User.Name+tag@Gmail.COM") == "username@gmail.com"

    def test_googlemail_normalization(self) -> None:
        assert reclink.normalize_email("test@googlemail.com") == "test@gmail.com"

    def test_non_gmail(self) -> None:
        assert reclink.normalize_email("User@Example.COM") == "user@example.com"

    def test_no_at_sign(self) -> None:
        assert reclink.normalize_email("not-an-email") == "not-an-email"


class TestNormalizeUrl:
    def test_basic(self) -> None:
        result = reclink.normalize_url("HTTP://WWW.Example.COM/path/")
        assert result == "http://example.com/path"

    def test_default_port_http(self) -> None:
        assert reclink.normalize_url("http://example.com:80/path") == "http://example.com/path"

    def test_default_port_https(self) -> None:
        assert reclink.normalize_url("https://example.com:443/path") == "https://example.com/path"

    def test_non_default_port(self) -> None:
        assert (
            reclink.normalize_url("http://example.com:8080/path") == "http://example.com:8080/path"
        )

    def test_sort_query_params(self) -> None:
        assert (
            reclink.normalize_url("http://example.com/path?z=1&a=2")
            == "http://example.com/path?a=2&z=1"
        )

    def test_remove_fragment(self) -> None:
        assert reclink.normalize_url("http://example.com/path#section") == "http://example.com/path"

    def test_empty(self) -> None:
        assert reclink.normalize_url("") == ""


# ---------------------------------------------------------------------------
# Synonym expansion
# ---------------------------------------------------------------------------


class TestSynonymExpand:
    def test_basic(self) -> None:
        result = reclink.synonym_expand("the big cat", {"big": "large", "cat": "feline"})
        assert result == "the large feline"

    def test_case_insensitive(self) -> None:
        result = reclink.synonym_expand("The Big Cat", {"big": "large"})
        assert result == "The large Cat"

    def test_no_match(self) -> None:
        result = reclink.synonym_expand("hello world", {"foo": "bar"})
        assert result == "hello world"

    def test_empty_table(self) -> None:
        result = reclink.synonym_expand("hello world", {})
        assert result == "hello world"
