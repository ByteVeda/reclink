"""reclink: Blazing-fast fuzzy matching and record linkage library.

Built on Rust via PyO3 for maximum performance.
"""

import contextlib

from reclink import benchmark as benchmark
from reclink import evaluation as evaluation
from reclink import export as export
from reclink import metrics as metrics
from reclink import phonetic as phonetic
from reclink import pipeline as pipeline
from reclink import presets as presets
from reclink import streaming as streaming
from reclink import utils as utils
from reclink._core import (
    PyBkTree as BkTree,
)
from reclink._core import (
    PyCompositeScorer as CompositeScorer,
)
from reclink._core import (
    PyEmResult as EmResult,
)
from reclink._core import (
    PyMatchResult as MatchResult,
)
from reclink._core import (
    PyMinHashIndex as MinHashIndex,
)
from reclink._core import (
    PyMmapNgramIndex as MmapNgramIndex,
)
from reclink._core import (
    PyNgramIndex as NgramIndex,
)
from reclink._core import (
    PyPipeline as Pipeline,
)
from reclink._core import (
    # Pipeline
    PyRecord as Record,
)
from reclink._core import (
    PyStreamingMatcher as StreamingMatcher,
)
from reclink._core import (
    PyTfIdfMatcher as TfIdfMatcher,
)
from reclink._core import (
    PyVpTree as VpTree,
)
from reclink._core import (
    beider_morse,
    # Phonetic
    caverphone,
    cdist,
    # Arrow-friendly batch operations
    cdist_arrow,
    # Domain preprocessors
    clean_address,
    clean_company,
    clean_name,
    cologne_phonetic,
    cosine,
    damerau_levenshtein,
    damerau_levenshtein_similarity,
    damerau_levenshtein_threshold,
    detect_language,
    double_metaphone,
    # Preprocessing
    expand_abbreviations,
    explain,
    fold_case,
    get_max_string_length,
    hamming,
    hamming_similarity,
    jaccard,
    jaro,
    jaro_winkler,
    lcs_length,
    lcs_similarity,
    # String metrics
    levenshtein,
    levenshtein_align,
    levenshtein_similarity,
    levenshtein_threshold,
    longest_common_substring_length,
    longest_common_substring_similarity,
    # Batch matching
    match_batch,
    match_batch_arrow,
    match_best,
    match_best_arrow,
    metaphone,
    # Tokenization & Unicode normalization
    ngram_similarity,
    ngram_tokenize,
    ngram_tokenize_batch,
    normalize_email,
    normalize_unicode,
    normalize_url,
    normalize_whitespace,
    nysiis,
    pairwise_similarity,
    partial_ratio,
    phonetic_batch_arrow,
    phonetic_hybrid,
    # Batch preprocessing
    preprocess_batch,
    regex_replace,
    remove_stop_words,
    set_max_string_length,
    smith_waterman,
    smith_waterman_similarity,
    sorensen_dice,
    soundex,
    standardize_name,
    strip_diacritics,
    strip_punctuation,
    synonym_expand,
    token_set_ratio,
    token_sort_ratio,
    # Transliteration
    transliterate_cyrillic,
    transliterate_greek,
    weighted_levenshtein,
    weighted_levenshtein_similarity,
    whitespace_tokenize,
    whitespace_tokenize_batch,
)
from reclink._core import (
    estimate_fellegi_sunter_params as estimate_fellegi_sunter,
)


def _register_pandas_accessor() -> None:
    with contextlib.suppress(ImportError):
        from reclink._pandas_accessor import (  # noqa: F401
            ReclinkDataFrameAccessor,
            ReclinkSeriesAccessor,
        )


def _register_polars_accessor() -> None:
    with contextlib.suppress(ImportError):
        from reclink._polars_accessor import (  # noqa: F401
            ReclinkDataFrameNamespace,
            ReclinkNamespace,
        )


_register_pandas_accessor()
_register_polars_accessor()

__version__ = "0.1.0"

__all__ = [
    "BkTree",
    "CompositeScorer",
    "EmResult",
    "MatchResult",
    "MinHashIndex",
    "MmapNgramIndex",
    "NgramIndex",
    "Pipeline",
    "Record",
    "StreamingMatcher",
    "TfIdfMatcher",
    "VpTree",
    "beider_morse",
    "benchmark",
    "caverphone",
    "cdist",
    "cdist_arrow",
    "clean_address",
    "clean_company",
    "clean_name",
    "cologne_phonetic",
    "cosine",
    "damerau_levenshtein",
    "damerau_levenshtein_similarity",
    "damerau_levenshtein_threshold",
    "detect_language",
    "double_metaphone",
    "estimate_fellegi_sunter",
    "evaluation",
    "expand_abbreviations",
    "explain",
    "export",
    "fold_case",
    "get_max_string_length",
    "hamming",
    "hamming_similarity",
    "jaccard",
    "jaro",
    "jaro_winkler",
    "lcs_length",
    "lcs_similarity",
    "levenshtein",
    "levenshtein_align",
    "levenshtein_similarity",
    "levenshtein_threshold",
    "longest_common_substring_length",
    "longest_common_substring_similarity",
    "match_batch",
    "match_batch_arrow",
    "match_best",
    "match_best_arrow",
    "metaphone",
    "metrics",
    "ngram_similarity",
    "ngram_tokenize",
    "ngram_tokenize_batch",
    "normalize_email",
    "normalize_unicode",
    "normalize_url",
    "normalize_whitespace",
    "nysiis",
    "pairwise_similarity",
    "partial_ratio",
    "phonetic",
    "phonetic_batch_arrow",
    "phonetic_hybrid",
    "pipeline",
    "preprocess_batch",
    "presets",
    "regex_replace",
    "remove_stop_words",
    "set_max_string_length",
    "smith_waterman",
    "smith_waterman_similarity",
    "sorensen_dice",
    "soundex",
    "standardize_name",
    "streaming",
    "strip_diacritics",
    "strip_punctuation",
    "synonym_expand",
    "token_set_ratio",
    "token_sort_ratio",
    "transliterate_cyrillic",
    "transliterate_greek",
    "utils",
    "weighted_levenshtein",
    "weighted_levenshtein_similarity",
    "whitespace_tokenize",
    "whitespace_tokenize_batch",
]
