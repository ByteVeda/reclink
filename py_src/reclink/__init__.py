"""reclink: Blazing-fast fuzzy matching and record linkage library.

Built on Rust via PyO3 for maximum performance.
"""

import contextlib
from typing import TYPE_CHECKING

from reclink import async_api as async_api
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
    PyBoundedStreamingMatcher as BoundedStreamingMatcher,
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
    # CJK tokenization
    character_tokenize,
    character_tokenize_batch,
    cjk_ngram_tokenize,
    cjk_ngram_tokenize_batch,
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
    # Custom plugins
    list_custom_blockers,
    list_custom_classifiers,
    list_custom_comparators,
    list_custom_metrics,
    list_custom_preprocessors,
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
    # Arabic/Hebrew preprocessing
    normalize_arabic,
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
    register_blocker,
    register_classifier,
    register_comparator,
    register_preprocessor,
    remove_stop_words,
    set_max_string_length,
    smart_tokenize,
    smart_tokenize_batch,
    smart_tokenize_ngram,
    smart_tokenize_ngram_batch,
    smith_waterman,
    smith_waterman_similarity,
    sorensen_dice,
    soundex,
    standardize_name,
    strip_arabic_diacritics,
    strip_bidi_marks,
    strip_diacritics,
    strip_hebrew_diacritics,
    strip_punctuation,
    synonym_expand,
    token_set_ratio,
    token_sort_ratio,
    # Transliteration
    transliterate_arabic,
    transliterate_cyrillic,
    transliterate_devanagari,
    transliterate_greek,
    transliterate_hangul,
    transliterate_hebrew,
    unregister_blocker,
    unregister_classifier,
    unregister_comparator,
    unregister_metric,
    unregister_preprocessor,
    weighted_levenshtein,
    weighted_levenshtein_similarity,
    whitespace_tokenize,
    whitespace_tokenize_batch,
)
from reclink._core import (
    estimate_fellegi_sunter_params as estimate_fellegi_sunter,
)
from reclink._core import (
    register_metric as _register_metric_raw,
)

if TYPE_CHECKING:
    from reclink._core import (
        CompositePreset as CompositePreset,
    )
    from reclink._core import (
        DateResolution as DateResolution,
    )
    from reclink._core import (
        Linkage as Linkage,
    )
    from reclink._core import (
        NormalizationForm as NormalizationForm,
    )
    from reclink._core import (
        PhoneticAlgorithm as PhoneticAlgorithm,
    )
    from reclink._core import (
        Scorer as Scorer,
    )


def register_metric(name_or_func: object = None, func: object = None) -> object:
    """Register a custom similarity metric.

    Supports three calling conventions:

    1. ``@register_metric("name")`` — decorator with explicit name
    2. ``@register_metric`` — decorator using the function's ``__name__``
    3. ``register_metric("name", func)`` — direct call

    Parameters
    ----------
    name_or_func : str or callable
        Either a name string or the function itself.
    func : callable, optional
        The metric function when called as ``register_metric("name", func)``.

    Returns
    -------
    callable
        The original function (unchanged) when used as a decorator.
    """
    if func is not None:
        # Direct call: register_metric("name", func)
        _register_metric_raw(str(name_or_func), func)
        return func

    if callable(name_or_func):
        # Bare decorator: @register_metric
        fn = name_or_func
        _register_metric_raw(fn.__name__, fn)  # type: ignore[attr-defined]
        return fn

    # Decorator with name: @register_metric("name")
    name = str(name_or_func)

    def _decorator(fn: object) -> object:
        _register_metric_raw(name, fn)
        return fn

    return _decorator


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
    "BoundedStreamingMatcher",
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
    "async_api",
    "beider_morse",
    "benchmark",
    "caverphone",
    "cdist",
    "cdist_arrow",
    "character_tokenize",
    "character_tokenize_batch",
    "cjk_ngram_tokenize",
    "cjk_ngram_tokenize_batch",
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
    "list_custom_blockers",
    "list_custom_classifiers",
    "list_custom_comparators",
    "list_custom_metrics",
    "list_custom_preprocessors",
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
    "normalize_arabic",
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
    "register_blocker",
    "register_classifier",
    "register_comparator",
    "register_metric",
    "register_preprocessor",
    "remove_stop_words",
    "set_max_string_length",
    "smart_tokenize",
    "smart_tokenize_batch",
    "smart_tokenize_ngram",
    "smart_tokenize_ngram_batch",
    "smith_waterman",
    "smith_waterman_similarity",
    "sorensen_dice",
    "soundex",
    "standardize_name",
    "streaming",
    "strip_arabic_diacritics",
    "strip_bidi_marks",
    "strip_diacritics",
    "strip_hebrew_diacritics",
    "strip_punctuation",
    "synonym_expand",
    "token_set_ratio",
    "token_sort_ratio",
    "transliterate_arabic",
    "transliterate_cyrillic",
    "transliterate_devanagari",
    "transliterate_greek",
    "transliterate_hangul",
    "transliterate_hebrew",
    "unregister_blocker",
    "unregister_classifier",
    "unregister_comparator",
    "unregister_metric",
    "unregister_preprocessor",
    "utils",
    "weighted_levenshtein",
    "weighted_levenshtein_similarity",
    "whitespace_tokenize",
    "whitespace_tokenize_batch",
]
