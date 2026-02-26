"""reclink: Blazing-fast fuzzy matching and record linkage library.

Built on Rust via PyO3 for maximum performance.
"""

from reclink import evaluation as evaluation
from reclink import export as export
from reclink import metrics as metrics
from reclink import phonetic as phonetic
from reclink import pipeline as pipeline
from reclink._core import (
    PyEmResult as EmResult,
)
from reclink._core import (
    PyMatchResult as MatchResult,
)
from reclink._core import (
    PyPipeline as Pipeline,
)
from reclink._core import (
    # Pipeline
    PyRecord as Record,
)
from reclink._core import (
    cdist,
    cosine,
    damerau_levenshtein,
    damerau_levenshtein_similarity,
    double_metaphone,
    # Preprocessing
    fold_case,
    hamming,
    hamming_similarity,
    jaccard,
    jaro,
    jaro_winkler,
    # String metrics
    levenshtein,
    levenshtein_similarity,
    metaphone,
    # Tokenization & Unicode normalization
    ngram_tokenize,
    ngram_tokenize_batch,
    normalize_unicode,
    normalize_whitespace,
    nysiis,
    # Batch preprocessing
    preprocess_batch,
    sorensen_dice,
    # Phonetic
    soundex,
    standardize_name,
    strip_punctuation,
    whitespace_tokenize,
    whitespace_tokenize_batch,
)
from reclink._core import (
    estimate_fellegi_sunter_params as estimate_fellegi_sunter,
)

__version__ = "0.1.0"

__all__ = [
    "EmResult",
    "MatchResult",
    "Pipeline",
    # Pipeline
    "Record",
    "cdist",
    "cosine",
    "damerau_levenshtein",
    "damerau_levenshtein_similarity",
    "double_metaphone",
    "estimate_fellegi_sunter",
    "evaluation",
    "export",
    # Preprocessing
    "fold_case",
    "hamming",
    "hamming_similarity",
    "jaccard",
    "jaro",
    "jaro_winkler",
    # Metrics
    "levenshtein",
    "levenshtein_similarity",
    "metaphone",
    # Submodules
    "metrics",
    # Tokenization & Unicode normalization
    "ngram_tokenize",
    "ngram_tokenize_batch",
    "normalize_unicode",
    "normalize_whitespace",
    "nysiis",
    "phonetic",
    "pipeline",
    # Batch preprocessing
    "preprocess_batch",
    "sorensen_dice",
    # Phonetic
    "soundex",
    "standardize_name",
    "strip_punctuation",
    "whitespace_tokenize",
    "whitespace_tokenize_batch",
]
