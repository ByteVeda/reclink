"""reclink Interactive Playground — Home page."""

import streamlit as st

st.set_page_config(
    page_title="reclink Playground",
    page_icon="🔗",
    layout="wide",
)

st.title("reclink Interactive Playground")
st.markdown(
    """
Explore **reclink**'s fuzzy matching and record linkage capabilities
interactively. Choose a tool from the sidebar to get started.

---

### Quick Try

Compare two strings with any metric:
"""
)

col1, col2 = st.columns(2)
with col1:
    string_a = st.text_input("String A", value="Jon Smith")
with col2:
    string_b = st.text_input("String B", value="John Smyth")

import reclink  # noqa: E402

METRICS = [
    "jaro_winkler",
    "jaro",
    "levenshtein_similarity",
    "damerau_levenshtein_similarity",
    "hamming_similarity",
    "cosine",
    "jaccard",
    "sorensen_dice",
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "ngram_similarity",
    "lcs_similarity",
    "longest_common_substring_similarity",
    "smith_waterman_similarity",
]

metric_name = st.selectbox("Metric", METRICS, index=0)

if st.button("Compare", type="primary"):
    fn = getattr(reclink, metric_name)
    score = fn(string_a, string_b)
    st.metric(label=f"{metric_name} score", value=f"{score:.4f}")

    # Show explain output
    st.subheader("All metrics breakdown")
    result = reclink.explain(string_a, string_b)
    cols = st.columns(3)
    for i, (name, value) in enumerate(sorted(result.items())):
        with cols[i % 3]:
            if isinstance(value, float):
                st.metric(name, f"{value:.4f}")
            else:
                st.metric(name, str(value))

st.markdown("---")

st.markdown(
    """
### Available Tools

| Page | Description |
|------|-------------|
| **String Metrics** | Compare two strings with 20+ metrics side-by-side |
| **Phonetic Algorithms** | Encode strings with 7 phonetic algorithms |
| **Preprocessing** | Build and test text preprocessing pipelines |
| **Batch Operations** | Upload CSV files for batch matching and cdist heatmaps |
| **Indexes** | Build and query BK-tree, VP-tree, N-gram, and MinHash indexes |
| **Record Linkage** | Run full deduplication and linkage pipelines on your data |
"""
)
