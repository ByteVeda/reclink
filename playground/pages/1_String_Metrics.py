"""String Metrics — Compare two strings with any metric."""

import streamlit as st

st.set_page_config(page_title="String Metrics — reclink", page_icon="📏", layout="wide")
st.title("String Metrics")
st.markdown("Compare two strings using 20+ similarity and distance metrics.")

import reclink  # noqa: E402

# --- Inputs ---
col1, col2 = st.columns(2)
with col1:
    string_a = st.text_input("String A", value="Jon Smith")
with col2:
    string_b = st.text_input("String B", value="John Smyth")

tab1, tab2, tab3 = st.tabs(["Single Metric", "Compare All", "Alignment"])

# --- Single metric ---
with tab1:
    SIMILARITY_METRICS = [
        "jaro_winkler",
        "jaro",
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
        "levenshtein_similarity",
        "damerau_levenshtein_similarity",
        "hamming_similarity",
        "phonetic_hybrid",
    ]

    DISTANCE_METRICS = [
        "levenshtein",
        "damerau_levenshtein",
        "hamming",
        "lcs_length",
        "longest_common_substring_length",
    ]

    metric_type = st.radio("Type", ["Similarity", "Distance"], horizontal=True)

    if metric_type == "Similarity":
        metric_name = st.selectbox("Metric", SIMILARITY_METRICS)
    else:
        metric_name = st.selectbox("Metric", DISTANCE_METRICS)

    if st.button("Calculate", type="primary", key="single"):
        fn = getattr(reclink, metric_name)
        try:
            result = fn(string_a, string_b)
            if metric_type == "Similarity":
                st.metric(f"{metric_name}", f"{result:.4f}")
                st.progress(float(result))
            else:
                st.metric(f"{metric_name}", str(result))
        except Exception as e:
            st.error(f"Error: {e}")

    # Early termination
    st.markdown("---")
    st.subheader("Early Termination")
    st.markdown("Check if distance is within a threshold (returns `None` if exceeded).")
    threshold_col1, threshold_col2 = st.columns(2)
    with threshold_col1:
        max_dist = st.number_input("Max distance", min_value=0, value=2, step=1)
    with threshold_col2:
        threshold_metric = st.selectbox(
            "Threshold metric",
            ["levenshtein_threshold", "damerau_levenshtein_threshold"],
        )
    if st.button("Check Threshold", key="threshold"):
        fn = getattr(reclink, threshold_metric)
        result = fn(string_a, string_b, max_dist)
        if result is None:
            st.warning(f"Distance exceeds threshold of {max_dist}")
        else:
            st.success(f"Distance = {result} (within threshold of {max_dist})")

# --- Compare all ---
with tab2:
    if st.button("Compare All Metrics", type="primary", key="all"):
        result = reclink.explain(string_a, string_b)
        similarity_results = {}
        distance_results = {}
        for name, value in sorted(result.items()):
            if isinstance(value, float) and 0 <= value <= 1:
                similarity_results[name] = value
            else:
                distance_results[name] = value

        if similarity_results:
            st.subheader("Similarity Metrics")
            cols = st.columns(3)
            for i, (name, value) in enumerate(
                sorted(similarity_results.items(), key=lambda x: x[1], reverse=True)
            ):
                with cols[i % 3]:
                    st.metric(name, f"{value:.4f}")

        if distance_results:
            st.subheader("Distance Metrics")
            cols = st.columns(3)
            for i, (name, value) in enumerate(sorted(distance_results.items())):
                with cols[i % 3]:
                    st.metric(name, str(value))

# --- Alignment ---
with tab3:
    st.markdown("Visualize the edit-distance alignment between two strings.")
    if st.button("Show Alignment", type="primary", key="align"):
        result = reclink.levenshtein_align(string_a, string_b)
        st.code(result["visual"], language=None)
        st.metric("Edit Distance", result["distance"])
        st.markdown("**Operations:**")
        for op in result["ops"]:
            if op.startswith("match"):
                st.markdown(f"- ✅ {op}")
            elif op.startswith("sub"):
                st.markdown(f"- 🔄 {op}")
            elif op.startswith("ins"):
                st.markdown(f"- ➕ {op}")
            elif op.startswith("del"):
                st.markdown(f"- ➖ {op}")
            else:
                st.markdown(f"- {op}")
