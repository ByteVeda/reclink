"""Batch Operations — Upload CSV files for batch matching and similarity matrices."""

import io

import streamlit as st

st.set_page_config(page_title="Batch Operations — reclink", page_icon="📊", layout="wide")
st.title("Batch Operations")
st.markdown("Run batch matching, compute similarity matrices, and pairwise comparisons.")

import pandas as pd  # noqa: E402
import reclink  # noqa: E402

SCORERS = [
    "jaro_winkler",
    "jaro",
    "levenshtein_similarity",
    "damerau_levenshtein_similarity",
    "cosine",
    "jaccard",
    "sorensen_dice",
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "ngram_similarity",
    "lcs_similarity",
    "smith_waterman_similarity",
]

tab1, tab2, tab3 = st.tabs(["Match Best / Batch", "Similarity Matrix (cdist)", "Pairwise"])

# --- Match best/batch ---
with tab1:
    query = st.text_input("Query string", value="Jon Smith")

    candidates_mode = st.radio(
        "Candidates input", ["Type manually", "Upload CSV"], horizontal=True
    )

    candidates = []
    if candidates_mode == "Type manually":
        candidates_text = st.text_area(
            "Candidates (one per line)",
            value="John Smith\nJane Doe\nJon Smyth\nJames Smith\nJanet Smith",
        )
        candidates = [c.strip() for c in candidates_text.split("\n") if c.strip()]
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="match_csv")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            col = st.selectbox("Column", df.columns, key="match_col")
            candidates = df[col].dropna().astype(str).tolist()
            st.write(f"Loaded {len(candidates)} candidates")

    match_col1, match_col2, match_col3 = st.columns(3)
    with match_col1:
        scorer = st.selectbox("Scorer", SCORERS, key="match_scorer")
    with match_col2:
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05, key="match_threshold")
    with match_col3:
        limit = st.number_input("Max results", 1, 100, 10, key="match_limit")

    col_best, col_batch = st.columns(2)
    with col_best:
        if st.button("Find Best Match", type="primary", key="best"):
            if candidates:
                result = reclink.match_best(
                    query, candidates, scorer=scorer, threshold=threshold
                )
                if result:
                    st.metric("Best Match", result[0])
                    st.metric("Score", f"{result[1]:.4f}")
                    st.metric("Index", result[2])
                else:
                    st.warning("No match found above threshold.")

    with col_batch:
        if st.button("Find All Matches", type="primary", key="batch"):
            if candidates:
                results = reclink.match_batch(
                    query,
                    candidates,
                    scorer=scorer,
                    threshold=threshold,
                    limit=limit,
                )
                if results:
                    df_results = pd.DataFrame(
                        results, columns=["Match", "Score", "Index"]
                    )
                    st.dataframe(df_results, use_container_width=True)
                else:
                    st.warning("No matches found above threshold.")

# --- cdist ---
with tab2:
    st.markdown("Compute an all-pairs similarity matrix between two sets of strings.")

    cdist_mode = st.radio(
        "Input mode", ["Type manually", "Upload CSV"], horizontal=True, key="cdist_mode"
    )

    sources = []
    targets = []

    if cdist_mode == "Type manually":
        c1, c2 = st.columns(2)
        with c1:
            sources_text = st.text_area(
                "Source strings (one per line)", value="Jon\nJane\nBob", key="cdist_src"
            )
            sources = [s.strip() for s in sources_text.split("\n") if s.strip()]
        with c2:
            targets_text = st.text_area(
                "Target strings (one per line)",
                value="John\nJanet\nRobert",
                key="cdist_tgt",
            )
            targets = [s.strip() for s in targets_text.split("\n") if s.strip()]
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="cdist_csv")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            c1, c2 = st.columns(2)
            with c1:
                src_col = st.selectbox("Source column", df.columns, key="cdist_src_col")
                sources = df[src_col].dropna().astype(str).tolist()
            with c2:
                tgt_col = st.selectbox("Target column", df.columns, key="cdist_tgt_col")
                targets = df[tgt_col].dropna().astype(str).tolist()

    cdist_scorer = st.selectbox("Scorer", SCORERS, key="cdist_scorer")

    if st.button("Compute Matrix", type="primary", key="cdist"):
        if sources and targets:
            if len(sources) > 200 or len(targets) > 200:
                st.warning("Large matrices may take a while. Consider reducing input size.")

            matrix = reclink.cdist(sources, targets, scorer=cdist_scorer)
            df_matrix = pd.DataFrame(matrix, index=sources, columns=targets)

            st.subheader("Similarity Matrix")
            st.dataframe(
                df_matrix.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                use_container_width=True,
            )

            # Download
            csv_data = df_matrix.to_csv()
            st.download_button(
                "Download as CSV",
                csv_data,
                file_name="cdist_matrix.csv",
                mime="text/csv",
            )

# --- Pairwise ---
with tab3:
    st.markdown(
        "Compute element-wise similarity between two equal-length lists of strings."
    )

    pw_mode = st.radio(
        "Input mode", ["Type manually", "Upload CSV"], horizontal=True, key="pw_mode"
    )

    list_a = []
    list_b = []

    if pw_mode == "Type manually":
        c1, c2 = st.columns(2)
        with c1:
            a_text = st.text_area(
                "List A (one per line)", value="Jon\nJane\nBob", key="pw_a"
            )
            list_a = [s.strip() for s in a_text.split("\n") if s.strip()]
        with c2:
            b_text = st.text_area(
                "List B (one per line)", value="John\nJanet\nRobert", key="pw_b"
            )
            list_b = [s.strip() for s in b_text.split("\n") if s.strip()]
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="pw_csv")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            c1, c2 = st.columns(2)
            with c1:
                a_col = st.selectbox("Column A", df.columns, key="pw_a_col")
                list_a = df[a_col].dropna().astype(str).tolist()
            with c2:
                b_col = st.selectbox("Column B", df.columns, key="pw_b_col")
                list_b = df[b_col].dropna().astype(str).tolist()

    pw_scorer = st.selectbox("Scorer", SCORERS, key="pw_scorer")

    if st.button("Compute Pairwise", type="primary", key="pairwise"):
        if list_a and list_b:
            if len(list_a) != len(list_b):
                st.error("Lists must have the same length for pairwise comparison.")
            else:
                scores = reclink.pairwise_similarity(list_a, list_b, scorer=pw_scorer)
                df_result = pd.DataFrame(
                    {"String A": list_a, "String B": list_b, "Score": scores}
                )
                st.dataframe(df_result, use_container_width=True)

                csv_buf = io.StringIO()
                df_result.to_csv(csv_buf, index=False)
                st.download_button(
                    "Download Results",
                    csv_buf.getvalue(),
                    file_name="pairwise_scores.csv",
                    mime="text/csv",
                )
