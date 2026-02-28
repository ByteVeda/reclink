"""Indexes — Build and query BK-tree, VP-tree, N-gram, and MinHash indexes."""

import streamlit as st

st.set_page_config(page_title="Indexes — reclink", page_icon="🌲", layout="wide")
st.title("Index Structures")
st.markdown(
    "Build search indexes for sub-linear nearest-neighbor queries on string collections."
)

import pandas as pd  # noqa: E402
import reclink  # noqa: E402

# --- Input ---
st.subheader("Input Data")
input_mode = st.radio("Input mode", ["Type manually", "Upload CSV"], horizontal=True)

strings = []
if input_mode == "Type manually":
    strings_text = st.text_area(
        "Strings (one per line)",
        value="smith\nsmyth\njohn\njane\njohnson\njanet\njonathan\njames",
        height=150,
    )
    strings = [s.strip() for s in strings_text.split("\n") if s.strip()]
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="idx_csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        col = st.selectbox("Column", df.columns)
        strings = df[col].dropna().astype(str).tolist()
        st.write(f"Loaded {len(strings)} strings")

if not strings:
    st.info("Enter or upload some strings to build an index.")
    st.stop()

st.write(f"**{len(strings)} strings loaded**")

# --- Index type ---
tab1, tab2, tab3, tab4 = st.tabs(["BK-Tree", "VP-Tree", "N-gram Index", "MinHash/LSH"])

# --- BK-tree ---
with tab1:
    st.markdown("BK-trees support exact threshold search with edit-distance metrics.")
    bk_metric = st.selectbox(
        "Metric", ["levenshtein", "damerau_levenshtein", "hamming"], key="bk_metric"
    )

    if st.button("Build BK-Tree", type="primary", key="bk_build"):
        tree = reclink.BkTree.build(strings, metric=bk_metric)
        st.session_state["bk_tree"] = tree
        st.success(f"BK-tree built with {len(strings)} strings")

    if "bk_tree" in st.session_state:
        tree = st.session_state["bk_tree"]

        st.markdown("---")
        query = st.text_input("Query string", value="smith", key="bk_query")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Find Within Distance")
            max_distance = st.number_input("Max distance", 0, 10, 2, key="bk_max")
            if st.button("Search", key="bk_within"):
                results = tree.find_within(query, max_distance=max_distance)
                if results:
                    df_r = pd.DataFrame(results, columns=["String", "Distance", "Index"])
                    st.dataframe(df_r, use_container_width=True)
                else:
                    st.info("No results found.")

        with c2:
            st.subheader("Find Nearest")
            k = st.number_input("k", 1, 20, 3, key="bk_k")
            if st.button("Search", key="bk_nearest"):
                results = tree.find_nearest(query, k=k)
                if results:
                    df_r = pd.DataFrame(results, columns=["String", "Distance", "Index"])
                    st.dataframe(df_r, use_container_width=True)
                else:
                    st.info("No results found.")

# --- VP-tree ---
with tab2:
    st.markdown("VP-trees support any metric, with k-nearest and range search.")
    vp_metric = st.selectbox(
        "Metric",
        ["jaro_winkler", "jaro", "levenshtein_similarity", "cosine"],
        key="vp_metric",
    )

    if st.button("Build VP-Tree", type="primary", key="vp_build"):
        tree = reclink.VpTree.build(strings, metric=vp_metric)
        st.session_state["vp_tree"] = tree
        st.success(f"VP-tree built with {len(strings)} strings")

    if "vp_tree" in st.session_state:
        tree = st.session_state["vp_tree"]

        st.markdown("---")
        query = st.text_input("Query string", value="smith", key="vp_query")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Find Nearest")
            k = st.number_input("k", 1, 20, 3, key="vp_k")
            if st.button("Search", key="vp_nearest"):
                results = tree.find_nearest(query, k=k)
                if results:
                    df_r = pd.DataFrame(results, columns=["String", "Distance", "Index"])
                    st.dataframe(df_r, use_container_width=True)
                else:
                    st.info("No results found.")

        with c2:
            st.subheader("Find Within Distance")
            max_dist = st.number_input(
                "Max distance", 0.0, 1.0, 0.3, 0.05, key="vp_max"
            )
            if st.button("Search", key="vp_within"):
                results = tree.find_within(query, max_distance=max_dist)
                if results:
                    df_r = pd.DataFrame(results, columns=["String", "Distance", "Index"])
                    st.dataframe(df_r, use_container_width=True)
                else:
                    st.info("No results found.")

# --- N-gram index ---
with tab3:
    st.markdown("N-gram indexes provide fast approximate matching via shared n-gram overlap.")
    ngram_n = st.number_input("N-gram size", 1, 5, 2, key="ngram_n")

    if st.button("Build N-gram Index", type="primary", key="ngram_build"):
        index = reclink.NgramIndex.build(strings, n=ngram_n)
        st.session_state["ngram_index"] = index
        st.success(f"N-gram index built with {len(strings)} strings")

    if "ngram_index" in st.session_state:
        index = st.session_state["ngram_index"]

        st.markdown("---")
        query = st.text_input("Query string", value="smith", key="ngram_query")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Search by Threshold")
            threshold = st.number_input(
                "Min shared n-grams", 1, 20, 2, key="ngram_threshold"
            )
            if st.button("Search", key="ngram_search"):
                results = index.search(query, threshold=threshold)
                if results:
                    df_r = pd.DataFrame(results, columns=["Index", "String", "Shared"])
                    st.dataframe(df_r, use_container_width=True)
                else:
                    st.info("No results found.")

        with c2:
            st.subheader("Search Top-K")
            k = st.number_input("k", 1, 20, 3, key="ngram_k")
            if st.button("Search", key="ngram_topk"):
                results = index.search_top_k(query, k=k)
                if results:
                    df_r = pd.DataFrame(results, columns=["Index", "String", "Shared"])
                    st.dataframe(df_r, use_container_width=True)
                else:
                    st.info("No results found.")

# --- MinHash/LSH ---
with tab4:
    st.markdown(
        "MinHash/LSH indexes provide approximate nearest-neighbor search for large collections."
    )
    mh_col1, mh_col2 = st.columns(2)
    with mh_col1:
        num_hashes = st.number_input("Num hashes", 10, 500, 100, key="mh_hashes")
    with mh_col2:
        num_bands = st.number_input("Num bands", 5, 100, 20, key="mh_bands")

    if st.button("Build MinHash Index", type="primary", key="mh_build"):
        index = reclink.MinHashIndex.build(
            strings, num_hashes=num_hashes, num_bands=num_bands
        )
        st.session_state["mh_index"] = index
        st.success(f"MinHash index built with {len(strings)} strings")

    if "mh_index" in st.session_state:
        index = st.session_state["mh_index"]

        st.markdown("---")
        query = st.text_input("Query string", value="smith", key="mh_query")
        mh_threshold = st.slider("Threshold", 0.0, 1.0, 0.3, 0.05, key="mh_threshold")

        if st.button("Query", type="primary", key="mh_search"):
            results = index.query(query, threshold=mh_threshold)
            if results:
                df_r = pd.DataFrame(results, columns=["Index", "String", "Score"])
                st.dataframe(df_r, use_container_width=True)
            else:
                st.info("No results found.")
