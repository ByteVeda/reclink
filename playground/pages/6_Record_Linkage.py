"""Record Linkage — Full pipeline demo with file upload."""

import io

import streamlit as st

st.set_page_config(page_title="Record Linkage — reclink", page_icon="🔗", layout="wide")
st.title("Record Linkage Pipeline")
st.markdown(
    "Configure and run a full deduplication or record linkage pipeline on your data."
)

import pandas as pd  # noqa: E402
from reclink.pipeline import ReclinkPipeline  # noqa: E402

tab1, tab2 = st.tabs(["Deduplication", "Record Linkage"])

# --- Deduplication ---
with tab1:
    st.subheader("Upload Dataset")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="dedup_csv")

    if uploaded is None:
        st.info("Upload a CSV file, or try the demo dataset below.")
        if st.button("Load Demo Dataset", key="demo_dedup"):
            demo_data = pd.DataFrame(
                {
                    "id": ["1", "2", "3", "4", "5"],
                    "first_name": ["Jon", "John", "Jane", "Janet", "Bob"],
                    "last_name": ["Smith", "Smyth", "Doe", "Doe", "Jones"],
                    "city": ["New York", "New York", "Boston", "Boston", "Chicago"],
                }
            )
            st.session_state["dedup_df"] = demo_data

    if uploaded is not None:
        st.session_state["dedup_df"] = pd.read_csv(uploaded)

    if "dedup_df" not in st.session_state:
        st.stop()

    df = st.session_state["dedup_df"]
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"**{len(df)} rows, {len(df.columns)} columns**")

    # Pipeline configuration
    st.subheader("Configure Pipeline")

    # ID column
    id_col = st.selectbox("ID column", df.columns, key="dedup_id")

    # Fields to compare
    compare_fields = st.multiselect(
        "Fields to compare",
        [c for c in df.columns if c != id_col],
        key="dedup_fields",
    )

    if not compare_fields:
        st.warning("Select at least one field to compare.")
        st.stop()

    # Preprocessing
    st.markdown("**Preprocessing**")
    preprocess_config = {}
    for field in compare_fields:
        ops = st.multiselect(
            f"Preprocessing for '{field}'",
            ["fold_case", "normalize_whitespace", "strip_punctuation", "strip_diacritics"],
            default=["fold_case", "normalize_whitespace"],
            key=f"preprocess_{field}",
        )
        if ops:
            preprocess_config[field] = ops

    # Blocking
    st.markdown("**Blocking Strategy**")
    blocking_field = st.selectbox(
        "Blocking field",
        compare_fields,
        key="dedup_block_field",
    )
    blocking_strategy = st.selectbox(
        "Strategy",
        ["none", "exact", "phonetic", "sorted_neighborhood", "qgram"],
        key="dedup_block_strategy",
    )

    # Comparison metrics
    st.markdown("**Comparison Metrics**")
    compare_config = {}
    for field in compare_fields:
        metric = st.selectbox(
            f"Metric for '{field}'",
            ["jaro_winkler", "jaro", "levenshtein_similarity", "cosine", "token_sort_ratio"],
            key=f"metric_{field}",
        )
        compare_config[field] = metric

    # Classifier
    st.markdown("**Classification**")
    classifier_type = st.selectbox(
        "Classifier",
        ["threshold", "weighted", "fellegi_sunter_auto"],
        key="dedup_classifier",
    )
    threshold = st.slider("Threshold", 0.0, 1.0, 0.85, 0.05, key="dedup_threshold")

    # Clustering
    use_clustering = st.checkbox("Enable clustering", value=True, key="dedup_cluster")
    if use_clustering:
        cluster_method = st.selectbox(
            "Clustering method",
            ["connected_components", "hierarchical"],
            key="dedup_cluster_method",
        )

    # Run
    if st.button("Run Deduplication", type="primary", key="run_dedup"):
        with st.spinner("Building and running pipeline..."):
            try:
                builder = ReclinkPipeline.builder()

                # Preprocessing
                for field, ops in preprocess_config.items():
                    builder = builder.preprocess(field, ops)

                # Blocking
                if blocking_strategy == "exact":
                    builder = builder.block_exact(blocking_field)
                elif blocking_strategy == "phonetic":
                    builder = builder.block_phonetic(blocking_field)
                elif blocking_strategy == "sorted_neighborhood":
                    builder = builder.block_sorted_neighborhood(blocking_field)
                elif blocking_strategy == "qgram":
                    builder = builder.block_qgram(blocking_field)

                # Comparators
                for field, metric in compare_config.items():
                    builder = builder.compare_string(field, metric=metric)

                # Classifier
                if classifier_type == "threshold":
                    builder = builder.classify_threshold(threshold)
                elif classifier_type == "weighted":
                    weights = [1.0 / len(compare_fields)] * len(compare_fields)
                    builder = builder.classify_weighted(weights, threshold)
                elif classifier_type == "fellegi_sunter_auto":
                    builder = builder.classify_fellegi_sunter_auto()

                pipeline = builder.build()

                if use_clustering:
                    results = pipeline.dedup_cluster(df, id_column=id_col)
                    st.subheader("Clusters")
                    if results:
                        for i, cluster in enumerate(results):
                            st.markdown(f"**Cluster {i + 1}:** {cluster}")
                    else:
                        st.info("No duplicate clusters found.")
                else:
                    results = pipeline.dedup(df, id_column=id_col)
                    st.subheader("Matches")
                    if hasattr(results, "__len__") and len(results) > 0:
                        st.dataframe(results, use_container_width=True)

                        # Export
                        if isinstance(results, pd.DataFrame):
                            csv_data = results.to_csv(index=False)
                        else:
                            csv_data = pd.DataFrame(
                                [
                                    {
                                        "left_id": r.left_id,
                                        "right_id": r.right_id,
                                        "score": r.score,
                                    }
                                    for r in results
                                ]
                            ).to_csv(index=False)

                        st.download_button(
                            "Download Results CSV",
                            csv_data,
                            file_name="dedup_results.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No matches found.")

            except Exception as e:
                st.error(f"Pipeline error: {e}")

# --- Record Linkage ---
with tab2:
    st.subheader("Upload Two Datasets")

    c1, c2 = st.columns(2)
    with c1:
        left_file = st.file_uploader("Left CSV", type=["csv"], key="link_left")
    with c2:
        right_file = st.file_uploader("Right CSV", type=["csv"], key="link_right")

    if left_file is None or right_file is None:
        st.info("Upload two CSV files to link records between them.")
        if st.button("Load Demo Datasets", key="demo_link"):
            left_demo = pd.DataFrame(
                {
                    "id": ["L1", "L2", "L3"],
                    "name": ["Jon Smith", "Jane Doe", "Bob Jones"],
                    "city": ["NYC", "Boston", "Chicago"],
                }
            )
            right_demo = pd.DataFrame(
                {
                    "id": ["R1", "R2", "R3"],
                    "name": ["John Smyth", "Janet Doe", "Robert Jones"],
                    "city": ["New York", "Boston", "Chicago"],
                }
            )
            st.session_state["link_left"] = left_demo
            st.session_state["link_right"] = right_demo

    if left_file is not None:
        st.session_state["link_left"] = pd.read_csv(left_file)
    if right_file is not None:
        st.session_state["link_right"] = pd.read_csv(right_file)

    if "link_left" not in st.session_state or "link_right" not in st.session_state:
        st.stop()

    df_left = st.session_state["link_left"]
    df_right = st.session_state["link_right"]

    lc, rc = st.columns(2)
    with lc:
        st.markdown("**Left dataset**")
        st.dataframe(df_left.head(), use_container_width=True)
    with rc:
        st.markdown("**Right dataset**")
        st.dataframe(df_right.head(), use_container_width=True)

    # Configuration
    st.subheader("Configure Linkage")

    link_id_col = st.text_input("ID column name (must exist in both)", value="id", key="link_id")

    common_cols = list(set(df_left.columns) & set(df_right.columns) - {link_id_col})
    link_fields = st.multiselect("Fields to compare", common_cols, key="link_fields")

    if not link_fields:
        st.warning("Select at least one common field to compare.")
        st.stop()

    link_threshold = st.slider("Threshold", 0.0, 1.0, 0.8, 0.05, key="link_threshold")

    if st.button("Run Linkage", type="primary", key="run_link"):
        with st.spinner("Running linkage pipeline..."):
            try:
                builder = ReclinkPipeline.builder()

                for field in link_fields:
                    builder = builder.preprocess(field, ["fold_case", "normalize_whitespace"])
                    builder = builder.compare_string(field, metric="jaro_winkler")

                builder = builder.classify_threshold(link_threshold)
                pipeline = builder.build()

                results = pipeline.link(df_left, df_right, id_column=link_id_col)

                st.subheader("Linked Records")
                if hasattr(results, "__len__") and len(results) > 0:
                    st.dataframe(results, use_container_width=True)

                    if isinstance(results, pd.DataFrame):
                        csv_data = results.to_csv(index=False)
                    else:
                        csv_data = pd.DataFrame(
                            [
                                {
                                    "left_id": r.left_id,
                                    "right_id": r.right_id,
                                    "score": r.score,
                                }
                                for r in results
                            ]
                        ).to_csv(index=False)

                    st.download_button(
                        "Download Results CSV",
                        csv_data,
                        file_name="linkage_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No links found above threshold.")

            except Exception as e:
                st.error(f"Pipeline error: {e}")
