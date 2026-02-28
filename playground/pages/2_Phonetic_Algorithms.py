"""Phonetic Algorithms — Encode strings and compare phonetically."""

import streamlit as st

st.set_page_config(page_title="Phonetic Algorithms — reclink", page_icon="🔊", layout="wide")
st.title("Phonetic Algorithms")
st.markdown("Encode strings using phonetic algorithms and compare how names sound.")

import reclink  # noqa: E402

ALGORITHMS = {
    "soundex": reclink.soundex,
    "metaphone": reclink.metaphone,
    "double_metaphone": reclink.double_metaphone,
    "nysiis": reclink.nysiis,
    "caverphone": reclink.caverphone,
    "cologne_phonetic": reclink.cologne_phonetic,
    "beider_morse": reclink.beider_morse,
}

tab1, tab2, tab3 = st.tabs(["Encode", "Compare Names", "Phonetic Hybrid"])

# --- Encode ---
with tab1:
    text = st.text_input("Enter a string", value="Smith", key="encode_input")
    algorithm = st.selectbox("Algorithm", list(ALGORITHMS.keys()))

    if st.button("Encode", type="primary", key="encode"):
        fn = ALGORITHMS[algorithm]
        try:
            result = fn(text)
            if isinstance(result, tuple):
                st.metric("Primary Code", result[0])
                st.metric("Alternate Code", result[1])
            else:
                st.metric("Phonetic Code", result)
        except Exception as e:
            st.error(f"Error: {e}")

    # Encode with all algorithms
    st.markdown("---")
    if st.button("Encode with All Algorithms", key="encode_all"):
        cols = st.columns(2)
        for i, (name, fn) in enumerate(ALGORITHMS.items()):
            with cols[i % 2]:
                try:
                    result = fn(text)
                    if isinstance(result, tuple):
                        st.metric(name, f"{result[0]} / {result[1]}")
                    else:
                        st.metric(name, result)
                except Exception as e:
                    st.metric(name, f"Error: {e}")

    # Language detection
    st.markdown("---")
    st.subheader("Language Detection")
    lang_input = st.text_input("Enter a name", value="Müller", key="lang_input")
    if st.button("Detect Language", key="detect_lang"):
        lang = reclink.detect_language(lang_input)
        st.metric("Detected Language", lang)

# --- Compare names ---
with tab2:
    st.markdown("Compare two names phonetically — do they sound the same?")
    col1, col2 = st.columns(2)
    with col1:
        name_a = st.text_input("Name A", value="Smith", key="compare_a")
    with col2:
        name_b = st.text_input("Name B", value="Smyth", key="compare_b")

    compare_algo = st.selectbox(
        "Algorithm", list(ALGORITHMS.keys()), key="compare_algo"
    )

    if st.button("Compare", type="primary", key="compare"):
        fn = ALGORITHMS[compare_algo]
        try:
            code_a = fn(name_a)
            code_b = fn(name_b)

            col1, col2 = st.columns(2)
            with col1:
                if isinstance(code_a, tuple):
                    st.metric(f"Code for '{name_a}'", f"{code_a[0]} / {code_a[1]}")
                else:
                    st.metric(f"Code for '{name_a}'", code_a)
            with col2:
                if isinstance(code_b, tuple):
                    st.metric(f"Code for '{name_b}'", f"{code_b[0]} / {code_b[1]}")
                else:
                    st.metric(f"Code for '{name_b}'", code_b)

            if isinstance(code_a, tuple):
                match = code_a[0] == code_b[0] or code_a[1] == code_b[1]
            else:
                match = code_a == code_b

            if match:
                st.success("These names sound the SAME phonetically!")
            else:
                st.warning("These names sound DIFFERENT phonetically.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Phonetic hybrid ---
with tab3:
    st.markdown(
        "Combine phonetic encoding with string similarity for a weighted hybrid score."
    )
    col1, col2 = st.columns(2)
    with col1:
        hybrid_a = st.text_input("String A", value="Jon Smith", key="hybrid_a")
    with col2:
        hybrid_b = st.text_input("String B", value="John Smyth", key="hybrid_b")

    hybrid_col1, hybrid_col2, hybrid_col3 = st.columns(3)
    with hybrid_col1:
        phonetic_algo = st.selectbox(
            "Phonetic algorithm",
            ["soundex", "metaphone", "double_metaphone", "nysiis", "caverphone"],
            key="hybrid_phonetic",
        )
    with hybrid_col2:
        string_metric = st.selectbox(
            "String metric",
            ["jaro_winkler", "jaro", "levenshtein_similarity"],
            key="hybrid_metric",
        )
    with hybrid_col3:
        phonetic_weight = st.slider(
            "Phonetic weight", 0.0, 1.0, 0.3, 0.05, key="hybrid_weight"
        )

    if st.button("Calculate Hybrid Score", type="primary", key="hybrid"):
        score = reclink.phonetic_hybrid(
            hybrid_a,
            hybrid_b,
            phonetic=phonetic_algo,
            metric=string_metric,
            phonetic_weight=phonetic_weight,
        )
        st.metric("Hybrid Score", f"{score:.4f}")
        st.progress(float(score))

        st.markdown(
            f"""
        **Breakdown:** `{phonetic_weight:.0%}` phonetic ({phonetic_algo})
        + `{1 - phonetic_weight:.0%}` string similarity ({string_metric})
        """
        )
