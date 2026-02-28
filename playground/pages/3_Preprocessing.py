"""Preprocessing — Build and test text preprocessing pipelines."""

import streamlit as st

st.set_page_config(page_title="Preprocessing — reclink", page_icon="🧹", layout="wide")
st.title("Preprocessing")
st.markdown("Build text preprocessing pipelines and see transformations step by step.")

import pandas as pd  # noqa: E402
import reclink  # noqa: E402

OPERATIONS = {
    "fold_case": reclink.fold_case,
    "normalize_whitespace": reclink.normalize_whitespace,
    "strip_punctuation": reclink.strip_punctuation,
    "strip_diacritics": reclink.strip_diacritics,
    "remove_stop_words": reclink.remove_stop_words,
    "expand_abbreviations": reclink.expand_abbreviations,
    "standardize_name": reclink.standardize_name,
    "clean_name": reclink.clean_name,
    "clean_address": reclink.clean_address,
    "clean_company": reclink.clean_company,
}

tab1, tab2, tab3 = st.tabs(["Pipeline Builder", "Domain Preprocessors", "Batch Processing"])

# --- Pipeline builder ---
with tab1:
    text = st.text_input("Input text", value="  DR. José  García-López  Jr.  ")

    selected_ops = st.multiselect(
        "Select operations (applied in order)",
        list(OPERATIONS.keys()),
        default=["fold_case", "normalize_whitespace", "strip_punctuation"],
    )

    if st.button("Apply Pipeline", type="primary", key="pipeline"):
        st.subheader("Step-by-step transformation")
        current = text
        st.code(f"Original: {repr(current)}")

        for op_name in selected_ops:
            fn = OPERATIONS[op_name]
            try:
                current = fn(current)
                st.code(f"After {op_name}: {repr(current)}")
            except Exception as e:
                st.error(f"Error in {op_name}: {e}")
                break

        st.success(f"Final result: {repr(current)}")

    # Unicode normalization
    st.markdown("---")
    st.subheader("Unicode Normalization")
    unicode_input = st.text_input("Input", value="café", key="unicode_input")
    norm_form = st.selectbox("Form", ["nfkc", "nfc", "nfkd", "nfd"])
    if st.button("Normalize", key="unicode_norm"):
        result = reclink.normalize_unicode(unicode_input, form=norm_form)
        st.code(f"Result: {repr(result)}")

    # Transliteration
    st.markdown("---")
    st.subheader("Transliteration")
    translit_input = st.text_input("Input", value="Москва", key="translit_input")
    translit_type = st.selectbox(
        "Script",
        ["cyrillic", "greek", "arabic", "hebrew", "devanagari", "hangul"],
    )
    if st.button("Transliterate", key="translit"):
        fn = getattr(reclink, f"transliterate_{translit_type}")
        result = fn(translit_input)
        st.metric("Result", result)

# --- Domain preprocessors ---
with tab2:
    st.markdown("Specialized preprocessors for common data types.")

    domain_col1, domain_col2 = st.columns(2)

    with domain_col1:
        st.subheader("Name Cleaning")
        name_input = st.text_input("Name", value="  DR. John  Smith Jr.  ", key="name")
        if st.button("Clean Name", key="clean_name"):
            st.metric("Result", reclink.clean_name(name_input))

        st.subheader("Address Cleaning")
        addr_input = st.text_input(
            "Address", value='123 N. Main St., Apt #4', key="addr"
        )
        if st.button("Clean Address", key="clean_addr"):
            st.metric("Result", reclink.clean_address(addr_input))

    with domain_col2:
        st.subheader("Company Cleaning")
        company_input = st.text_input(
            "Company", value="The Acme Corp., Inc.", key="company"
        )
        if st.button("Clean Company", key="clean_company"):
            st.metric("Result", reclink.clean_company(company_input))

        st.subheader("Email Normalization")
        email_input = st.text_input(
            "Email", value="John.Doe+tag@Gmail.COM", key="email"
        )
        if st.button("Normalize Email", key="norm_email"):
            st.metric("Result", reclink.normalize_email(email_input))

    st.markdown("---")
    st.subheader("URL Normalization")
    url_input = st.text_input(
        "URL", value="HTTP://WWW.Example.COM/path/", key="url"
    )
    if st.button("Normalize URL", key="norm_url"):
        st.metric("Result", reclink.normalize_url(url_input))

    # Synonym expansion
    st.markdown("---")
    st.subheader("Synonym Expansion")
    syn_input = st.text_input("Text", value="123 main st", key="syn_input")
    syn_table = st.text_area(
        "Synonym table (one per line, format: abbrev=expansion)",
        value="st=street\nave=avenue\ndr=drive",
        key="syn_table",
    )
    if st.button("Expand Synonyms", key="syn_expand"):
        table = {}
        for line in syn_table.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                table[k.strip()] = v.strip()
        result = reclink.synonym_expand(syn_input, table)
        st.metric("Result", result)

# --- Batch processing ---
with tab3:
    st.markdown("Apply preprocessing to multiple strings or a CSV column.")

    batch_mode = st.radio("Input mode", ["Text list", "Upload CSV"], horizontal=True)

    if batch_mode == "Text list":
        batch_input = st.text_area(
            "Enter strings (one per line)",
            value="  John Smith \nJANE  DOE\n  Dr. Bob Jones Jr.  ",
        )
        batch_ops = st.multiselect(
            "Operations",
            ["fold_case", "normalize_whitespace", "strip_punctuation", "strip_diacritics"],
            default=["fold_case", "normalize_whitespace"],
            key="batch_ops",
        )
        if st.button("Process Batch", type="primary", key="batch"):
            strings = [s for s in batch_input.split("\n") if s.strip()]
            result = reclink.preprocess_batch(strings, operations=batch_ops)
            df = pd.DataFrame({"Original": strings, "Processed": result})
            st.dataframe(df, use_container_width=True)
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head(), use_container_width=True)

            column = st.selectbox("Column to preprocess", df.columns)
            csv_ops = st.multiselect(
                "Operations",
                ["fold_case", "normalize_whitespace", "strip_punctuation", "strip_diacritics"],
                default=["fold_case", "normalize_whitespace"],
                key="csv_ops",
            )
            if st.button("Process Column", type="primary", key="csv_batch"):
                strings = df[column].astype(str).tolist()
                result = reclink.preprocess_batch(strings, operations=csv_ops)
                df[f"{column}_processed"] = result
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False)
                st.download_button(
                    "Download Result CSV",
                    csv_data,
                    file_name="preprocessed.csv",
                    mime="text/csv",
                )
