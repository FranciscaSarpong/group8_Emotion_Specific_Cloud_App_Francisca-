import streamlit as st              # Web app framework
import pandas as pd                 # Data handling
import numpy as np                  # Numerics
import matplotlib.pyplot as plt     # Plots (basic EDA)
import seaborn as sns
import re
import math
import os
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,
    confusion_matrix,roc_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
import io
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


# (Good practice after changing environments / versions)
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass


st.set_page_config(page_title="Emotion Word Clouds (Word2Vec)", layout="wide")


# One-line help per page
PAGE_HELP = {
    "Home": "Overview of the project, goals, objectives, and navigation.",
    "Data Load": "Upload CSV/Parquet, preview the data, inspect schema and distributions, and download the loaded dataset or a sample.",
    "Preprocessing": "Clean text and map Score→Emotion. Output must be `st.session_state['df_clean']` with `clean_text` & `Emotion` columns.",
    "Word2Vec Embeddings": "Train Word2Vec; create review-level embeddings; save X/y/label_map in session.",
    "Modeling": "Train & compare Random Forest vs XGBoost; cross‑validation and imbalance handling.",
    "Model Evaluation & Results": "Macro metrics, confusion matrices, ROC curves; identify best model.",
    "Emotion Word Clouds": "Generate emotion‑specific word clouds from TF‑IDF/embedding weighting.",
    "Predictions & Downloads": "Single/batch predictions + download models/metrics/cleaned CSV.",
    "Conclusion & Insights": "Summarize findings and show the three word clouds."
}


# Helpers (NO st.* calls except caching)

def safe_coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce useful dtypes and parse epoch Time → datetime if present."""
    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce").astype("Int64")
    if "Time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        df["Time"] = pd.to_datetime(df["Time"], unit="s", errors="coerce")
    return df

@st.cache_data(show_spinner=True)
def load_reviews(file_obj, usecols=None, nrows=None) -> pd.DataFrame:
    """
    Uploader‑only loader. Reads CSV/Parquet from st.file_uploader file-like object.
    - .parquet → pd.read_parquet (respects columns)
    - otherwise → pd.read_csv (respects usecols & nrows)
    """
    if file_obj is None:
        return pd.DataFrame()
    name = file_obj.name.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(file_obj, columns=usecols if usecols else None)
    else:
        df = pd.read_csv(file_obj, usecols=usecols if usecols else None, nrows=nrows)
    return safe_coerce_types(df).reset_index(drop=True)

def score_to_emotion(score: int) -> str:
    """1–2 → Negative, 3 → Neutral, 4–5 → Positive."""
    try:
        s = int(score)
    except Exception:
        return "Unknown"
    if s <= 2: return "Negative"
    if s == 3:  return "Neutral"
    return "Positive"

# STEP 1: HOMEPAGE

def page_home():
    st.title(" Emotion‑Specific Word Cloud from Amazon Reviews (Word2Vec)")
    st.caption(PAGE_HELP["Home"])

    # Home page logo
    st.image(
        "C:/Users/Francesca Manu/PycharmProjects/Text_Group_Project/Group8_logo.png",
        width=300,  # smaller width
        caption="Emotion-Specific Word Cloud from Amazon Reviews"

    )

    st.markdown("""
        ### **Goal**
        Build an automated system that analyzes Amazon Fine Food Reviews, detects **emotions** (Positive / Neutral / Negative),
        and visualizes the vocabulary per emotion via **word clouds**, using **Word2Vec** + ML classifiers.

        ### **Objectives**
        1. Clean & normalize reviews; map stars → emotions.  
        2. Train **Word2Vec**; create review embeddings.  
        3. Train & compare **Random Forest** and **XGBoost** with macro metrics (Precision, Recall, F1, ROC‑AUC).  
        4. Generate **emotion‑specific word clouds** for interpretability.  
        5. Provide an interactive **Streamlit** app for exploration & predictions.

        ### **Navigation**
        Use the sidebar to move step‑by‑step: Data Load → Preprocessing → Embeddings → Modeling → Evaluation → Word Clouds → Conclusion.
        """)

    # Team
    st.markdown("---")
    st.markdown("### Project Team")
    team = [
        ("George Owell", "22256146", "Evaluation, Cross-validation"),
        ("Francisca Manu Sarpong", "22255796", "Feature Engineering, Model Training"),
        ("Franklina Oppong", "11410681", "Evaluation, Cross-validation"),
        ("Ewuraben Biney", "22252464", "Prediction UI, Testing"),
        ("Esther Edem Tulasi Carr", "22253335", "Documentation, Deployment"),
    ]
    c1, c2, c3 = st.columns([4, 2, 6])
    with c1:
        st.markdown("**Name**")
        for n, _, _ in team: st.markdown(n)
    with c2:
        st.markdown("**Student ID**")
        for _, sid, _ in team: st.markdown(sid)
    with c3:
        st.markdown("**Role**")
        for _, _, role in team: st.markdown(role)

    st.info(" Start with **Data Load** in the sidebar.")


# Page: Data Load (+ downloads)

def page_data_load():
    st.title("Data Load")
    st.caption(PAGE_HELP["Data Load"])

    st.markdown(
        "Please Upload `Reviewsample.csv` or `Reviews.parquet`. The app will parse dates, cache the data, and show quick EDA.")

    #  File uploader
    up = st.file_uploader("Upload file", type=["csv", "parquet"])

    #  Column selection
    default_cols = ["HelpfulnessNumerator", "HelpfulnessDenominator",
                    "Score", "Time", "Summary", "Text"]
    st.markdown("#### Columns to load (applies to CSV and Parquet)")
    selected_cols = st.multiselect(
        "Choose columns (fewer → faster)",
        default_cols, default=["Score", "Time", "Summary", "Text"]
    )

    #  CSV-only partial read
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        first_n = st.checkbox("Read only first N (CSV)", value=True)
    with c2:
        nrows = st.number_input("N rows", min_value=5_000, value=50_000, step=5_000)
    with c3:
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    if st.button("Load Dataset"):
        if up is None:
            st.error("Please upload a CSV or Parquet file.")
            return

        try:
            df = load_reviews(
                file_obj=up,
                usecols=selected_cols if selected_cols else None,
                nrows=int(nrows) if (first_n and up.name.lower().endswith(".csv")) else None
            )
            if df.empty:
                st.error("Loaded an empty DataFrame. Check the file and selected columns.")
                return

            st.session_state["df"] = df
            st.success(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

            #  Preview & schema
            st.subheader("Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Column Types & Missingness")
            info = pd.DataFrame({
                "Column": df.columns,
                "Dtype": df.dtypes.astype(str),
                "Non‑Null": df.notnull().sum(),
                "Nulls": df.isnull().sum(),
                "Null %": (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(info, use_container_width=True)

            st.subheader("Duplicates")
            st.write(f"Exact duplicate rows: **{df.duplicated().sum():,}**")

            #  Target awareness
            if "Score" in df.columns:
                st.subheader("Score & Emotion Distribution")
                st.bar_chart(df["Score"].value_counts(dropna=False).sort_index())
                emo = df["Score"].dropna().astype(int).map(score_to_emotion)
                st.bar_chart(emo.value_counts())

            if "Time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Time"]):
                tmin, tmax = df["Time"].min(), df["Time"].max()
                if pd.notnull(tmin) and pd.notnull(tmax):
                    st.write(f"Time coverage: **{tmin.date()} → {tmax.date()}**")

            st.info("Dataset cached. Next step: **Preprocessing** page.")
        except Exception as e:
            st.error(f"Unexpected error while loading data: {e}")


def page_preprocess():

    st.title(" Preprocess & Emotion Mapping")
    st.markdown("""
        ##### This page cleans and prepares the Amazon Fine Food Reviews dataset for analysis.
        * It allows you to choose text-cleaning options (lowercasing, removing HTML, punctuation, digits, stopwords, etc.),limit rows for speed, and optionally remove duplicates.
        
        * The cleaned text is stored in a new column (clean_text),and each review is assigned an emotion label (Positive, Neutral, or Negative) based on its score.
        
        * It shows emotion distribution, previews the cleaned data, saves it in session state for later steps,and lets you download the processed dataset as a CSV..
        """
                )

    # 0) Get raw dataset from Step 2
    df = st.session_state.get("df", None)
    if df is None:
        st.error("No dataset found. Please load data in *Data Load* first.")
        st.stop()

    # 1) Cleaning options (global best-practice defaults)
    st.markdown("### Cleaning Options")
    c1, c2, c3 = st.columns(3)
    with c1:
        do_lower   = st.checkbox("Lowercase", value=True)
        rm_html    = st.checkbox("Remove HTML-like tags", value=True)
        rm_punct   = st.checkbox("Remove punctuation", value=True)
    with c2:
        rm_digits  = st.checkbox("Remove digits", value=True)
        collapse_ws= st.checkbox("Collapse extra spaces", value=True)
        drop_empty = st.checkbox("Drop empty/NA cleaned rows", value=True)
    with c3:
        use_nltk_stop = st.checkbox("Use NLTK stopwords", value=True)
        do_lemmatize  = st.checkbox("Lemmatize (WordNet)", value=True)
        do_stem       = st.checkbox("Stem (Porter)", value=False)

    st.caption("Tip: Prefer *lemmatization* for academic clarity. Avoid using stem & lemma together.")

    # Limit rows to keep the app responsive on huge files
    st.markdown("### Rows to Process")
    col_a, col_b = st.columns(2)
    with col_a:
        limit_rows = st.checkbox("Process only first N rows", value=True)
    with col_b:
        n_limit = st.number_input("N rows (if limited)", min_value=5_000, value=50_000, step=5_000)

    # Optional deduplication
    st.markdown("### Data Hygiene")
    rm_dups = st.checkbox("Remove exact duplicate rows before cleaning", value=False)

    # 2) Lazy NLTK resource fetchers
    @st.cache_resource(show_spinner=False)
    def _ensure_nltk_assets():
        try:
            import nltk
            nltk.data.find("corpora/stopwords")
        except Exception:
            try:
                nltk.download("stopwords", quiet=True)
            except Exception:
                pass
        try:
            import nltk
            nltk.data.find("corpora/wordnet")
            nltk.data.find("corpora/omw-1.4")
        except Exception:
            try:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            except Exception:
                pass

    # Local fallback stopwords (if NLTK not available)
    FALLBACK_STOPWORDS = set("""
    a an the and or but if while with without within into onto from to for of on in out by up down over under again further
    is are was were be been being do does did doing have has had having this that these those it its i me my we our you your
    he him his she her they them their what which who whom where when why how all any both each few more most other some such
    no nor not only own same so than too very can will just should now
    """.split())

    def get_stopwords():
        if use_nltk_stop:
            _ensure_nltk_assets()
            try:
                from nltk.corpus import stopwords as sw
                return set(sw.words("english"))
            except Exception:
                st.warning("NLTK stopwords unavailable; falling back to built-in list.")
        return FALLBACK_STOPWORDS

    #  3) Core text preprocessor
    def preprocess_text(text: str, stop_set: set) -> str:
        """
        Normalize one review into a clean string:
        - lowercase → remove HTML → remove punctuation → remove digits → collapse spaces
        - tokenize on whitespace → remove stopwords & 1-char tokens
        - optional lemmatization/stemming
        """
        s = str(text)

        if do_lower:
            s = s.lower()
        if rm_html:
            s = re.sub(r"<.*?>", " ", s)
        if rm_punct:
            s = re.sub(r"[^\w\s]", " ", s)
        if rm_digits:
            s = re.sub(r"\d+", " ", s)
        if collapse_ws:
            s = re.sub(r"\s+", " ", s).strip()

        tokens = s.split()
        if not tokens:
            return ""

        tokens = [t for t in tokens if len(t) > 1 and t not in stop_set]

        if do_lemmatize or do_stem:
            _ensure_nltk_assets()
            if do_lemmatize:
                try:
                    from nltk.stem import WordNetLemmatizer
                    lem = WordNetLemmatizer()
                    tokens = [lem.lemmatize(t) for t in tokens]
                except Exception:
                    st.warning("WordNet lemmatizer unavailable; skipping lemmatization.")
            if do_stem:
                try:
                    from nltk.stem import PorterStemmer
                    stemmer = PorterStemmer()
                    tokens = [stemmer.stem(t) for t in tokens]
                except Exception:
                    st.warning("Porter stemmer unavailable; skipping stemming.")

        return " ".join(tokens)

    #  4) Run preprocessing
    if st.button("Run Preprocessing"):
        work_df = df.copy()

        # Optional: restrict rows for speed
        if limit_rows:
            work_df = work_df.head(int(n_limit)).copy()

        # Optional: remove exact duplicate rows (entire row)
        if rm_dups:
            before = len(work_df)
            work_df = work_df.drop_duplicates().reset_index(drop=True)
            st.info(f"Removed {before - len(work_df)} duplicate rows.")

        # Ensure required columns exist
        for col in ["Text", "Score"]:
            if col not in work_df.columns:
                st.error(f"Required column '{col}' is missing from the dataset.")
                st.stop()

        # Ensure canonical dtypes
        work_df["Text"] = work_df["Text"].astype(str)
        work_df["Score"] = pd.to_numeric(work_df["Score"], errors="coerce").astype("Int64")

        # Build stopword set once
        stop_set = get_stopwords()

        # Apply normalization (progress bar)
        st.markdown("#### Cleaning Text…")
        with st.spinner("Applying text normalization to reviews…"):
            work_df["clean_text"] = work_df["Text"].apply(lambda x: preprocess_text(x, stop_set))

        # Optionally drop empties
        if drop_empty:
            before = len(work_df)
            work_df = work_df[work_df["clean_text"].str.len() > 0].reset_index(drop=True)
            removed = before - len(work_df)
            if removed > 0:
                st.info(f"Dropped {removed} rows with empty cleaned text.")

        #  5) Map Score → Emotion
        # Reuse the helper from Step 2 if present; else define a local fallback
        try:
            mapper = score_to_emotion  # defined earlier in your file
        except NameError:
            def mapper(score):
                try:
                    s = int(score)
                except Exception:
                    return "Unknown"
                if s in (1, 2): return "Negative"
                if s == 3:       return "Neutral"
                return "Positive"  # 4–5

        work_df["Emotion"] = work_df["Score"].apply(mapper)

        #  6) Quick checks & outputs
        st.subheader("Emotion Distribution")
        emo_counts = work_df["Emotion"].value_counts()
        st.bar_chart(emo_counts)
        st.write(emo_counts)

        st.subheader("Preview of Cleaned Text")
        cols_to_show = [c for c in ["Score", "Emotion", "Summary", "Text", "clean_text"] if c in work_df.columns]
        st.dataframe(work_df[cols_to_show].head(12), use_container_width=True)

        # 7) Save to session for next steps
        st.session_state["df_clean"] = work_df
        st.session_state["cleaned_df"] = work_df  # alias for any page expecting this key

        # Optional: allow download for reproducibility
        st.markdown("### Download Cleaned Subset")
        csv_bytes = work_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download cleaned CSV",
            data=csv_bytes,
            file_name="amazon_reviews_cleaned.csv",
            mime="text/csv"
        )

        st.success("Preprocessing complete ✅ — cleaned data saved to session as 'df_clean'.")



# STEP 4: POST‑CLEANING DIAGNOSTICS
def page_diagnostics():
    """
    Post‑Cleaning Diagnostics
    -------------------------
    Purpose: verify that cleaning & label mapping produced a healthy corpus.

    """

    st.title(" Post‑Cleaning Diagnostics")
    st.markdown("""
    This page **validates data quality** before embeddings & modeling:
    - Corpus shape, token length distribution  
    - Vocabulary size & lexical richness  
    - Overall top tokens (frequency)  
    - Emotion distribution  
    - Short‑review & duplicate guardrails  
    - *(Optional)* overall TF‑IDF signal check
    """)

    #  0) Load cleaned data
    df = st.session_state.get("df_clean")
    if df is None or "clean_text" not in df.columns or "Emotion" not in df.columns:
        st.error("No cleaned dataset found. Please run **Preprocess & Emotion Mapping** first.")
        st.stop()
    df = df.copy()

    #  1) Corpus shape & token lengths
    st.subheader("Corpus Shape & Hygiene")
    rows = len(df)
    nn_clean = df["clean_text"].notna().sum()
    nn_emot  = df["Emotion"].notna().sum()
    st.write(
        f"Rows: **{rows:,}** | non‑null `clean_text`: **{nn_clean:,}** | "
        f"non‑null `Emotion`: **{nn_emot:,}**"
    )

    lengths = df["clean_text"].fillna("").str.split().apply(len)
    df["_len_tokens"] = lengths  # temp col for guardrails

    m1, m2, m3 = st.columns(3)
    m1.metric("Median tokens/review", int(np.median(lengths)))
    m2.metric("Mean tokens/review", f"{np.mean(lengths):.2f}")
    m3.metric("Max tokens/review", int(np.max(lengths)))

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(lengths, bins=80, kde=False, ax=ax)
    ax.set_title("Distribution of Review Lengths (tokens)")
    ax.set_xlabel("Tokens per review"); ax.set_ylabel("Count")
    st.pyplot(fig)

    # 2) Lexical richness
    st.subheader("Vocabulary & Lexical Richness")
    tokens_series = df["clean_text"].fillna("").str.split()
    vocab_counter = Counter(t for row in tokens_series for t in row)
    vocab_size = len(vocab_counter)
    total_tokens = int(lengths.sum())
    ttr = (vocab_size / total_tokens) if total_tokens > 0 else 0.0
    herdan_c = (math.log(vocab_size + 1) / math.log(total_tokens + 1)) if total_tokens > 0 else 0.0
    hapax_prop = sum(1 for _, c in vocab_counter.items() if c == 1) / max(vocab_size, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vocabulary size", f"{vocab_size:,}")
    c2.metric("Total tokens", f"{total_tokens:,}")
    c3.metric("Type–Token Ratio (TTR)", f"{ttr:.4f}")
    c4.metric("Herdan’s C", f"{herdan_c:.4f}")
    st.caption(f"Hapax proportion (tokens with frequency = 1): **{hapax_prop:.3f}**")

    #  3) Emotion distribution
    st.subheader("Emotion Distribution")
    emo_counts = df["Emotion"].value_counts()
    st.bar_chart(emo_counts)
    st.dataframe(emo_counts.rename("Count").to_frame(), use_container_width=True)

    #  4) Top tokens overall (frequency)
    st.subheader("Top Tokens — Overall (frequency)")
    top_overall_k = st.slider("Show top N tokens", 10, 50, 20, 5, key="diag_top_overall_k")
    top_overall = pd.DataFrame(vocab_counter.most_common(top_overall_k),
                               columns=["Token", "Frequency"])
    st.dataframe(top_overall, use_container_width=True)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.barplot(y="Token", x="Frequency", data=top_overall, ax=ax2)
    ax2.set_title(f"Top {top_overall_k} Tokens (Overall)")
    st.pyplot(fig2)

    #  5) Optional: overall TF‑IDF signal check
    with st.expander("Optional: Overall TF‑IDF Signal Check"):
        max_feats = st.slider("Max features", 1000, 12000, 4000, 500,
                              help="Limit vocabulary to keep this fast.")
        tfidf = TfidfVectorizer(max_features=max_feats, token_pattern=r"(?u)\b\w+\b")
        X = tfidf.fit_transform(df["clean_text"].fillna(""))
        vocab = np.array(tfidf.get_feature_names_out())
        k = st.slider("Top N tokens to display", 10, 40, 20, 5, key="diag_tfidf_topk")
        mean_scores = np.asarray(X.mean(axis=0)).ravel()
        order = np.argsort(mean_scores)[::-1][:k]
        tfidf_overall = pd.DataFrame({"Token": vocab[order], "Mean TF‑IDF": mean_scores[order]})
        st.dataframe(tfidf_overall, use_container_width=True)

    #  6) Guardrail: short reviews
    st.subheader("Short Reviews (≤ threshold tokens)")
    thr = st.slider("Threshold (tokens)", 1, 10, 2, 1, key="diag_short_thr")
    n_short = int((df["_len_tokens"] <= thr).sum())
    st.write(f"Found **{n_short:,}** reviews with ≤ {thr} tokens.")
    with st.expander("Preview some short reviews"):
        st.dataframe(
            df.loc[df["_len_tokens"] <= thr, ["Score", "Emotion", "clean_text"]].head(20),
            use_container_width=True
        )

    if st.checkbox("Remove these short reviews and update session", value=False, key="diag_drop_short"):
        keep = df["_len_tokens"] > thr
        removed = int((~keep).sum())
        df2 = df.loc[keep].drop(columns=["_len_tokens"]).reset_index(drop=True)
        st.session_state["df_clean"] = df2
        st.success(f"Removed {removed:,} rows. Session updated. Re‑run diagnostics if you wish.")
    else:
        df.drop(columns=["_len_tokens"], inplace=True, errors="ignore")

    #  7) Optional: near‑duplicate detection
    with st.expander("Optional: Near‑duplicates by content hash"):
        hashes = df["clean_text"].fillna("").map(hash)
        dup_mask = hashes.duplicated(keep="first")
        n_dup = int(dup_mask.sum())
        st.write(f"Potential exact duplicates (same cleaned text): **{n_dup:,}**")
        if n_dup > 0:
            st.dataframe(df.loc[dup_mask, ["Emotion", "clean_text"]].head(20),
                         use_container_width=True)
            if st.checkbox("Drop exact duplicates by cleaned text", value=False, key="diag_drop_dups"):
                df2 = df.loc[~dup_mask].reset_index(drop=True)
                st.session_state["df_clean"] = df2
                st.success(f"Dropped {n_dup:,} duplicate rows. Session updated.")

# STEP 5: EMBEDDINGS (Word2Vec)

def page_word2vec():

    """
    Train a Word2Vec model on the cleaned corpus and compute per‑review embeddings.
    Outputs saved to session_state:
      - w2v_model  : trained gensim model
      - X_emb      : ndarray [n_docs, vector_size] (mean of token vectors per review)
      - y_labels   : ndarray [n_docs] (encoded labels from Emotion)
      - label_map  : dict {Emotion -> int}
    """

    st.title(" Word2Vec — Train & Vectorize")
    st.markdown("""
    ## Word2Vec — Turning Reviews into Numbers
    **Purpose:**  
    This page trains a **Word2Vec embedding model** on the cleaned text corpus and converts each review into a **dense vector representation** for machine learning.
    """)

    #  0) Fetch cleaned data
    df_clean = st.session_state.get("df_clean", None)
    if df_clean is None or "clean_text" not in df_clean.columns or "Emotion" not in df_clean.columns:
        st.error("No cleaned dataset found. Please run **Preprocess & Emotion Mapping** first.")
        st.stop()

    # Keep only the columns we need; create tokens column
    work_df = df_clean[["clean_text", "Emotion", "Score"]].copy()
    work_df["tokens"] = work_df["clean_text"].astype(str).apply(str.split)

    # Quick QA note on short docs
    n_short = int((work_df["tokens"].str.len() <= 2).sum())
    if n_short:
        st.info(f"{n_short} very short reviews (≤ 2 tokens) remain; "
                "they may yield zero vectors if none of their tokens are in the Word2Vec vocabulary.")

    # 1) Hyper‑parameters UI
    st.markdown("### Hyper‑parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        vector_size = st.number_input("vector_size (embedding dim)", 50, 600, 200, 25)
        window = st.number_input("window (context size)", 2, 15, 5, 1)
        min_count = st.number_input("min_count (discard rare words <)", 1, 20, 5, 1)
    with c2:
        sg_choice = st.selectbox("Architecture", ["Skip‑gram (sg=1)", "CBOW (sg=0)"], index=0)
        epochs = st.number_input("epochs", 3, 50, 10, 1)
        negative = st.number_input("negative sampling (0=off)", 0, 20, 10, 1)
    with c3:
        train_on_split = st.checkbox("Train on TRAIN split only (avoid leakage)", value=True)
        test_size = st.slider("Test size (if split)", 0.1, 0.4, 0.2, 0.05)
        seed = st.number_input("random_state / seed", 0, 9999, 42, 1)

    sg = 1 if "sg=1" in sg_choice else 0
    st.caption(
        "Guidance: With vocab ≈ 30–40k tokens, **min_count=5** and **vector_size=200** are strong defaults. "
        "Skip‑gram (sg=1) often performs better on rarer words."
    )

    # 2) Build training corpus
    if train_on_split:
        # Stratified split to avoid leakage; train Word2Vec only on train tokens
        trn, _ = train_test_split(
            work_df, test_size=float(test_size), random_state=int(seed), stratify=work_df["Emotion"]
        )
        corpus_tokens = trn["tokens"].tolist()
        st.info(f"Training on TRAIN only: **{len(trn):,}** documents.")
    else:
        corpus_tokens = work_df["tokens"].tolist()
        st.info(f"Training on ALL cleaned reviews: **{len(work_df):,}** documents.")

    #  3) Train Word2Vec (no double‑training)
    @st.cache_resource(show_spinner=True)
    def train_w2v(corpus, vec, win, minc, sg, epochs, negative, seed):
        """
        Build vocab once, then train once.
        Cached by hyper‑params & corpus tokens to avoid re‑training unnecessarily.
        """
        model = Word2Vec(
            vector_size=int(vec),
            window=int(win),
            min_count=int(minc),
            sg=int(sg),  # 0=CBOW, 1=Skip‑gram
            negative=int(negative),
            workers=min(4, os.cpu_count() or 1),  # safe for local/Cloud
            seed=int(seed),
        )
        model.build_vocab(corpus)
        model.train(corpus, total_examples=len(corpus), epochs=int(epochs))
        return model

    with st.spinner("Training Word2Vec…"):
        w2v_model = train_w2v(
            corpus=corpus_tokens,
            vec=vector_size,
            win=window,
            minc=min_count,
            sg=sg,
            epochs=epochs,
            negative=negative,
            seed=seed
        )

    vocab_size = len(w2v_model.wv.key_to_index)
    st.success(f"✅ Trained Word2Vec. Vocabulary size after min_count: **{vocab_size:,}**")

    # 4) Nearest‑neighbor sanity check
    st.markdown("#### Nearest words (sanity check)")
    default_terms = [w for w in ["good", "bad", "love", "terrible", "taste", "coffee"] if w in w2v_model.wv]
    # Limit options list length for UI responsiveness
    options_list = list(w2v_model.wv.key_to_index.keys())
    probe_terms = st.multiselect(
        "Pick words to inspect",
        options=options_list[: min(8000, len(options_list))],
        default=default_terms
    )
    for term in probe_terms:
        try:
            sims = w2v_model.wv.most_similar(term, topn=8)
            st.write("**" + term + "** → " + ", ".join([f"{w} ({s:.2f})" for w, s in sims]))
        except KeyError:
            # Shouldn’t happen since we gate by in-vocab terms, but safe-guard anyway
            pass

    #  5) Build per‑review embeddings
    st.markdown("### Build per‑review embeddings")

    dim = int(vector_size)

    def doc_vector(tokens, model, dim):
        """
        Average the word vectors for tokens that exist in the model vocab.
        Returns a zero vector if no token is in-vocab (all OOV).
        """
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if not vecs:
            return np.zeros(dim, dtype="float32")
        return np.mean(vecs, axis=0).astype("float32")

    # Compute document vectors for all reviews (including test if trained on split)
    X_emb = np.vstack([doc_vector(toks, w2v_model, dim) for toks in work_df["tokens"]])

    # QA: how many rows are all-zero (all OOV tokens)?
    zero_rows = int((X_emb == 0).all(axis=1).sum())
    st.caption(f"Zero‑vector reviews (all tokens OOV): **{zero_rows}** "
               f"({zero_rows / len(work_df):.2%}). Consider lowering `min_count` if this is high.")

    # Encode labels into integers for modeling
    label_map = {lbl: idx for idx, lbl in enumerate(sorted(work_df["Emotion"].unique()))}
    y_labels = work_df["Emotion"].map(label_map).values

    st.write(f"Embeddings shape: **{X_emb.shape}**  (rows: reviews, cols: {dim})")
    st.write(f"Label map: {label_map}")

    # 6) Persist to session for Modeling & Clouds
    st.session_state["w2v_model"] = w2v_model
    st.session_state["X_emb"] = X_emb
    st.session_state["y_labels"] = y_labels
    st.session_state["label_map"] = label_map
    st.success("Per‑review embeddings are ready ✅. Go to **Modeling** to train RF & XGBoost.")

    # 7) Optional: export a small embeddings sample
    st.markdown("#### Download a small sample (for your report)")
    max_rows = min(5000, len(work_df))
    sample_n = st.slider("Rows to include", 500, max_rows, min(1000, max_rows), 500)
    if st.button("Create sample CSV"):
        sample = pd.DataFrame(X_emb[:sample_n], columns=[f"v{i + 1}" for i in range(dim)])
        sample["Emotion"] = work_df["Emotion"].iloc[:sample_n].values
        sample["Score"] = work_df["Score"].iloc[:sample_n].values
        st.download_button(
            "Download embeddings sample",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name="embeddings_sample.csv",
            mime="text/csv"
        )


def page_modeling():
    """
    Train and compare Random Forest & XGBoost on Word2Vec review embeddings.
    - Handles imbalance via Class Weights or SMOTE (train folds only).
    - Reports macro-averaged metrics (precision/recall/F1/ROC-AUC) via CV.
    - Trains on a hold-out split for confusion matrices and ROC curves.
    - Shows compact feature-importance plots (top embedding dimensions).
    Saves a 'results' bundle in st.session_state for the Evaluation page.
    """

    # Optional deps
    try:
        from xgboost import XGBClassifier
        XGB_OK = True
    except Exception:
        XGB_OK = False
    try:
        from imblearn.over_sampling import SMOTE
        SMOTE_OK = True
    except Exception:
        SMOTE_OK = False

    st.title(" Modeling — Random Forest vs XGBoost")
    st.markdown("""
    Purpose: Train and compare two classifiers — **Random Forest** and **XGBoost** — on the Word2Vec review embeddings to predict the **Emotion** class (Negative / Neutral / Positive).

    """)

    # 0) Fetch features/labels
    X = st.session_state.get("X_emb", None)
    y = st.session_state.get("y_labels", None)
    label_map = st.session_state.get("label_map", None)

    if X is None or y is None or label_map is None:
        st.error("Embeddings not found. Please run **Word2Vec — Train & Vectorize** first.")
        st.stop()

    classes_sorted = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    labels_indices = np.array([label_map[c] for c in classes_sorted])
    n_classes = len(classes_sorted)
    st.write(f"Features: **{X.shape}**, Classes: **{classes_sorted}**")

    #  1) Controls
    with st.expander("🎛 Training Controls", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
            n_splits = st.number_input("CV folds (StratifiedKFold)", 3, 10, 5, 1)
        with c2:
            imb = st.selectbox(
                "Imbalance handling",
                ["Class weights", "SMOTE", "None"],
                help="SMOTE is applied on TRAIN folds only; Class weights uses balanced weighting."
            )
        with c3:
            random_state = st.number_input("random_state", 0, 9999, 42, 1)

    # RF params
    with st.expander(" Random Forest Hyper‑parameters"):
        rf_n_estimators = st.number_input("n_estimators", 100, 2000, 400, 50)
        rf_max_depth = st.number_input("max_depth (0 = None)", 0, 200, 0, 1)
        rf_min_samples_leaf = st.number_input("min_samples_leaf", 1, 20, 1, 1)

    # XGB params
    with st.expander("⚡ XGBoost Hyper‑parameters"):
        if not XGB_OK:
            st.warning("XGBoost not installed — results will include only Random Forest.")
        xgb_n_estimators = st.number_input("n_estimators", 100, 2000, 600, 50)
        xgb_learning_rate = st.number_input("learning_rate", 0.01, 0.5, 0.1, 0.01)
        xgb_max_depth = st.number_input("max_depth", 2, 20, 6, 1)
        xgb_subsample = st.slider("subsample", 0.5, 1.0, 0.8, 0.05)
        xgb_colsample = st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)
        xgb_reg_lambda = st.number_input("lambda (L2)", 0.0, 10.0, 1.0, 0.1)

    # Ensure numeric types from widgets
    test_size = float(test_size)
    n_splits = int(n_splits)
    random_state = int(random_state)
    rf_n_estimators = int(rf_n_estimators)
    rf_max_depth = int(rf_max_depth)
    rf_min_samples_leaf = int(rf_min_samples_leaf)
    xgb_n_estimators = int(xgb_n_estimators)
    xgb_learning_rate = float(xgb_learning_rate)
    xgb_max_depth = int(xgb_max_depth)
    xgb_subsample = float(xgb_subsample)
    xgb_colsample = float(xgb_colsample)
    xgb_reg_lambda = float(xgb_reg_lambda)

    # Helper: per-sample weights to emulate class_weight for XGB
    def sample_weights(y_vec):
        classes, counts = np.unique(y_vec, return_counts=True)
        total = len(y_vec)
        w = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        return np.array([w[yi] for yi in y_vec])

    #  2) Train/Test split
    if st.button(" Train & Compare"):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        st.caption("Train class counts: " + str(dict(Counter(y_tr))))
        st.caption("Test  class counts: " + str(dict(Counter(y_te))))

        # Optionally balance training set
        sw_tr = None
        rf_class_weight = None

        if imb == "Class weights":
            rf_class_weight = "balanced"
            sw_tr = sample_weights(y_tr)
        elif imb == "SMOTE":
            if not SMOTE_OK:
                st.warning("SMOTE unavailable (install `imbalanced-learn`). Continuing without SMOTE.")
            else:
                sm = SMOTE(random_state=random_state)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                st.info(f"SMOTE applied on train: X_train → {X_tr.shape}")

        #  3) Cross‑validation (macro metrics)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        def cv_eval(clf, uses_sample_weight=False):
            precs, recs, f1s, aucs = [], [], [], []
            for tr_idx, va_idx in skf.split(X, y):
                Xtr, Xva = X[tr_idx], X[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]

                # Re-apply SMOTE in each fold (train only)
                if imb == "SMOTE" and SMOTE_OK:
                    sm = SMOTE(random_state=random_state)
                    Xtr, ytr = sm.fit_resample(Xtr, ytr)

                sw = sample_weights(ytr) if (imb == "Class weights" and uses_sample_weight) else None
                clf.fit(Xtr, ytr, sample_weight=sw) if uses_sample_weight else clf.fit(Xtr, ytr)

                proba = clf.predict_proba(Xva)
                # Safe clipping to avoid numeric edge cases in ROC
                proba = np.clip(proba, 1e-8, 1 - 1e-8)

                yva_bin = label_binarize(yva, classes=np.unique(y))
                try:
                    auc = roc_auc_score(yva_bin, proba, average="macro", multi_class="ovr")
                except Exception:
                    auc = np.nan
                preds = clf.predict(Xva)

                precs.append(precision_score(yva, preds, average="macro", zero_division=0))
                recs.append(recall_score(yva, preds, average="macro", zero_division=0))
                f1s.append(f1_score(yva, preds, average="macro", zero_division=0))
                aucs.append(auc)
            return np.nanmean(precs), np.nanmean(recs), np.nanmean(f1s), np.nanmean(aucs)

        #  Train & collect results
        metrics_rows = []                 # for results["metrics"]
        conf_matrices = {}               # for results["conf_matrices"]
        roc_curves = {}                  # for results["roc_curves"]
        feature_importance = {}          # for results["feature_importance"]

        #  Random Forest
        rf_kwargs = dict(
            n_estimators=rf_n_estimators,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        if rf_max_depth > 0:
            rf_kwargs["max_depth"] = rf_max_depth
        if rf_class_weight:
            rf_kwargs["class_weight"] = rf_class_weight

        rf = RandomForestClassifier(**rf_kwargs)
        rf_p, rf_r, rf_f1, rf_auc = cv_eval(rf, uses_sample_weight=False)

        # Fit on train for confusion matrix / ROC / accuracy
        rf.fit(X_tr, y_tr)
        rf_preds = rf.predict(X_te)
        rf_prob = np.clip(rf.predict_proba(X_te), 1e-8, 1 - 1e-8)
        rf_acc = accuracy_score(y_te, rf_preds)

        y_te_bin = label_binarize(y_te, classes=labels_indices)
        try:
            rf_auc_macro = roc_auc_score(y_te_bin, rf_prob, average="macro", multi_class="ovr")
            fpr_rf, tpr_rf, _ = roc_curve(y_te_bin.ravel(), rf_prob.ravel())
        except Exception:
            rf_auc_macro = np.nan
            fpr_rf, tpr_rf = np.array([0, 1]), np.array([0, 1])

        metrics_rows.append({
            "Model": "Random Forest",
            "Precision": rf_p, "Recall": rf_r, "F1": rf_f1,
            "ROC-AUC": rf_auc, "Accuracy": rf_acc
        })
        conf_matrices["Random Forest"] = confusion_matrix(y_te, rf_preds, labels=labels_indices)
        roc_curves["Random Forest"] = (fpr_rf, tpr_rf, rf_auc_macro)

        # Feature importance (embedding dims)
        if hasattr(rf, "feature_importances_"):
            feature_importance["Random Forest"] = {
                "features": [f"dim_{i}" for i in range(X.shape[1])],
                "importance": rf.feature_importances_.tolist()
            }

        #  XGBoost
        if XGB_OK:
            xgb = XGBClassifier(
                n_estimators=xgb_n_estimators,
                learning_rate=xgb_learning_rate,
                max_depth=xgb_max_depth,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample,
                reg_lambda=xgb_reg_lambda,
                objective="multi:softprob",
                num_class=n_classes,
                tree_method="hist",
                n_jobs=-1,
                random_state=random_state,
            )
            xgb_p, xgb_r, xgb_f1, xgb_auc = cv_eval(xgb, uses_sample_weight=(imb == "Class weights"))
            sw = sample_weights(y_tr) if (imb == "Class weights") else None
            xgb.fit(X_tr, y_tr, sample_weight=sw)
            xgb_preds = xgb.predict(X_te)
            xgb_prob = np.clip(xgb.predict_proba(X_te), 1e-8, 1 - 1e-8)
            xgb_acc = accuracy_score(y_te, xgb_preds)

            try:
                xgb_auc_macro = roc_auc_score(y_te_bin, xgb_prob, average="macro", multi_class="ovr")
                fpr_xgb, tpr_xgb, _ = roc_curve(y_te_bin.ravel(), xgb_prob.ravel())
            except Exception:
                xgb_auc_macro = np.nan
                fpr_xgb, tpr_xgb = np.array([0, 1]), np.array([0, 1])

            metrics_rows.append({
                "Model": "XGBoost",
                "Precision": xgb_p, "Recall": xgb_r, "F1": xgb_f1,
                "ROC-AUC": xgb_auc, "Accuracy": xgb_acc
            })
            conf_matrices["XGBoost"] = confusion_matrix(y_te, xgb_preds, labels=labels_indices)
            roc_curves["XGBoost"] = (fpr_xgb, tpr_xgb, xgb_auc_macro)

            if hasattr(xgb, "feature_importances_"):
                feature_importance["XGBoost"] = {
                    "features": [f"dim_{i}" for i in range(X.shape[1])],
                    "importance": xgb.feature_importances_.tolist()
                }
        else:
            st.info("XGBoost not installed; skipping XGB metrics.")

        #  4) Side‑by‑side results
        st.subheader(" Side‑by‑Side Results (macro)")
        res_df = pd.DataFrame(metrics_rows).set_index("Model")
        st.dataframe(res_df.style.format("{:.4f}"), use_container_width=True)

        #  5) Confusion matrices
        st.subheader(" Confusion Matrices")
        fig, axes = plt.subplots(1, 2 if "XGBoost" in conf_matrices else 1, figsize=(12, 4))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        cm_rf = conf_matrices["Random Forest"]
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=classes_sorted, yticklabels=classes_sorted)
        axes[0].set_title("Random Forest"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

        if "XGBoost" in conf_matrices:
            cm_xgb = conf_matrices["XGBoost"]
            sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", ax=axes[1],
                        xticklabels=classes_sorted, yticklabels=classes_sorted)
            axes[1].set_title("XGBoost"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

        st.pyplot(fig)

        #  6) ROC curves
        st.subheader(" ROC Curves (macro AUC reported; curve is micro-style)")
        try:
            fig2, ax2 = plt.subplots()
            fpr_rf, tpr_rf, auc_rf = roc_curves["Random Forest"]
            ax2.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")
            if "XGBoost" in roc_curves:
                fpr_xgb, tpr_xgb, auc_xgb = roc_curves["XGBoost"]
                ax2.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.3f})")
            ax2.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
            ax2.legend()
            st.pyplot(fig2)
        except Exception:
            st.info("Could not render ROC curves (probabilities/labels issue).")

        # 7) Save everything for Evaluation page
        st.session_state["rf_model"] = rf
        if XGB_OK:
            st.session_state["xgb_model"] = xgb

        st.session_state["results"] = {
            "metrics": metrics_rows,
            "conf_matrices": conf_matrices,
            "labels": classes_sorted,          # used as tick labels on Evaluation page
            "roc_curves": roc_curves,
            "feature_importance": feature_importance
        }

        #  8) Feature importance plots (compact)
        st.subheader(" Feature Importance (Top Embedding Dimensions)")

        def plot_top_importances(model_name: str, importances: list, top_k: int = 15):
            """Render a horizontal bar chart of top_k most important embedding dims."""
            if not importances:
                st.info(f"No importances available for {model_name}.")
                return
            imp = np.asarray(importances)
            dims = np.array([f"dim_{i}" for i in range(len(importances))])

            order = np.argsort(imp)[::-1][:top_k]
            top_df = pd.DataFrame({
                "Feature": dims[order],
                "Importance": imp[order]
            })

            # Table
            st.markdown(f"**{model_name} — Top {top_k} dimensions**")
            st.dataframe(top_df, use_container_width=True)

            # Plot
            fig_imp, ax_imp = plt.subplots(figsize=(6, 4.5))
            sns.barplot(data=top_df, y="Feature", x="Importance", ax=ax_imp)
            ax_imp.set_title(f"{model_name}: Top {top_k} Feature Importances")
            ax_imp.set_xlabel("Importance")
            ax_imp.set_ylabel("Embedding Dimension")
            st.pyplot(fig_imp)

        fi = feature_importance  # already built above
        cols = st.columns(2 if "XGBoost" in fi else 1)
        with cols[0]:
            if "Random Forest" in fi:
                plot_top_importances("Random Forest", fi["Random Forest"]["importance"], top_k=15)
            else:
                st.info("Random Forest importances not available.")

        if "XGBoost" in fi and len(cols) > 1:
            with cols[1]:
                plot_top_importances("XGBoost", fi["XGBoost"]["importance"], top_k=15)

        st.success("Training complete ✅ — metrics, plots, and importances ready.")



def page_ModelEvaluation_and_Results():
    """
    Display evaluation metrics, confusion matrices, ROC curves, and feature importances
    using the results bundle saved by the Modeling page in st.session_state["results"].
    """

    st.title(" Model Evaluation & Results")
    st.markdown("""
    ###
    This page presents the **evaluation results** of the machine learning models trained on the *Modeling* page.  
    It summarizes how well each model performed on a **held-out test set** and helps in understanding their predictive strengths and weaknesses.
    """)

    #  1) Retrieve saved results
    results = st.session_state.get("results", None)
    if results is None:
        st.error("⚠ No saved results found. Please run the **Modeling** page first.")
        st.stop()

    metrics_rows = results.get("metrics", [])
    conf_matrices = results.get("conf_matrices", {})
    labels = results.get("labels", None)  # display order of classes
    roc_curves = results.get("roc_curves", {})
    feature_importance = results.get("feature_importance", {})

    if not metrics_rows or labels is None:
        st.error("⚠ Results object is incomplete. Re‑run the **Modeling** page.")
        st.stop()

    # 2) Metrics table
    st.subheader(" Evaluation Metrics (macro)")
    metrics_df = pd.DataFrame(metrics_rows).set_index("Model")

    # Format only numeric cols safely
    fmt_map = {col: "{:.4f}" for col in metrics_df.select_dtypes(include=["float", "int"]).columns}
    st.dataframe(metrics_df.style.format(fmt_map), use_container_width=True)

    # Download metrics CSV
    st.download_button(
        " Download metrics (CSV)",
        data=metrics_df.to_csv().encode("utf-8"),
        file_name="model_metrics.csv",
        mime="text/csv"
    )

    #  3) Confusion matrices
    st.subheader(" Confusion Matrices (hold‑out test)")
    model_names = list(conf_matrices.keys())
    if not model_names:
        st.info("No confusion matrices found in results.")
        cm_rf = None
        cm_xgb = None
    else:
        n_panels = len(model_names)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
        if n_panels == 1:
            axes = np.array([axes])

        cm_rf = conf_matrices.get("Random Forest")
        cm_xgb = conf_matrices.get("XGBoost")

        idx = 0
        for mname in model_names:
            ax = axes[idx]
            cm = np.asarray(conf_matrices[mname])
            sns.heatmap(cm, annot=True, fmt="d",
                        cmap="Blues" if "Forest" in mname else "Greens",
                        ax=ax, xticklabels=labels, yticklabels=labels)
            ax.set_title(mname)
            ax.set_xlabel("Predicted");
            ax.set_ylabel("True")
            idx += 1
        st.pyplot(fig)

    #  4) ROC curves
    st.subheader(" ROC Curves (macro AUC reported; curve is micro‑style)")
    if not roc_curves:
        st.info("No ROC data available. Re‑run the Modeling page.")
    else:
        fig2, ax2 = plt.subplots()
        for mname, (fpr, tpr, auc_macro) in roc_curves.items():
            fpr = np.asarray(fpr);
            tpr = np.asarray(tpr)
            label = f"{mname} (AUC={auc_macro:.3f})" if np.isfinite(auc_macro) else f"{mname} (AUC=n/a)"
            ax2.plot(fpr, tpr, label=label)
        ax2.plot([0, 1], [0, 1], "k--", label="Random guess")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        st.pyplot(fig2)

    # 5) Feature importances (top embedding dimensions)
    st.subheader(" Feature Importance (Top Embedding Dimensions)")

    def plot_top_importances(model_name: str, fi_dict: dict, top_k: int = 15):
        if not fi_dict or "importance" not in fi_dict:
            st.info(f"No importances available for {model_name}.")
            return
        imp = np.asarray(fi_dict["importance"])
        dims = np.array(fi_dict.get("features", [f"dim_{i}" for i in range(len(imp))]))
        order = np.argsort(imp)[::-1][:top_k]
        top_df = pd.DataFrame({"Feature": dims[order], "Importance": imp[order]})

        st.markdown(f"**{model_name} — Top {top_k} dimensions**")
        st.dataframe(top_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=top_df, y="Feature", x="Importance", ax=ax)
        ax.set_title(f"{model_name}: Top {top_k} Feature Importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Embedding Dimension")
        st.pyplot(fig)

    cols = st.columns(2 if "XGBoost" in feature_importance else 1)
    with cols[0]:
        if "Random Forest" in feature_importance:
            plot_top_importances("Random Forest", feature_importance["Random Forest"], top_k=15)
        else:
            st.info("Random Forest importances not available.")
    if "XGBoost" in feature_importance and len(cols) > 1:
        with cols[1]:
            plot_top_importances("XGBoost", feature_importance["XGBoost"], top_k=15)

    # 6) Interpretation & Recommendations (with highlights)
    # Small CSS for colored badges
    st.markdown("""
           <style>
           .badge {display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.85rem; font-weight:600; margin:0 4px;}
           .bg-green {background:#e6f4ea; color:#127a2a; border:1px solid #b7e2c0;}
           .bg-blue  {background:#e8f0fe; color:#1b4dd7; border:1px solid #c6d2ff;}
           .bg-red   {background:#fdecea; color:#b3261e; border:1px solid #f6c7c3;}
           .bg-amber {background:#fff7e6; color:#8a5a00; border:1px solid #ffe0a3;}
           .chip     {display:inline-block; padding:2px 8px; border-radius:16px; font-size:0.80rem; font-weight:600; margin-left:6px;}
           .chip-neutral {background:#eef2f7; color:#334155; border:1px solid #cbd5e1;}
           </style>
       """, unsafe_allow_html=True)

    st.markdown("##  Interpretation & Recommendations")

    def _class_metrics_from_cm(cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
        tp = np.diag(cm).astype(float)
        support = cm.sum(axis=1).astype(float)  # true counts
        predsum = cm.sum(axis=0).astype(float)  # predicted counts
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        precision = np.divide(tp, predsum, out=np.zeros_like(tp), where=predsum > 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp), where=(precision + recall) > 0)
        return pd.DataFrame({
            "Class": class_names,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support.astype(int)
        })

    # Per‑class tables (hold‑out)
    per_class_rf = _class_metrics_from_cm(cm_rf, labels) if cm_rf is not None else None
    if per_class_rf is not None:
        st.markdown("**Random Forest — class‑wise metrics (hold‑out):**")
        st.dataframe(per_class_rf.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                     use_container_width=True)

    per_class_xgb = _class_metrics_from_cm(cm_xgb, labels) if cm_xgb is not None else None
    if per_class_xgb is not None:
        st.markdown("**XGBoost — class‑wise metrics (hold‑out):**")
        st.dataframe(per_class_xgb.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                     use_container_width=True)

    # Pull macro rows from metrics table
    rf_row = metrics_df.loc["Random Forest"].to_dict() if "Random Forest" in metrics_df.index else {}
    xgb_row = metrics_df.loc["XGBoost"].to_dict() if "XGBoost" in metrics_df.index else None

    # Winner by (macro F1, then ROC‑AUC)
    def _pick_score(row):
        return (row.get("F1", 0), row.get("ROC-AUC", 0))

    winner_name = "Random Forest"
    if xgb_row and _pick_score(xgb_row) > _pick_score(rf_row):
        winner_name = "XGBoost"

    # Hardest class across available models (lowest recall)
    hardest_class = None
    hardest_recall = 1.0
    for df_ in [d for d in [per_class_rf, per_class_xgb] if d is not None]:
        idx = df_["Recall"].idxmin()
        if df_["Recall"].iloc[idx] < hardest_recall:
            hardest_recall = float(df_["Recall"].iloc[idx])
            hardest_class = df_["Class"].iloc[idx]

    # Header chips
    st.markdown(
        f'**Overall winner:** <span class="badge bg-green">{winner_name}</span> '
        f'&nbsp;&nbsp;|&nbsp;&nbsp; '
        f'**Hardest class (lowest recall):** '
        f'<span class="badge bg-red">{hardest_class if hardest_class is not None else "N/A"}</span> '
        f'<span class="chip chip-neutral">≈ {hardest_recall:.3f}</span>',
        unsafe_allow_html=True
    )

    # Compact metric badges per model
    def _model_badges(name: str, row: dict, color_class: str):
        p = row.get("Precision", np.nan);
        r = row.get("Recall", np.nan)
        f1 = row.get("F1", np.nan);
        auc = row.get("ROC-AUC", np.nan)
        st.markdown(
            f'**{name}** '
            f'<span class="badge {color_class}">Precision {p:.3f}</span>'
            f'<span class="badge {color_class}">Recall {r:.3f}</span>'
            f'<span class="badge {color_class}">F1 {f1:.3f}</span>'
            f'<span class="badge {color_class}">ROC‑AUC {auc:.3f}</span>',
            unsafe_allow_html=True
        )

    if rf_row:
        _model_badges("Random Forest", rf_row, "bg-blue")
    if xgb_row:
        _model_badges("XGBoost", xgb_row, "bg-amber")

    st.markdown("### What to Improve Next")
    st.markdown(
        """
        - **Boost recall** on the hardest class (often *Neutral*): try **Class Weights**, **SMOTE/ADASYN**, or **threshold tuning**.
        - **Richer features**: add **bigrams** (e.g., “not good”), or combine Word2Vec with **TF‑IDF** / **SIF‑weighted** embeddings.
        - **Hyperparameter tuning**:  
          • RF — `n_estimators`, `max_depth`, `min_samples_leaf`  
          • XGB — `learning_rate`, `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`, `reg_lambda`
        - **Probability calibration** (Platt / isotonic) if you’ll use decision thresholds or cost‑sensitive alerts.
        """
    )

    st.success(
        "✅ Evaluation complete — metrics, confusion matrices, ROC curves, importances, and interpretation displayed.")




# STEP 7: EMOTION‑SPECIFIC WORD CLOUDS (ultimate)

def page_wordclouds():
    """
    Generate emotion-specific word clouds for Amazon reviews with robust filtering.
    Methods:
      • Contrastive log‑odds (recommended) – most discriminative vs other emotions, VADER gating
      • TF‑IDF per class (lexical)
      • Word2Vec centroid (semantic)
      • Word2Vec centroid (TF‑IDF‑weighted)

    Visual-only filters (do not affect modeling):
      • Token sanity (alphabetic 3–15 chars)
      • Global document frequency floor (drop ultra-rare/garbled tokens)
      • Per-emotion frequency floor (min count within that emotion)
      • VADER polarity gating for Pos/Neg/Neu
      • Brand/product stop-list
    """

    st.title(" Emotion‑Specific Word Clouds")
    st.markdown(""" 
    This page generates visual word clouds and ranked token tables for each emotion category (Negative, Neutral, Positive), allowing you to explore the most distinctive terms used in reviews for each sentiment.
    """)

    #  guards
    df_clean = st.session_state.get("df_clean", None)
    if df_clean is None or "clean_text" not in df_clean.columns or "Emotion" not in df_clean.columns:
        st.error("No cleaned dataset found. Run **Preprocess & Labels** first.")
        st.stop()

    EMOTIONS = [e for e in ["Negative", "Neutral", "Positive"] if e in df_clean["Emotion"].unique()]
    if not EMOTIONS:
        st.error("No emotion labels available.")
        st.stop()

    EMO2CMAP = {"Negative": "Reds", "Neutral": "Blues", "Positive": "Greens"}

    # controls
    method = st.radio(
        "Weighting method",
        [
            "Contrastive log‑odds (recommended)",
            "TF‑IDF per class (lexical)",
            "Word2Vec centroid (semantic)",
            "Word2Vec centroid (TF‑IDF‑weighted)",
        ],
        index=0,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        top_n = st.slider("Top‑N table / CSV", 20, 100, 50, 5)
    with c2:
        cloud_words_cap = st.slider("Words used in each cloud", 60, 300, 180, 20)
    with c3:
        background = st.selectbox("Background", ["white", "black"], index=0)

    limit_to_emotion_vocab = st.checkbox("Use only tokens appearing in that emotion’s reviews", value=True)
    min_freq_per_emotion = st.number_input("Min frequency in emotion", 1, 100, 10, 1)

    global_min_df = st.number_input(
        "Global document frequency (min #reviews containing token)",
        1, 1000, 20, 1,
        help="Higher → fewer rare/garbled tokens (especially helps Neutral).",
    )

    st.markdown("#### (Optional) Hide brand/product words (visual‑only)")
    default_stoplist = "amazon, starbucks, folgers, keurig, nespresso, kitkat, trader, joes, walmart, costco"
    custom_stop = st.text_input("Comma‑separated words to exclude", value=default_stoplist)
    HIDE = set(w.strip().lower() for w in custom_stop.split(",") if w.strip())

    with st.expander("Polarity gate (VADER) for Positive / Negative / Neutral",
                     expanded=method.startswith("Contrastive")):
        use_vader = st.checkbox("Gate by VADER polarity", value=True)
        pos_thresh = st.slider("Positive threshold (>=)", 0.1, 2.0, 0.5, 0.1)
        neg_thresh = st.slider("Negative threshold (<=)", -2.0, -0.1, -0.5, 0.1)
        neu_band = st.slider("Neutral band (|valence| ≤)", 0.1, 1.0, 0.2, 0.1)

    # domain stop‑list toggle (optional but helpful)
    use_domain_stop = st.checkbox("Apply domain stop‑list (like, product, good, etc.)", value=True)
    DOMAIN_STOP = {
        "like", "taste", "product", "one", "would", "good", "great", "get", "make", "really", "time", "much", "also",
        "food", "coffee",
        "tea", "amazon", "buy", "use", "used", "got", "well", "bit", "little", "thing", "things", "even"
    }

    # - token sanity & global DF
    TOKEN_RE = re.compile(r"^[a-z]+$")

    def token_ok(w: str) -> bool:
        return 3 <= len(w) <= 15 and TOKEN_RE.match(w) is not None and (
            w not in DOMAIN_STOP if use_domain_stop else True)

    def get_emotion_tokens(emo: str) -> list[str]:
        docs = df_clean.loc[df_clean["Emotion"] == emo, "clean_text"].astype(str)
        return [t for doc in docs for t in doc.split()]

    @st.cache_data(show_spinner=False)
    def build_global_docfreq(dc: pd.DataFrame) -> dict:
        dfreq = Counter()
        for text in dc["clean_text"].astype(str):
            toks = set(t for t in text.split() if token_ok(t))
            dfreq.update(toks)
        return dict(dfreq)

    GLOBAL_DF = build_global_docfreq(df_clean)

    #  VADER polarity sets
    @st.cache_resource(show_spinner=False)
    def vader_sets(pos_thr: float, neg_thr: float, neu_abs: float):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            lex = sia.lexicon  # token -> valence (-4..+4)
            POS = {w for w, v in lex.items() if v >= pos_thr}
            NEG = {w for w, v in lex.items() if v <= neg_thr}
            NEU = {w for w, v in lex.items() if -neu_abs <= v <= neu_abs}
            return POS, NEG, NEU
        except Exception as e:
            st.warning(f"VADER not available ({e}); polarity gating disabled.")
            return set(), set(), None

    POS_SET, NEG_SET, NEU_SET = (set(), set(), None)
    if use_vader:
        POS_SET, NEG_SET, NEU_SET = vader_sets(pos_thresh, neg_thresh, neu_band)

    #  per‑emotion filtered counts
    emo_counts: dict[str, Counter] = {}
    for emo in EMOTIONS:
        toks = [
            t for t in get_emotion_tokens(emo)
            if token_ok(t) and GLOBAL_DF.get(t, 0) >= global_min_df
        ]
        emo_counts[emo] = Counter(toks)

    #  renderer helpers
    def render_cloud(freqs: dict[str, float], title: str, cmap: str):
        freqs = {w: v for w, v in freqs.items() if w.lower() not in HIDE and v > 0}
        if not freqs:
            st.info(f"No terms to display for **{title}** after filtering.")
            return

        wc = WordCloud(width=1000, height=500, background_color=background,
                       collocations=False, colormap=cmap)
        wc = wc.generate_from_frequencies(freqs)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        st.download_button(
            label=f"Download '{title}' PNG",
            data=buf.getvalue(),
            file_name=f"{title.replace(' ', '_').lower()}.png",
            mime="image/png",
        )
        plt.close(fig)

        top_items = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        top_df = pd.DataFrame(top_items, columns=["Word", "Weight"])
        st.dataframe(top_df, use_container_width=True)
        st.download_button(
            label=f"Download '{title}' Top‑{top_n} CSV",
            data=top_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{title.replace(' ', '_').lower()}_top_{top_n}.csv",
            mime="text/csv",
        )

    #  compute weights per method
    weights_per_emotion: dict[str, dict[str, float]] = {}

    # A) Contrastive log‑odds
    if method.startswith("Contrastive"):
        st.info("Contrastive log‑odds with +1 smoothing (most discriminative vs other emotions). VADER gates applied.")
        global_counts = Counter()
        for emo in EMOTIONS:
            global_counts.update(emo_counts[emo])

        for emo in EMOTIONS:
            in_counts = emo_counts[emo]
            out_counts = global_counts.copy()
            for w, c in in_counts.items():
                out_counts[w] -= c

            in_total = sum(in_counts.values())
            out_total = sum(out_counts.values())

            allowed = {w for w, c in in_counts.items()} if limit_to_emotion_vocab else set(global_counts.keys())
            allowed = {w for w in allowed if in_counts.get(w, 0) >= min_freq_per_emotion}

            scores = {}
            for w in allowed:
                a = in_counts[w] + 1
                b = (in_total - in_counts[w]) + 1
                c = out_counts[w] + 1
                d = (out_total - out_counts[w]) + 1
                scores[w] = float(np.log(a / b) - np.log(c / d))

            if use_vader:
                if emo == "Positive" and POS_SET:
                    scores = {w: s for w, s in scores.items() if w in POS_SET}
                elif emo == "Negative" and NEG_SET:
                    scores = {w: s for w, s in scores.items() if w in NEG_SET}
                elif emo == "Neutral" and NEU_SET is not None:
                    scores = {w: s for w, s in scores.items() if w in NEU_SET}

            top_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:cloud_words_cap]
            if top_items:
                m = min(v for _, v in top_items)
                if m < 0:
                    top_items = [(w, v - m) for w, v in top_items]
            weights_per_emotion[emo] = dict(top_items)

    # B) TF‑IDF per class
    elif method.startswith("TF‑IDF"):
        st.info("TF‑IDF within each emotion’s documents (lexical salience).")
        vectorizer = TfidfVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False, min_df=5, max_df=0.8)
        for emo in EMOTIONS:
            docs = df_clean.loc[df_clean["Emotion"] == emo, "clean_text"].astype(str).tolist()
            clean_docs = []
            for d in docs:
                toks = [t for t in d.split() if token_ok(t) and GLOBAL_DF.get(t, 0) >= global_min_df]
                clean_docs.append(" ".join(toks))
            if not clean_docs:
                weights_per_emotion[emo] = {}
                continue
            X = vectorizer.fit_transform(clean_docs)
            vocab = np.array(vectorizer.get_feature_names_out())
            scores = np.asarray(X.sum(axis=0)).ravel()
            idx = np.argsort(-scores)[:cloud_words_cap]
            weights_per_emotion[emo] = {vocab[i]: float(scores[i]) for i in idx}

    # C) Word2Vec centroid (semantic)
    elif method.startswith("Word2Vec centroid (semantic)"):
        w2v_model = st.session_state.get("w2v_model", None)
        if w2v_model is None:
            st.error("Word2Vec model not found. Run **Word2Vec** or choose another method.")
            st.stop()

        st.info("Cosine similarity to each emotion’s unweighted Word2Vec centroid (semantic).")
        centroids = {}
        for emo in EMOTIONS:
            toks = [t for t, c in emo_counts[emo].items() for _ in range(c) if t in w2v_model.wv]
            if not toks:
                centroids[emo] = None
                continue
            vecs = np.vstack([w2v_model.wv[t] for t in toks]).astype("float32")
            centroids[emo] = vecs.mean(axis=0)

        for emo in EMOTIONS:
            c = centroids[emo]
            if c is None:
                weights_per_emotion[emo] = {}
                continue

            if limit_to_emotion_vocab:
                allow = {w for w, cnt in emo_counts[emo].items() if cnt >= min_freq_per_emotion}
                vocab_words = [w for w in w2v_model.wv.key_to_index if w in allow]
            else:
                vocab_words = list(w2v_model.wv.key_to_index.keys())

            c = c / (np.linalg.norm(c) + 1e-12)
            sims = {}
            for w in vocab_words:
                if not token_ok(w):  # keep alpha / length / domain stop
                    continue
                v = w2v_model.wv[w]
                v = v / (np.linalg.norm(v) + 1e-12)
                sims[w] = float(np.dot(v, c))

            top_items = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)[:cloud_words_cap]
            if top_items:
                m = min(val for _, val in top_items)
                if m < 0:
                    top_items = [(w, val - m) for w, val in top_items]
            weights_per_emotion[emo] = dict(top_items)

    # D) Word2Vec centroid (TF‑IDF‑weighted)
    else:
        w2v_model = st.session_state.get("w2v_model", None)
        if w2v_model is None:
            st.error("Word2Vec model not found. Run **Word2Vec** or choose another method.")
            st.stop()

        st.info("Cosine similarity to TF‑IDF‑weighted centroids (semantic + lexical).")

        def weighted_centroid(emo: str):
            docs = df_clean.loc[df_clean["Emotion"] == emo, "clean_text"].astype(str).tolist()
            clean_docs = []
            for d in docs:
                toks = [t for t in d.split() if token_ok(t) and GLOBAL_DF.get(t, 0) >= global_min_df]
                clean_docs.append(" ".join(toks))
            if not clean_docs:
                return None
            vec = TfidfVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False, min_df=2, max_df=0.9)
            X = vec.fit_transform(clean_docs)
            vocab = vec.get_feature_names_out()
            weights = np.asarray(X.sum(axis=0)).ravel()
            token2w = {tok: float(w) for tok, w in zip(vocab, weights) if tok in w2v_model.wv and w > 0}
            if not token2w:
                return None
            toks, ws = zip(*token2w.items())
            mat = np.vstack([w2v_model.wv[t] for t in toks]).astype("float32")
            w_arr = np.asarray(ws, dtype="float32").reshape(-1, 1)
            return (mat * w_arr).sum(axis=0) / (w_arr.sum() + 1e-12)

        centroids = {emo: weighted_centroid(emo) for emo in EMOTIONS}

        for emo in EMOTIONS:
            c = centroids[emo]
            if c is None:
                weights_per_emotion[emo] = {}
                continue

            if limit_to_emotion_vocab:
                allow = {w for w, cnt in emo_counts[emo].items() if cnt >= min_freq_per_emotion}
                vocab_words = [w for w in w2v_model.wv.key_to_index if w in allow]
            else:
                vocab_words = list(w2v_model.wv.key_to_index.keys())

            c = c / (np.linalg.norm(c) + 1e-12)
            sims = {}
            for w in vocab_words:
                if not token_ok(w):
                    continue
                v = w2v_model.wv[w]
                v = v / (np.linalg.norm(v) + 1e-12)
                sims[w] = float(np.dot(v, c))

            top_items = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)[:cloud_words_cap]
            if top_items:
                m = min(val for _, val in top_items)
                if m < 0:
                    top_items = [(w, val - m) for w, val in top_items]
            weights_per_emotion[emo] = dict(top_items)

    # render clouds
    st.markdown("---")
    cols = st.columns(len(EMOTIONS))
    for col, emo in zip(cols, EMOTIONS):
        with col:
            render_cloud(weights_per_emotion.get(emo, {}), f"{emo} — {method}", EMO2CMAP.get(emo, "viridis"))

    #  quick interpretation
    st.markdown("### Quick Interpretation")
    st.write(
        "- **Positive:** praise words (e.g., *delicious, amazing, wonderful*).  \n"
        "- **Negative:** complaint words (e.g., *awful, rancid, disappointing*).  \n"
        "- **Neutral:** descriptive / transactional terms (e.g., *package, shipped, ingredients*)."
    )



# FINAL PAGE: Predictions & Downloads (single + batch)

def page_predictions():
    """
     Predict emotion for new reviews (single text or uploaded CSV/TXT).
    Uses Word2Vec embeddings + the trained model saved in session_state.
    Shows selected model's accuracy and displays a local image per emotion.
    """

    st.title("Predictions")

# check dependencies
    w2v = st.session_state.get("w2v_model")
    label_map = st.session_state.get("label_map")
    rf_model = st.session_state.get("rf_model")
    xgb_model = st.session_state.get("xgb_model")

    if w2v is None or label_map is None:
        st.error("Word2Vec and label map not found. Please run **Word2Vec** first.")
        st.stop()
    if rf_model is None and xgb_model is None:
        st.error("No trained model found. Please run **Modeling** first.")
        st.stop()

    idx2lbl = {v: k for k, v in label_map.items()}
    ordered_labels = [idx2lbl[i] for i in range(len(idx2lbl))]

    # emoji + local image paths
    EMOJI = {"Negative": "😞", "Neutral": "😐", "Positive": "😀"}

    IMAGE_PATHS = {
        "Negative": r"C:\Users\Francesca Manu\PycharmProjects\Text_Group_Project\Negative.jpg",
        "Neutral":  r"C:\Users\Francesca Manu\PycharmProjects\Text_Group_Project\Neutral.jpg",
        "Positive": r"C:\Users\Francesca Manu\PycharmProjects\Text_Group_Project\Positive.jpg",  # fixed path
    }

    # last trained accuracy (optional)
    def last_accuracy_for(model_name: str) -> float | None:
        res = st.session_state.get("results")
        if res and "metrics" in res:
            df = pd.DataFrame(res["metrics"])
            row = df.loc[df["Model"] == model_name]
            if not row.empty and "Accuracy" in row.columns:
                return float(row["Accuracy"].values[0])
        res_df = st.session_state.get("model_results")
        if isinstance(res_df, pd.DataFrame) and model_name in res_df.index and "Accuracy" in res_df.columns:
            return float(res_df.loc[model_name, "Accuracy"])
        return None

    # cleaning & vectorization
    preprocess_text_fn = st.session_state.get("preprocess_text")
    stop_set = st.session_state.get("preproc_stopwords", set())

    def _fallback_clean(s: str) -> str:
        s = str(s).lower()
        s = re.sub(r"<.*?>", " ", s)
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\d+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        toks = [t for t in s.split() if len(t) > 1 and t not in stop_set]
        return " ".join(toks)

    def clean_text(raw: str) -> str:
        if callable(preprocess_text_fn):
            try:
                return preprocess_text_fn(raw, stop_set)
            except TypeError:
                return preprocess_text_fn(raw)
        return _fallback_clean(raw)

    def doc_vec(tokens: list[str]) -> np.ndarray:
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        return np.mean(vecs, axis=0).astype("float32") if vecs else np.zeros(w2v.vector_size, dtype="float32")

    def vectorize_texts(texts: list[str]) -> np.ndarray:
        toks_list = [clean_text(t).split() for t in texts]
        return np.vstack([doc_vec(toks) for toks in toks_list])

    #  model picker
    model_options = []
    if xgb_model is not None: model_options.append("XGBoost")
    if rf_model  is not None: model_options.append("Random Forest")

    model_choice = st.radio("Choose model for prediction", model_options, horizontal=True)
    acc = last_accuracy_for(model_choice)
    if acc is not None:
        st.caption(f"**{model_choice}** accuracy (last training run): **{acc:.3f}**")

    def predict_proba(X: np.ndarray) -> np.ndarray:
        mdl = xgb_model if model_choice == "XGBoost" else rf_model
        return mdl.predict_proba(X)

    #  single prediction
    st.markdown("###  Single Review")
    txt = st.text_area("Enter a review", height=120, placeholder="Type or paste a review…")

    if st.button("Predict emotion", type="primary"):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            X = vectorize_texts([txt])
            p = predict_proba(X)[0]
            pred_idx = int(np.argmax(p))
            pred_lbl = idx2lbl[pred_idx]
            conf = float(p[pred_idx])

            # 1) Predicted sentence
            st.success(f"The predicted emotion is {EMOJI.get(pred_lbl,'')} **{pred_lbl}** (confidence **{conf:.3f}**).")

            # 2) Image immediately below the sentence
            img_path = IMAGE_PATHS.get(pred_lbl)
            if img_path and os.path.exists(img_path):
                st.image(img_path, caption=pred_lbl, width=260)
            else:
                st.caption(f"(No image found for {pred_lbl}. Expected at: {img_path})")

            # 3) Probabilities as a bullet list (sorted descending)
            pairs = [(lbl, float(p[label_map[lbl]])) for lbl in ordered_labels]
            pairs.sort(key=lambda t: t[1], reverse=True)
            st.markdown("**Probabilities:**")
            for lbl, prob in pairs:
                st.markdown(f"- **{lbl}**: {prob:.4f}")

    st.markdown("---")

    #  batch CSV prediction
    st.markdown("### Batch Prediction (CSV)")
    st.caption("Upload a CSV with a **Text** column (or **clean_text**). We’ll return predictions and probabilities.")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        text_col = "Text" if "Text" in df_in.columns else ("clean_text" if "clean_text" in df_in.columns else None)
        if text_col is None:
            st.error("CSV must contain a 'Text' or 'clean_text' column.")
            return

        st.write(f"Found **{len(df_in):,}** rows. Using column **{text_col}**.")
        if st.button("Run batch prediction"):
            texts = df_in[text_col].astype(str).tolist()
            Xb = vectorize_texts(texts)
            Pb = predict_proba(Xb)
            pred_idx = Pb.argmax(axis=1)
            pred_lbl = [idx2lbl[i] for i in pred_idx]
            confs = Pb.max(axis=1)

            out = df_in.copy()
            out["PredictedEmotion"] = pred_lbl
            out["Confidence"] = confs
            for lbl in ordered_labels:
                out[f"Prob_{lbl}"] = Pb[:, label_map[lbl]]

            st.success("Batch prediction complete ✅")
            st.dataframe(out.head(20), use_container_width=True)
            st.download_button(
                " Download predictions (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{model_choice.lower().replace(' ','_')}.csv",
                mime="text/csv"
            )
# ROUTER

PAGES = {
    "HomePage": page_home,
    "Data Load": page_data_load,
    "Preprocess & Labels": page_preprocess,
    "Post‑Cleaning Diagnostics": page_diagnostics,
    "Embeddings (Word2Vec)": page_word2vec,
    "Modeling (RF & XGBoost)": page_modeling,
    "Model Evaluation & Results":page_ModelEvaluation_and_Results,
    "Word Clouds": page_wordclouds,
    "Prediction Page":page_predictions,
}
#Sidebarlogo
st.sidebar.image(
        "C:/Users/Francesca Manu/PycharmProjects/Text_Group_Project/Homepage.png",
        use_column_width=False,
        width=200,
)
# Sidebar navigation
choice = st.sidebar.selectbox("Please Select Page", list(PAGES.keys()))
PAGES[choice]()


