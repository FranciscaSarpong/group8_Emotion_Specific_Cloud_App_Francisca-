# ==== Imports & global setup ====
import os, re, io, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image

import streamlit as st
from gensim.models import Word2Vec
import hashlib

from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,
    roc_curve, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

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

# Streamlit safety
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass

st.set_page_config(page_title="Emotion Word Clouds (Word2Vec)", layout="wide")



PAGE_HELP = {
    "Home": "Overview of the project, goals, objectives, and navigation.",
    "Data Load": "Load via path or upload CSV/Parquet, preview/EDA, cache to session.",
    "Preprocess & Labels": "Clean text and map Score→Emotion. Produces df_clean with clean_text & Emotion.",
    "Post-Cleaning Diagnostics": "Quality checks before embeddings.",
    "Embeddings (Word2Vec)": "Train W2V on TRAIN only; build doc vectors (mean or SIF).",
    "Modeling (RF & XGBoost)": "Stratified CV + Hold-out results. Class weights/SMOTE options.",
    "Model Evaluation & Results": "Compare CV vs Test metrics, CMs, ROC, importance.",
    "Word Clouds": "Emotion-specific word clouds (contrastive log-odds, centroid similarity).",
    "Prediction Page": "Single/batch predictions using chosen model."
}

# ==== Helpers ====

def score_to_emotion(score: int) -> str:
    """Amazon 1–5 stars → emotion label."""
    try:
        s = int(score)
    except Exception:
        return "Unknown"
    if s <= 2: return "Negative"
    if s == 3:  return "Neutral"
    return "Positive"

def safe_coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Score to Int64 and parse epoch Time → datetime if present."""
    if "Score" in df.columns:
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce").astype("Int64")
    if "Time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Time"]):
        df["Time"] = pd.to_datetime(df["Time"], unit="s", errors="coerce")
    return df

@st.cache_data(show_spinner=True)
def load_reviews_uploaded(file_obj, usecols=None, nrows=None) -> pd.DataFrame:
    """Read from Streamlit uploader (csv/parquet)."""
    if file_obj is None:
        return pd.DataFrame()
    name = file_obj.name.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(file_obj, columns=usecols if usecols else None)
    else:
        df = pd.read_csv(file_obj, usecols=usecols if usecols else None, nrows=nrows)
    return safe_coerce_types(df).reset_index(drop=True)

@st.cache_data(show_spinner=True)
def load_reviews_from_path(path: str, usecols=None, nrows=None) -> pd.DataFrame:
    """Read from a local absolute path (csv/parquet)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    p = path.lower()
    if p.endswith(".parquet"):
        df = pd.read_parquet(path, columns=usecols if usecols else None)
    else:
        df = pd.read_csv(path, usecols=usecols if usecols else None, nrows=nrows, encoding="utf-8")
    return safe_coerce_types(df).reset_index(drop=True)



# PAGES

def page_home():
    st.title(" Emotion-Specific Word Cloud from Amazon Reviews (Word2Vec)")
    st.caption(PAGE_HELP.get("Home", ""))

    # Create columns to position logo on the right
    col1, col2 = st.columns([2, 1])

    with col2:
        # Home page logo (use a relative path or put image in /assets)
        st.image(
            "Group82.png",
            width=300,
            caption="Emotion-Specific Word Cloud from Amazon Reviews"
        )

    with col1:
        st.markdown("""
        ### **Goal**
        Build a pipeline that analyzes Amazon Fine Food Reviews, detects **emotions** (Positive / Neutral / Negative),
        and visualizes vocabulary per emotion via **word clouds**, using **Word2Vec** + **Random Forest vs XGBoost**.

        ### **Navigation**
        Use the sidebar to move step-by-step: Data Load → Preprocessing → Embeddings → Modeling → Evaluation → Word Clouds → Conclusion.
        """)

    # Team
    st.markdown("---")
    st.markdown("### Project Team")
    team = [
        ("George Owell", "22256146", "http://emotionbasedwordcloud-kaqsykwxudaljgan3srpws.streamlit.app"),
        ("Francisca Manu Sarpong", "22255796", ""),
        ("Franklina Oppong", "11410681", ""),
        ("Ewurabena Biney", "22252464", ""),
        ("Esther Edem Tulasi Carr", "22253335", ""),
    ]
    c1, c2, c3 = st.columns([4, 2, 6])
    with c1:
        st.markdown("**Name**")
        for n, _, _ in team: st.markdown(n)
    with c2:
        st.markdown("**Student ID**")
        for _, sid, _ in team: st.markdown(sid)
    with c3:
        st.markdown("**App Link**")
        for _, _, role in team: st.markdown(role)

    st.info(" Start with **Data Load** in the sidebar.")

def page_data_load():
    st.title("Data Load")
    st.caption(PAGE_HELP["Data Load"])
    st.markdown("Load from **local path** or **file uploader**. We’ll cache the DataFrame and show quick EDA.")

    # --- Inputs ---
    default_cols = ["Score", "Summary", "Text"]
    st.markdown("#### Columns to load")
    selected_cols = st.multiselect("Choose columns (fewer → faster)", default_cols,
                                   default=["Score", "Summary"])

    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        first_n = st.checkbox("Read only first N (CSV)", value=True)
    with c2:
        nrows = st.number_input("N rows", min_value=5_000, value=20_000, step=5_000)
    with c3:
        _seed = st.number_input("Random seed (for any sampling)", min_value=0, value=42, step=1)

    # --- Path loader ---
    st.markdown("#### Load from local path")
    local_path = st.text_input("Absolute path (.csv or .parquet)",
                               value="Reviewsample.csv")
    btn_path = st.button("Load from path")

    # --- Uploader loader ---
    st.markdown("#### Or upload a file")
    up = st.file_uploader("Upload file", type=["csv", "parquet"])
    btn_up = st.button("Load from upload")

    df = None
    try:
        if btn_path:
            df = load_reviews_from_path(
                local_path.strip(),
                usecols=selected_cols if selected_cols else None,
                nrows=int(nrows) if (first_n and local_path.lower().endswith(".csv")) else None
            )
        elif btn_up:
            if up is None:
                st.error("Please upload a CSV or Parquet file.")
                return
            df = load_reviews_uploaded(
                up,
                usecols=selected_cols if selected_cols else None,
                nrows=int(nrows) if (first_n and up.name.lower().endswith(".csv")) else None
            )
        else:
            st.stop()  # wait for a button press
    except Exception as e:
        st.error(f"Unexpected error while loading data: {e}")
        st.stop()

    if df is None or df.empty:
        st.error("Loaded an empty DataFrame. Check the path/file and selected columns.")
        st.stop()

    # Cache to session for later pages
    st.session_state["df"] = df
    st.success(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # --- Quick EDA ---
    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Column Types & Missingness")
    info = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str),
        "Non-Null": df.notnull().sum(),
        "Nulls": df.isnull().sum(),
        "Null %": (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(info, use_container_width=True)

    st.subheader("Duplicates")
    st.write(f"Exact duplicate rows: **{df.duplicated().sum():,}**")

    if "Score" in df.columns:
        st.subheader("Score & Emotion Distribution")
        st.bar_chart(df["Score"].value_counts(dropna=False).sort_index())
        emo = df["Score"].dropna().astype(int).map(score_to_emotion)
        st.bar_chart(emo.value_counts())

    if "Time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Time"]):
        tmin, tmax = df["Time"].min(), df["Time"].max()
        if pd.notnull(tmin) and pd.notnull(tmax):
            st.write(f"Time coverage: **{tmin.date()} → {tmax.date()}**")

    st.info("Dataset cached. Next step: **Preprocess & Labels** page.")


def page_preprocess():
    st.title("Preprocess & Emotion Mapping")
    st.caption("Clean **Summary**, map stars to emotions, and cache as df_clean.")

    df = st.session_state.get("df")
    if df is None:
        st.error("No dataset found. Please load data in *Data Load* first.")
        st.stop()

    # ---- Which text to use ----
    st.markdown("#### Text Source")
    use_summary_only = st.checkbox("Use Summary only (recommended for this project)", value=True)
    fallback_to_text  = st.checkbox("Fallback to Text when Summary is missing/empty", value=True)

    # ---- Cleaning options ----
    c1, c2, c3 = st.columns(3)
    with c1:
        do_lower    = st.checkbox("Lowercase", True)
        rm_html     = st.checkbox("Remove HTML-like tags", True)
        rm_punct    = st.checkbox("Remove punctuation (after expanding negations)", True)
    with c2:
        rm_digits   = st.checkbox("Remove digits", True)
        collapse_ws = st.checkbox("Collapse extra spaces", True)
        drop_empty  = st.checkbox("Drop empty/NA cleaned rows", True)
    with c3:
        use_nltk_stop = st.checkbox("Use NLTK stopwords", True)
        do_lemmatize  = st.checkbox("Lemmatize (WordNet)", True)
        do_stem       = st.checkbox("Stem (Porter) — not recommended for Word2Vec", False)

    # Guardrail: Word2Vec ≠ stemming
    if do_stem:
        st.info("Stemming may harm Word2Vec semantics; consider lemmatization only.")

    # Row limits & hygiene
    col_a, col_b = st.columns(2)
    with col_a:
        limit_rows = st.checkbox("Process only first N rows", True)
    with col_b:
        n_limit = st.number_input("N rows (if limited)", min_value=5_000, value=20_000, step=5_000)
    rm_dups = st.checkbox("Remove exact duplicate rows before cleaning", value=False)

    # ---- NLTK assets (lazy) ----
    @st.cache_resource(show_spinner=False)
    def _ensure_nltk_assets():
        import nltk
        try: nltk.data.find("corpora/stopwords")
        except: nltk.download("stopwords", quiet=True)
        try:
            nltk.data.find("corpora/wordnet"); nltk.data.find("corpora/omw-1.4")
        except:
            nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)

    FALLBACK_STOPWORDS = set("""
    a an the and or but if while with without within into onto from to for of on in out by up down over under again further
    is are was were be been being do does did doing have has had having this that these those it its i me my we our you your
    he him his she her they them their what which who whom where when why how all any both each few more most other some such
    no nor not only own same so than too very can will just should now
    """.split())

    # Keep negators even if they appear in stopwords
    NEGATORS = {"not", "no", "never", "nor", "cannot", "can_not"}

    def get_stopwords():
        if use_nltk_stop:
            try:
                _ensure_nltk_assets()
                from nltk.corpus import stopwords as sw
                sw_set = set(sw.words("english"))
            except Exception:
                st.warning("NLTK stopwords unavailable; using fallback.")
                sw_set = set(FALLBACK_STOPWORDS)
        else:
            sw_set = set(FALLBACK_STOPWORDS)
        # ensure negators stay
        sw_set = {w for w in sw_set if w not in NEGATORS}
        return sw_set

    # --- small normalizer helpers ---
    CONTRACTIONS = [
        (r"won't", "will not"),
        (r"can't", "can not"),
        (r"ain't", "is not"),
        (r"n['’]t\b", " not"),
        (r"y['’]all", "you all"),
        (r"gonna", "going to"),
        (r"wanna", "want to"),
    ]

    def expand_contractions(s: str) -> str:
        for pat, rep in CONTRACTIONS:
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        return s

    def preprocess_text(text: str, stop_set: set) -> str:
        """
        Normalize summary text with negation-aware cleaning.
        """
        s = str(text)

        if do_lower:
            s = s.lower()

        # Expand contractions FIRST so we preserve negations
        s = expand_contractions(s)

        if rm_html:
            s = re.sub(r"<.*?>", " ", s)

        # Remove punctuation AFTER expanding negations (apostrophes already handled)
        if rm_punct:
            s = re.sub(r"[^\w\s]", " ", s)

        if rm_digits:
            s = re.sub(r"\d+", " ", s)

        if collapse_ws:
            s = re.sub(r"\s+", " ", s).strip()

        toks = s.split()
        if not toks:
            return ""

        # Keep tokens not in stopwords OR are negators
        toks = [t for t in toks if len(t) > 1 and (t not in stop_set or t in NEGATORS)]

        if do_lemmatize or do_stem:
            try:
                _ensure_nltk_assets()
                if do_lemmatize:
                    from nltk.stem import WordNetLemmatizer
                    lem = WordNetLemmatizer()
                    toks = [lem.lemmatize(t) for t in toks]
                if do_stem:
                    from nltk.stem import PorterStemmer
                    ps = PorterStemmer()
                    toks = [ps.stem(t) for t in toks]
            except Exception:
                st.warning("Lemmatizer/stemmer unavailable; skipping.")

        return " ".join(toks)

    if st.button("Run Preprocessing"):
        work = df.copy()
        if limit_rows:
            work = work.head(int(n_limit)).copy()
        if rm_dups:
            before = len(work)
            work = work.drop_duplicates().reset_index(drop=True)
            st.info(f"Removed {before - len(work)} duplicate rows.")

        # ---- Ensure required columns ----
        need_cols = ["Score", "Summary"]
        if not all(c in work.columns for c in need_cols):
            st.error(f"Required columns missing. Found: {list(work.columns)}. Need: {need_cols}.")
            st.stop()

        work["Score"]   = pd.to_numeric(work["Score"], errors="coerce").astype("Int64")
        work["Summary"] = work["Summary"].astype(str)

        # Choose source text
        if use_summary_only:
            # summary only; optionally fall back to Text when summary is empty
            if fallback_to_text and "Text" in work.columns:
                text_fallback = work["Text"].astype(str)
                src = np.where(work["Summary"].str.strip().eq(""), text_fallback, work["Summary"])
                work["__source_text"] = pd.Series(src, index=work.index).astype(str)
            else:
                work["__source_text"] = work["Summary"]
        else:
            # combine Summary + Text (if you want to compare later)
            if "Text" in work.columns:
                work["__source_text"] = (work["Summary"].fillna("") + " " + work["Text"].astype(str).fillna("")).str.strip()
            else:
                work["__source_text"] = work["Summary"]

        stop_set = get_stopwords()

        st.markdown("#### Cleaning Summaries…")
        with st.spinner("Normalizing summaries (negation-aware)…"):
            work["clean_text"] = work["__source_text"].apply(lambda x: preprocess_text(x, stop_set))

        if drop_empty:
            before = len(work)
            work = work[work["clean_text"].str.len() > 0].reset_index(drop=True)
            removed = before - len(work)
            if removed > 0:
                st.info(f"Dropped {removed} rows with empty cleaned text.")

        # Map to Emotion (1–2 Neg, 3 Neu, 4–5 Pos)
        work["Emotion"] = work["Score"].apply(score_to_emotion)

        # Show quick outputs
        st.subheader("Emotion Distribution")
        emo_counts = work["Emotion"].value_counts()
        st.bar_chart(emo_counts)

        st.subheader("Preview (first 12)")
        cols = [c for c in ["Score", "Emotion", "Summary", "clean_text"] if c in work.columns]
        st.dataframe(work[cols].head(12), use_container_width=True)

        # Cache
        st.session_state["df_clean"] = work.drop(columns=["__source_text"])
        st.session_state["preprocess_text"] = preprocess_text
        st.session_state["preproc_stopwords"] = stop_set

        # Download
        st.markdown("### Download Cleaned Subset")
        st.download_button(
            "Download cleaned CSV",
            data=work.drop(columns=["__source_text"]).to_csv(index=False).encode("utf-8"),
            file_name="amazon_reviews_cleaned.csv",
            mime="text/csv"
        )
        st.success("Preprocessing complete — df_clean ready (cleaned on **Summary**).")


# STEP 4: POST‑CLEANING DIAGNOSTICS
def page_diagnostics():
    """
    Post-Cleaning Diagnostics
    -------------------------
    Validates data quality before embeddings & modeling:
      • Corpus shape, token length distribution (+ quantiles)
      • Vocabulary size & lexical richness (TTR, Herdan's C, hapax)
      • Top tokens (overall + per-emotion)
      • Emotion distribution
      • Short-review & duplicate guardrails (exact + stable MD5 hash)
      • Optional TF-IDF signal check (diagnostic only)
    Writes back to st.session_state['df_clean'] ONLY when you confirm.
    """

    st.title(" Post-Cleaning Diagnostics")
    st.markdown("""
    This page **validates data quality** before embeddings & modeling:
    - Corpus shape, token length distribution  
    - Vocabulary size & lexical richness  
    - Top tokens (overall & per emotion)  
    - Emotion distribution  
    - Short-review & duplicate guardrails  
    - *(Optional)* overall TF-IDF signal check
    """)

    # 0) Load cleaned data
    df = st.session_state.get("df_clean")
    if df is None or "clean_text" not in df.columns or "Emotion" not in df.columns:
        st.error("No cleaned dataset found. Please run **Preprocess & Emotion Mapping** first.")
        st.stop()
    df = df.copy()

    # Optional sampling for very large datasets (keeps page responsive)
    with st.expander("Performance options"):
        do_sample = st.checkbox("Diagnose on a random sample", value=False)
        # adapt bounds to dataset size to avoid forcing 5k on small sets
        max_n = int(min(200_000, len(df)))
        default_n = int(min(20_000, max_n)) if max_n >= 20_000 else max_n
        sample_n = st.number_input("Sample size", 1_000, max_n, default_n, 1_000)
        sample_seed = st.number_input("Sample seed", 0, 9999, 42, 1)

    if do_sample and len(df) > sample_n:
        df = df.sample(n=int(sample_n), random_state=int(sample_seed)).reset_index(drop=True)
        try:
            total_rows = st.session_state["df_clean"].shape[0]
        except Exception:
            total_rows = "?"
        st.info(f"Diagnosing on a random sample of **{len(df):,}** rows (out of {total_rows:,}).")

    # 1) Corpus shape & token lengths
    st.subheader("Corpus Shape & Hygiene")
    rows = len(df)
    nn_clean = df["clean_text"].notna().sum()
    nn_emot  = df["Emotion"].notna().sum()
    st.write(
        f"Rows: **{rows:,}** | non-null `clean_text`: **{nn_clean:,}** | "
        f"non-null `Emotion`: **{nn_emot:,}**"
    )

    # token lengths (fast split count)
    lengths = df["clean_text"].fillna("").str.split().str.len()
    m1, m2, m3 = st.columns(3)
    m1.metric("Median tokens/review", int(np.median(lengths)))
    m2.metric("Mean tokens/review", f"{np.mean(lengths):.2f}")
    m3.metric("Max tokens/review", int(np.max(lengths)))

    # helpful quantiles to understand tails
    try:
        q10, q50, q90, q99 = np.quantile(lengths, [0.1, 0.5, 0.9, 0.99])
        st.caption(f"Length quantiles — 10%: {q10:.0f}, 50%: {q50:.0f}, 90%: {q90:.0f}, 99%: {q99:.0f}")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(lengths, bins=80, kde=False, ax=ax)
    ax.set_title("Distribution of Review Lengths (tokens)")
    ax.set_xlabel("Tokens per review"); ax.set_ylabel("Count")
    st.pyplot(fig)

    # 2) Lexical richness
    st.subheader("Vocabulary & Lexical Richness")

    @st.cache_data(show_spinner=False)
    def _token_series(clean_col: pd.Series) -> pd.Series:
        return clean_col.fillna("").str.split()

    tokens_series = _token_series(df["clean_text"])

    @st.cache_data(show_spinner=False)
    def _vocab_counter(tokens: pd.Series) -> Counter:
        return Counter(t for row in tokens for t in row)

    vocab_counter = _vocab_counter(tokens_series)
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
    st.caption(f"Hapax proportion (frequency = 1): **{hapax_prop:.3f}**")

    # 3) Emotion distribution
    st.subheader("Emotion Distribution")
    emo_counts = df["Emotion"].value_counts()
    st.bar_chart(emo_counts)
    st.dataframe(emo_counts.rename("Count").to_frame(), use_container_width=True)

    # 4) Top tokens overall (frequency) + per emotion
    st.subheader("Top Tokens — Overall (frequency)")
    top_overall_k = st.slider("Show top N tokens", 10, 50, 20, 5, key="diag_top_overall_k")
    top_overall = pd.DataFrame(vocab_counter.most_common(top_overall_k),
                               columns=["Token", "Frequency"])
    st.dataframe(top_overall, use_container_width=True)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.barplot(y="Token", x="Frequency", data=top_overall, ax=ax2)
    ax2.set_title(f"Top {top_overall_k} Tokens (Overall)")
    st.pyplot(fig2)

    # Per-emotion token view (helps spot leakage/bias)
    st.subheader("Top Tokens — Per Emotion")
    per_k = st.slider("Top N per emotion", 5, 40, 15, 5, key="diag_top_per_emo")
    cols = st.columns(min(3, df["Emotion"].nunique()))
    for i, emo in enumerate(sorted(df["Emotion"].dropna().unique())):
        with cols[i % len(cols)]:
            c = Counter(t for row in tokens_series[df["Emotion"] == emo] for t in row)
            top_e = pd.DataFrame(c.most_common(per_k), columns=["Token", "Frequency"])
            st.markdown(f"**{emo}**")
            st.dataframe(top_e, use_container_width=True)

    # 5) Optional: overall TF-IDF signal check (diagnostic only)
    with st.expander("Optional: Overall TF-IDF Signal Check"):
        max_feats = st.slider("Max features", 1_000, 12_000, 4_000, 500,
                              help="Limit vocabulary to keep this fast.")
        token_pattern = r"(?u)\b\w+\b"  # keep all word chars; your clean_text already removed most noise

        @st.cache_data(show_spinner=False)
        def _tfidf_overall(clean_text: pd.Series, max_features: int):
            vec = TfidfVectorizer(max_features=max_features, token_pattern=token_pattern)
            X = vec.fit_transform(clean_text.fillna(""))
            vocab = np.array(vec.get_feature_names_out())
            mean_scores = np.asarray(X.mean(axis=0)).ravel()
            return vocab, mean_scores

        vocab, mean_scores = _tfidf_overall(df["clean_text"], int(max_feats))
        k = st.slider("Top N tokens to display", 10, 40, 20, 5, key="diag_tfidf_topk")
        order = np.argsort(mean_scores)[::-1][:k]
        tfidf_overall = pd.DataFrame({"Token": vocab[order], "Mean TF-IDF": mean_scores[order]})
        st.dataframe(tfidf_overall, use_container_width=True)

    # 6) Guardrail: short reviews
    st.subheader("Short Reviews (≤ threshold tokens)")
    thr = st.slider("Threshold (tokens)", 1, 10, 2, 1, key="diag_short_thr")
    short_mask = lengths <= thr
    n_short = int(short_mask.sum())
    st.write(f"Found **{n_short:,}** reviews with ≤ {thr} tokens.")
    with st.expander("Preview some short reviews"):
        st.dataframe(
            df.loc[short_mask, ["Score", "Emotion", "clean_text"]].head(20),
            use_container_width=True
        )

    apply_short = st.checkbox("Remove these short reviews and update session",
                              value=False, key="diag_drop_short")
    # We do NOT mutate df in place; only write back if confirmed:
    if apply_short and n_short > 0:
        df2 = df.loc[~short_mask].reset_index(drop=True)
        st.session_state["df_clean"] = df2
        st.success(f"Removed {n_short:,} rows. Session updated. Re-run diagnostics if you wish.")

    # 7) Optional: duplicate detection (exact + stable hash)
    with st.expander("Optional: Duplicates / near-duplicates"):
        st.caption("Uses exact `clean_text` and a **stable MD5** content hash (Python's builtin hash is not stable).")

        # A) exact duplicates by cleaned text
        dup_mask_exact = df["clean_text"].duplicated(keep="first")
        n_dup_exact = int(dup_mask_exact.sum())
        st.write(f"Exact duplicates by `clean_text`: **{n_dup_exact:,}**")

        # B) stable content hash (md5 of cleaned text)
        @st.cache_data(show_spinner=False)
        def _content_hashes(series: pd.Series) -> pd.Series:
            return series.fillna("").map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())

        hashes = _content_hashes(df["clean_text"])
        dup_mask_hash = hashes.duplicated(keep="first")
        n_dup_hash = int(dup_mask_hash.sum())
        st.write(f"Stable-hash duplicates: **{n_dup_hash:,}**")

        if n_dup_exact > 0:
            st.dataframe(df.loc[dup_mask_exact, ["Emotion", "clean_text"]].head(20), use_container_width=True)

        # Drop options (separate; pick one)
        drop_exact = st.checkbox("Drop exact `clean_text` duplicates and update session",
                                 value=False, key="diag_drop_dups_exact")
        if drop_exact and n_dup_exact > 0:
            df2 = df.loc[~dup_mask_exact].reset_index(drop=True)
            st.session_state["df_clean"] = df2
            st.success(f"Dropped {n_dup_exact:,} exact duplicates. Session updated.")

        drop_hash = st.checkbox("Drop stable-hash duplicates and update session",
                                value=False, key="diag_drop_dups_hash")
        if drop_hash and n_dup_hash > 0:
            df2 = df.loc[~dup_mask_hash].reset_index(drop=True)
            st.session_state["df_clean"] = df2
            st.success(f"Dropped {n_dup_hash:,} hash-identified duplicates. Session updated.")

    st.info("Diagnostics complete. If you applied any guardrails, re-run this page to see updated stats.")


def page_word2vec():
    """
    Embeddings (Word2Vec)
    ---------------------
    - Trains Word2Vec on TRAIN split only (prevents leakage).
    - Builds review vectors by mean or SIF (train-only token frequencies).
    - Optional: remove 1st principal component (full SIF) and/or L2-normalize.
    - Reports vocab size, zero-vector rate, and in-vocab token coverage (overall & per emotion).
    Saves in session:
      w2v_model, X_emb, y_labels, label_map, train_index, test_index, embedding_used
    """

    st.title("Embeddings (Word2Vec)")
    st.caption("Train on TRAIN only; build review vectors; cache for modeling. Optional full SIF + L2 norm.")

    # ---------- Guards ----------
    df_clean = st.session_state.get("df_clean")
    if df_clean is None or "clean_text" not in df_clean.columns or "Emotion" not in df_clean.columns:
        st.error("Run **Preprocess & Emotion Mapping** first.")
        st.stop()

    # Tokens
    work = df_clean[["clean_text", "Emotion", "Score"]].copy()
    work["tokens"] = work["clean_text"].astype(str).apply(str.split)

    # ---------- Hyper-parameters & options ----------
    c1, c2, c3 = st.columns(3)
    with c1:
        vector_size = int(st.number_input("vector_size", 50, 600, 200, 25))
        window      = int(st.number_input("window", 2, 15, 5, 1))
        min_count   = int(st.number_input("min_count", 1, 20, 5, 1))
    with c2:
        sg_choice   = st.selectbox("Architecture", ["Skip-gram (sg=1)", "CBOW (sg=0)"], index=0)
        epochs      = int(st.number_input("epochs", 3, 50, 10, 1))
        negative    = int(st.number_input("negative sampling (0=off)", 0, 20, 10, 1))
    with c3:
        test_size   = float(st.slider("Test size (hold-out)", 0.1, 0.4, 0.2, 0.05))
        seed        = int(st.number_input("random_state / seed", 0, 9999, 42, 1))
        use_sif     = st.checkbox("Use SIF token weighting", value=False,
                                  help="Smooth Inverse Frequency weighting on tokens (TRAIN-only stats).")

    # Advanced options
    with st.expander("Advanced (recommended defaults)"):
        sample = float(st.number_input("Downsampling of very frequent words (sample)", 0.0, 0.01, 1e-3, 1e-4,
                                       help="Higher → more aggressive subsampling of very frequent words. 1e-3 is a strong default."))
        remove_pc = st.checkbox("Remove 1st principal component (full SIF)", value=use_sif,
                                help="Standard SIF: subtract projection on top PC fitted on TRAIN embeddings.")
        do_l2     = st.checkbox("L2-normalize review vectors", value=True)

    sg = 1 if "sg=1" in sg_choice else 0
    st.caption("Tip: With vocab ≈30–40k, min_count=5, vector_size=200, sg=1, sample≈1e-3 are strong defaults.")

    # ---------- Stratified split ----------
    trn, tst = train_test_split(
        work, test_size=test_size, random_state=seed, stratify=work["Emotion"]
    )
    st.caption(f"Training Word2Vec on TRAIN only: **{len(trn):,}** documents.")

    # ---------- Train Word2Vec (cached) ----------
    @st.cache_resource(show_spinner=True)
    def train_w2v(corpus_tokens, vec, win, minc, sg, epochs, negative, sample, seed):
        model = Word2Vec(
            vector_size=vec,
            window=win,
            min_count=minc,
            sg=sg,
            negative=negative,
            sample=sample,
            workers=min(4, os.cpu_count() or 1),
            seed=seed
        )
        model.build_vocab(corpus_tokens)
        model.train(corpus_tokens, total_examples=len(corpus_tokens), epochs=epochs)
        return model

    with st.spinner("Training Word2Vec…"):
        w2v = train_w2v(
            trn["tokens"].tolist(), vector_size, window, min_count, sg, epochs, negative, sample, seed
        )

    vocab_size = len(w2v.wv.key_to_index)
    st.success(f"✅ Trained Word2Vec. Vocabulary size (after min_count={min_count}): **{vocab_size:,}**")

    # ---------- Nearest-neighbor sanity check ----------
    st.markdown("#### Nearest words (sanity check)")
    default_terms = [w for w in ["good", "bad", "love", "terrible", "taste"] if w in w2v.wv]
    options_list = list(w2v.wv.key_to_index.keys())
    probe_terms = st.multiselect(
        "Pick words to inspect",
        options=options_list[: min(8000, len(options_list))],
        default=default_terms
    )
    for term in probe_terms:
        try:
            sims = w2v.wv.most_similar(term, topn=8)
            st.write("**" + term + "** → " + ", ".join([f"{w} ({s:.2f})" for w, s in sims]))
        except KeyError:
            pass

    # Optional tiny intrinsic check
    def _cos(a, b):
        if a not in w2v.wv or b not in w2v.wv: return None
        va, vb = w2v.wv[a], w2v.wv[b]
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    if {"good", "great", "terrible"} <= set(w2v.wv.key_to_index):
        c1 = _cos("good", "great"); c2 = _cos("good", "terrible")
        if c1 is not None and c2 is not None:
            st.caption(f"Intrinsic sanity: cos(good, great)={c1:.3f} vs cos(good, terrible)={c2:.3f}")

    # ---------- Build review vectors (mean or SIF) ----------
    dim = vector_size

    # Train-only token frequency for SIF
    train_token_freq = Counter(t for row in trn["tokens"] for t in row)
    total_train_tokens = max(sum(train_token_freq.values()), 1)
    a = 1e-3  # SIF smoothing (fixed strong default)

    def sif_weight(tok: str) -> float:
        return a / (a + train_token_freq.get(tok, 0) / total_train_tokens)

    def doc_vector(tokens, model, dim, use_sif=False):
        vecs = []
        for t in tokens:
            if t in model.wv:
                w = sif_weight(t) if use_sif else 1.0
                vecs.append(w * model.wv[t])
        return np.mean(vecs, axis=0).astype("float32") if vecs else np.zeros(dim, dtype="float32")

    # All-doc embeddings (built with the model trained on TRAIN)
    X_emb = np.vstack([doc_vector(toks, w2v, dim, use_sif=use_sif) for toks in work["tokens"]])

    # Optional: full SIF (remove 1st principal component) — fit PC on TRAIN ONLY
    if use_sif and remove_pc and len(trn) > 2:
        trn_mask = work.index.isin(trn.index)
        svd = TruncatedSVD(n_components=1, random_state=seed)
        try:
            svd.fit(X_emb[trn_mask, :])
            u = svd.components_[0]  # shape (dim,)
            # subtract projection on u
            X_emb = X_emb - (X_emb @ u[:, None]) * u[None, :]
            st.caption("Applied full SIF: removed 1st principal component fitted on TRAIN.")
        except Exception:
            st.warning("Could not apply PC removal (SVD issue). Proceeding without it.")

    # Optional: L2-normalize
    if do_l2:
        X_emb = X_emb / (np.linalg.norm(X_emb, axis=1, keepdims=True) + 1e-12)
        st.caption("Applied L2 normalization to review vectors.")

    # QA stats
    zero_rows = int((X_emb == 0).all(axis=1).sum())
    st.caption(f"Zero-vector reviews (all tokens OOV): **{zero_rows}** "
               f"({zero_rows / len(work):.2%}). Consider lowering `min_count` if high.")

    # Token coverage (in-vocab) overall & per emotion
    def token_coverage(rows_tokens) -> tuple[int, int]:
        total, invocab = 0, 0
        for toks in rows_tokens:
            total += len(toks)
            invocab += sum(1 for t in toks if t in w2v.wv)
        return invocab, total

    inv, tot = token_coverage(work["tokens"])
    st.write(f"**Token coverage overall**: {inv:,}/{tot:,} = {(inv/max(tot,1)):.2%}")

    cov_cols = st.columns(min(3, work["Emotion"].nunique()))
    for i, emo in enumerate(sorted(work["Emotion"].unique())):
        with cov_cols[i % len(cov_cols)]:
            inv_e, tot_e = token_coverage(work.loc[work["Emotion"] == emo, "tokens"])
            st.metric(f"{emo} coverage", f"{(inv_e/max(tot_e,1)):.2%}", help=f"{inv_e:,}/{tot_e:,} tokens in-vocab")

    # Show most down-weighted (frequent) tokens under SIF for transparency
    if use_sif:
        with st.expander("Most frequent tokens in TRAIN (lowest SIF weights)"):
            top_n = int(st.slider("Show top N frequent tokens", 10, 100, 30, 5))
            items = sorted(train_token_freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            rows = [{"Token": t, "Freq(TRAIN)": c, "SIF weight": sif_weight(t)} for t, c in items]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Encode labels
    label_map = {lbl: idx for idx, lbl in enumerate(sorted(work["Emotion"].unique()))}
    y_labels = work["Emotion"].map(label_map).values

    st.write(f"Embeddings shape: **{X_emb.shape}**   |   Labels map: {label_map}")

    # ---------- Persist to session ----------
    st.session_state["w2v_model"] = w2v
    st.session_state["X_emb"] = X_emb
    st.session_state["y_labels"] = y_labels
    st.session_state["label_map"] = label_map
    st.session_state["train_index"] = trn.index.to_numpy()
    st.session_state["test_index"]  = tst.index.to_numpy()
    st.session_state["embedding_used"] = f"Word2Vec({'SIF' if use_sif else 'mean'}{'-PCrm' if (use_sif and remove_pc) else ''}{'-L2' if do_l2 else ''})"

    st.success("Per-review embeddings are ready ✅ — proceed to **Modeling & Results**.")
    st.caption("Note: Multi-threaded W2V can have tiny non-determinism across runs; a fixed seed reduces this.")



def page_modeling_and_results():
    """
    One-page: train Random Forest & XGBoost on Word2Vec embeddings and show results.
    Pipeline:
      - Uses saved TRAIN/TEST indices from the Word2Vec page (prevents leakage).
      - Runs Stratified K-Fold CV on TRAIN ONLY (orthodox).
      - Optional imbalance handling: Class weights (RF & XGB via sample weights) or SMOTE (TRAIN-only).
      - Reports macro metrics for CV and held-out TEST; shows CMs, ROC curves, importances.
      - Persists models + results bundle in st.session_state.
    """

    st.title("Modeling & Results — Random Forest vs XGBoost")
    st.caption("Stratified CV on TRAIN only + final held-out TEST evaluation. Macro metrics throughout.")

    # --------- Pull data from session ---------
    X = st.session_state.get("X_emb")
    y = st.session_state.get("y_labels")
    label_map = st.session_state.get("label_map")
    tr_idx = st.session_state.get("train_index")
    te_idx = st.session_state.get("test_index")
    emb_used = st.session_state.get("embedding_used", "Word2Vec(mean)")

    if any(v is None for v in [X, y, label_map, tr_idx, te_idx]):
        st.error("Embeddings or indices missing. Please run **Embeddings (Word2Vec)** first.")
        st.stop()

    classes_sorted = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    labels_indices = np.array([label_map[c] for c in classes_sorted])
    n_classes = len(classes_sorted)
    st.write(f"Features: **{X.shape}**  |  Classes: **{classes_sorted}**")
    st.caption(f"Embedding used: **{emb_used}**")

    # --------- TRAIN / TEST split (fixed from Word2Vec page) ---------
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    st.caption(f"Train counts: {dict(Counter(y_tr))}  |  Test counts: {dict(Counter(y_te))}")

    # --------- Controls ---------
    with st.expander("🎛 Training Controls", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_splits = int(st.number_input("CV folds (StratifiedKFold on TRAIN)", 3, 10, 5, 1))
            random_state = int(st.number_input("random_state", 0, 9999, 42, 1))
        with c2:
            imb = st.selectbox("Imbalance handling", ["Class weights", "SMOTE", "None"],
                               help="SMOTE and class weights applied only on TRAIN folds and final TRAIN fit.")
        with c3:
            # RF params
            rf_n_estimators    = int(st.number_input("RF n_estimators", 100, 2000, 400, 50))
            rf_max_depth       = int(st.number_input("RF max_depth (0=None)", 0, 200, 0, 1))
            rf_min_samples_leaf= int(st.number_input("RF min_samples_leaf", 1, 20, 1, 1))

    with st.expander("⚡ XGBoost Hyper-parameters", expanded=True):
        if not XGB_OK:
            st.warning("XGBoost not installed — only Random Forest will run.")
        xgb_n_estimators = int(st.number_input("xgb n_estimators", 100, 2000, 600, 50))
        xgb_lr           = float(st.number_input("learning_rate", 0.01, 0.5, 0.1, 0.01))
        xgb_max_depth    = int(st.number_input("max_depth", 2, 20, 6, 1))
        xgb_subsample    = float(st.slider("subsample", 0.5, 1.0, 0.8, 0.05))
        xgb_colsample    = float(st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05))
        xgb_reg_lambda   = float(st.number_input("lambda (L2)", 0.0, 10.0, 1.0, 0.1))
        use_early_stop   = st.checkbox("Use early stopping (uses 15% of TRAIN as val)", value=True)

    # helper: sample weights for class weighting
    def sample_weights(y_vec):
        classes, counts = np.unique(y_vec, return_counts=True)
        total = len(y_vec)
        w = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        return np.array([w[yi] for yi in y_vec])

    # --------- Train button ---------
    if st.button(" Train & Evaluate ", type="primary"):
        # Optionally rebalance TRAIN
        rf_class_weight = None
        sw_tr = None
        Xtr_fit, ytr_fit = X_tr, y_tr  # local copies for final fit

        if imb == "Class weights":
            rf_class_weight = "balanced"
            sw_tr = sample_weights(y_tr)
        elif imb == "SMOTE":
            if SMOTE_OK:
                sm = SMOTE(random_state=random_state)
                Xtr_fit, ytr_fit = sm.fit_resample(X_tr, y_tr)
                st.info(f"SMOTE applied on TRAIN for final fit: {X_tr.shape} → {Xtr_fit.shape}")
            else:
                st.warning("SMOTE unavailable (install `imbalanced-learn`). Continuing without SMOTE.")

        # --------- CV on TRAIN only ---------
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        def cv_eval(clf, uses_sample_weight=False):
            precs, recs, f1s, aucs = [], [], [], []
            for tr_i, va_i in skf.split(X_tr, y_tr):
                XtrF, XvaF = X_tr[tr_i], X_tr[va_i]
                ytrF, yvaF = y_tr[tr_i], y_tr[va_i]

                # SMOTE on fold-train only
                if imb == "SMOTE" and SMOTE_OK:
                    sm = SMOTE(random_state=random_state)
                    XtrF, ytrF = sm.fit_resample(XtrF, ytrF)

                swF = sample_weights(ytrF) if (imb == "Class weights" and uses_sample_weight) else None
                clf.fit(XtrF, ytrF, sample_weight=swF) if uses_sample_weight else clf.fit(XtrF, ytrF)

                proba = np.clip(clf.predict_proba(XvaF), 1e-8, 1 - 1e-8)
                yva_bin = label_binarize(yvaF, classes=np.arange(n_classes))
                try:
                    auc = roc_auc_score(yva_bin, proba, average="macro", multi_class="ovr")
                except Exception:
                    auc = np.nan
                preds = clf.predict(XvaF)

                precs.append(precision_score(yvaF, preds, average="macro", zero_division=0))
                recs.append(recall_score(yvaF, preds, average="macro", zero_division=0))
                f1s.append(f1_score(yvaF, preds, average="macro", zero_division=0))
                aucs.append(auc)
            return float(np.nanmean(precs)), float(np.nanmean(recs)), float(np.nanmean(f1s)), float(np.nanmean(aucs))

        metrics_rows = []
        conf_matrices = {}
        roc_curves = {}
        feature_importance = {}

        # ----- Random Forest -----
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
        rf_cvP, rf_cvR, rf_cvF1, rf_cvAUC = cv_eval(rf, uses_sample_weight=False)

        # Fit on (optionally rebalanced) TRAIN and eval on TEST
        rf.fit(Xtr_fit, ytr_fit)
        rf_preds = rf.predict(X_te)
        rf_prob = np.clip(rf.predict_proba(X_te), 1e-8, 1 - 1e-8)
        rf_acc = accuracy_score(y_te, rf_preds)
        y_te_bin = label_binarize(y_te, classes=np.arange(n_classes))
        try:
            rf_auc_macro = roc_auc_score(y_te_bin, rf_prob, average="macro", multi_class="ovr")
            fpr_rf, tpr_rf, _ = roc_curve(y_te_bin.ravel(), rf_prob.ravel())
        except Exception:
            rf_auc_macro = np.nan; fpr_rf, tpr_rf = np.array([0,1]), np.array([0,1])

        metrics_rows.append({
            "Model": "Random Forest",
            "CV_Precision": rf_cvP, "CV_Recall": rf_cvR, "CV_F1": rf_cvF1, "CV_ROC-AUC": rf_cvAUC,
            "Test_Precision": precision_score(y_te, rf_preds, average="macro", zero_division=0),
            "Test_Recall": recall_score(y_te, rf_preds, average="macro", zero_division=0),
            "Test_F1": f1_score(y_te, rf_preds, average="macro", zero_division=0),
            "Test_ROC-AUC": rf_auc_macro,
            "Test_Accuracy": rf_acc
        })
        conf_matrices["Random Forest"] = confusion_matrix(y_te, rf_preds, labels=np.arange(n_classes))
        roc_curves["Random Forest"] = (fpr_rf, tpr_rf, rf_auc_macro)
        if hasattr(rf, "feature_importances_"):
            feature_importance["Random Forest"] = {
                "features": [f"dim_{i}" for i in range(X.shape[1])],
                "importance": rf.feature_importances_.tolist()
            }

        # ----- XGBoost -----
        if XGB_OK:
            xgb = XGBClassifier(
                n_estimators=xgb_n_estimators,
                learning_rate=xgb_lr,
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
            xgb_cvP, xgb_cvR, xgb_cvF1, xgb_cvAUC = cv_eval(xgb, uses_sample_weight=(imb == "Class weights"))

            # early stopping using a split from TRAIN (never touch TEST)
            if use_early_stop:
                Xtr2, Xva2, ytr2, yva2 = train_test_split(
                    Xtr_fit, ytr_fit, test_size=0.15, stratify=ytr_fit, random_state=random_state
                )
                sw2 = sample_weights(ytr2) if (imb == "Class weights") else None
                xgb.fit(
                    Xtr2, ytr2,
                    sample_weight=sw2,
                    eval_set=[(Xva2, yva2)],
                    eval_metric="mlogloss",
                    verbose=False,
                    early_stopping_rounds=30
                )
            else:
                sw2 = sample_weights(ytr_fit) if (imb == "Class weights") else None
                xgb.fit(Xtr_fit, ytr_fit, sample_weight=sw2)

            xgb_preds = xgb.predict(X_te)
            xgb_prob = np.clip(xgb.predict_proba(X_te), 1e-8, 1 - 1e-8)
            xgb_acc = accuracy_score(y_te, xgb_preds)
            try:
                xgb_auc_macro = roc_auc_score(y_te_bin, xgb_prob, average="macro", multi_class="ovr")
                fpr_xgb, tpr_xgb, _ = roc_curve(y_te_bin.ravel(), xgb_prob.ravel())
            except Exception:
                xgb_auc_macro = np.nan; fpr_xgb, tpr_xgb = np.array([0,1]), np.array([0,1])

            metrics_rows.append({
                "Model": "XGBoost",
                "CV_Precision": xgb_cvP, "CV_Recall": xgb_cvR, "CV_F1": xgb_cvF1, "CV_ROC-AUC": xgb_cvAUC,
                "Test_Precision": precision_score(y_te, xgb_preds, average="macro", zero_division=0),
                "Test_Recall": recall_score(y_te, xgb_preds, average="macro", zero_division=0),
                "Test_F1": f1_score(y_te, xgb_preds, average="macro", zero_division=0),
                "Test_ROC-AUC": xgb_auc_macro,
                "Test_Accuracy": xgb_acc
            })
            conf_matrices["XGBoost"] = confusion_matrix(y_te, xgb_preds, labels=np.arange(n_classes))
            roc_curves["XGBoost"] = (fpr_xgb, tpr_xgb, xgb_auc_macro)

            if hasattr(xgb, "feature_importances_"):
                feature_importance["XGBoost"] = {
                    "features": [f"dim_{i}" for i in range(X.shape[1])],
                    "importance": xgb.feature_importances_.tolist()
                }
            st.session_state["xgb_model"] = xgb
        else:
            st.info("XGBoost not available; skipping XGB metrics.")

        # Persist models & results
        st.session_state["rf_model"] = rf
        results_bundle = {
            "metrics": metrics_rows,
            "conf_matrices": conf_matrices,
            "labels": classes_sorted,
            "roc_curves": roc_curves,
            "feature_importance": feature_importance,
            "embedding_used": emb_used
        }
        st.session_state["results"] = results_bundle

        # --------- Presentation ---------
        res_df = pd.DataFrame(metrics_rows).set_index("Model")
        st.subheader("Results (CV on TRAIN vs Held-out TEST)")
        st.dataframe(res_df.style.format("{:.4f}"), use_container_width=True)

        # Winner by Test_F1 then Test_ROC-AUC
        winner = max(res_df.index, key=lambda m: (res_df.loc[m, "Test_F1"], res_df.loc[m, "Test_ROC-AUC"]))
        st.success(f"🏆 Overall winner (TEST): **{winner}**")

        # Download
        st.download_button(
            "Download metrics (CSV)",
            data=res_df.to_csv().encode("utf-8"),
            file_name="model_metrics.csv",
            mime="text/csv"
        )

        # Confusion matrices
        st.subheader("Confusion Matrices — Held-out TEST")
        n = len(conf_matrices)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 4))
        if n == 1: axes = np.array([axes])
        for ax, (mname, cm) in zip(axes, conf_matrices.items()):
            sns.heatmap(np.asarray(cm), annot=True, fmt="d",
                        cmap="Blues" if "Forest" in mname else "Greens",
                        ax=ax, xticklabels=classes_sorted, yticklabels=classes_sorted)
            ax.set_title(mname); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)

        # ROC curves
        st.subheader("ROC Curves (macro AUC label; micro-style curve)")
        fig2, ax2 = plt.subplots()
        for mname, (fpr, tpr, auc_macro) in roc_curves.items():
            fpr = np.asarray(fpr); tpr = np.asarray(tpr)
            ax2.plot(fpr, tpr, label=f"{mname} (AUC={auc_macro:.3f})" if np.isfinite(auc_macro) else f"{mname} (AUC=n/a)")
        ax2.plot([0,1],[0,1],"k--", label="Random guess")
        ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate"); ax2.legend()
        st.pyplot(fig2)

        # Feature importances (top embedding dims)
        st.subheader("Feature Importance (Top Embedding Dimensions)")
        def plot_top_importances(model_name: str, fi_dict: dict, top_k: int = 15):
            if not fi_dict or "importance" not in fi_dict:
                st.info(f"No importances for {model_name}."); return
            imp = np.asarray(fi_dict["importance"])
            dims = np.array(fi_dict.get("features", [f"dim_{i}" for i in range(len(imp))]))
            order = np.argsort(imp)[::-1][:top_k]
            top_df = pd.DataFrame({"Feature": dims[order], "Importance": imp[order]})
            st.markdown(f"**{model_name} — Top {top_k} dims**")
            st.dataframe(top_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.barplot(data=top_df, y="Feature", x="Importance", ax=ax)
            ax.set_title(f"{model_name}: Top {top_k} Importances")
            st.pyplot(fig)

        cols = st.columns(2 if "XGBoost" in feature_importance else 1)
        with cols[0]:
            if "Random Forest" in feature_importance: plot_top_importances("Random Forest", feature_importance["Random Forest"])
        if "XGBoost" in feature_importance and len(cols) > 1:
            with cols[1]:
                plot_top_importances("XGBoost", feature_importance["XGBoost"])

        st.info("Done. Results are cached in session for the Predictions page.")



def page_wordclouds():
    """
    Emotion-Specific Word Clouds (Word2Vec-centric)
    ------------------------------------------------
    Methods:
      • Word2Vec centroid (semantic) — cosine similarity to per-emotion centroid
      • Word2Vec centroid (SIF) — SIF-weighted centroid using TRAIN frequencies
      • Contrastive log-odds — purely frequency-based baseline (no TF-IDF)

    Visual filters (display-only): token sanity, global doc freq floor, per-class min freq,
    domain/custom stoplists, optional VADER polarity gating.

    Notes:
    - When 'Use TRAIN only' is ON, all counts/centroids use the TRAIN split from session to avoid leakage.
    - Word clouds are deterministic via user-set seed.
    """

    st.title("Emotion-Specific Word Clouds")
    st.caption("Word2Vec-centric clouds with leakage-safe option and deterministic layout.")

    # ---- Guards
    df_clean = st.session_state.get("df_clean")
    if df_clean is None or "clean_text" not in df_clean.columns or "Emotion" not in df_clean.columns:
        st.error("No cleaned dataset found. Run **Preprocess & Labels** first.")
        st.stop()

    EMOTIONS = [e for e in ["Negative", "Neutral", "Positive"] if e in df_clean["Emotion"].unique()]
    if not EMOTIONS:
        st.error("No emotion labels available.")
        st.stop()

    w2v_model = st.session_state.get("w2v_model")  # required for centroid methods
    WV_KEYS = set(w2v_model.wv.key_to_index) if w2v_model is not None else set()

    EMO2CMAP = {"Negative": "Reds", "Neutral": "Blues", "Positive": "Greens"}

    # ---- Controls
    c_top = st.columns(2)
    with c_top[0]:
        method = st.radio(
            "Weighting method",
            ["Word2Vec centroid (semantic)", "Word2Vec centroid (SIF)", "Contrastive log-odds (counts)"],
            index=0,
        )
    with c_top[1]:
        use_train_only_for_counts = st.checkbox("Use TRAIN only for counts/centroids (avoid leakage)", value=True)

    # Source df for counting/centroids
    tr_idx = st.session_state.get("train_index")
    df_counts_source = (
        df_clean.loc[tr_idx].reset_index(drop=True)
        if (use_train_only_for_counts and tr_idx is not None)
        else df_clean
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        top_n = st.slider("Top-N table / CSV", 20, 200, 60, 5)
        cloud_words_cap = st.slider("Words used in each cloud", 60, 300, 180, 20)
    with c2:
        background = st.selectbox("Background", ["white", "black"], index=0)
        seed = st.number_input("Word cloud seed (deterministic)", 0, 9999, 42, 1)
    with c3:
        max_vocab_for_similarity = st.slider("Max vocab for similarity (per emotion)", 500, 20000, 5000, 500)

    limit_to_emotion_vocab = st.checkbox("Use only tokens appearing in that emotion’s reviews", value=True)
    min_freq_per_emotion = st.number_input("Min frequency in emotion", 1, 100, 10, 1)

    global_min_df = st.number_input(
        "Global document frequency (min #reviews containing token)",
        1, 1000, 20, 1,
        help="Helps remove rare/garbled tokens (especially Neutral).",
    )

    st.markdown("#### Hide brand/product/common words (visual-only)")
    default_stoplist = "amazon, starbucks, folgers, keurig, nespresso, kitkat, trader, joes, walmart, costco"
    custom_stop = st.text_input("Comma-separated words to exclude", value=default_stoplist)
    HIDE = set(w.strip().lower() for w in custom_stop.split(",") if w.strip())

    with st.expander("Optional: Polarity gate (VADER) for Positive/Negative/Neutral", expanded=False):
        use_vader = st.checkbox("Gate by VADER polarity", value=False)
        pos_thresh = st.slider("Positive threshold (>=)", 0.1, 2.0, 0.5, 0.1)
        neg_thresh = st.slider("Negative threshold (<=)", -2.0, -0.1, -0.5, 0.1)
        neu_band = st.slider("Neutral band (|valence| ≤)", 0.1, 1.0, 0.2, 0.1)

    use_domain_stop = st.checkbox("Apply domain stop-list", value=True)
    DOMAIN_STOP = {
        "like", "taste", "product", "one", "would", "good", "great", "get", "make", "really", "time", "much", "also",
        "food", "coffee", "tea", "amazon", "buy", "use", "used", "got", "well", "bit", "little", "thing", "things", "even"
    }
    DOMAIN_STOP_TUPLE = tuple(sorted(DOMAIN_STOP))  # cache-friendly

    TOKEN_RE = re.compile(r"^[a-z]+$")

    def token_ok(w: str) -> bool:
        return 3 <= len(w) <= 15 and TOKEN_RE.match(w) is not None and (w not in DOMAIN_STOP if use_domain_stop else True)

    # ---- Cached global document frequency (dfreq) from the chosen source
    @st.cache_data(show_spinner=False)
    def build_global_docfreq_from_texts(texts: list[str], use_domain_stop: bool, domain_stop_tuple: tuple) -> dict:
        token_re = re.compile(r"^[a-z]+$")
        dfreq = Counter()
        stopset = set(domain_stop_tuple) if use_domain_stop else set()
        for text in texts:
            toks = set(t for t in text.split() if token_re.match(t) and 3 <= len(t) <= 15 and (t not in stopset))
            dfreq.update(toks)
        return dict(dfreq)

    GLOBAL_DF = build_global_docfreq_from_texts(
        df_counts_source["clean_text"].astype(str).tolist(), use_domain_stop, DOMAIN_STOP_TUPLE
    )

    # ---- Optional VADER polarity sets
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

    # ---- Per-emotion filtered counts (respect token_ok + global_min_df)
    def get_emotion_tokens(emo: str) -> list[str]:
        docs = df_counts_source.loc[df_counts_source["Emotion"] == emo, "clean_text"].astype(str)
        return [t for doc in docs for t in doc.split()]

    emo_counts: dict[str, Counter] = {}
    for emo in EMOTIONS:
        toks = [t for t in get_emotion_tokens(emo) if token_ok(t) and GLOBAL_DF.get(t, 0) >= global_min_df]
        emo_counts[emo] = Counter(toks)

    # ---- Helpers
    def render_cloud(freqs: dict[str, float], title: str, cmap: str):
        freqs = {w: v for w, v in freqs.items() if w.lower() not in HIDE and v > 0}
        if not freqs:
            st.info(f"No terms to display for **{title}** after filtering.")
            return
        wc = WordCloud(
            width=1000, height=500, background_color=background,
            collocations=False, colormap=cmap, random_state=int(seed)   # deterministic placement
        ).generate_from_frequencies(freqs)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        st.download_button(
            label=f"Download '{title}' PNG",
            data=buf.getvalue(),
            file_name=f"{title.replace(' ', '_').lower()}.png",
            mime="image/png",
        )

    def shift_to_nonnegative(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
        if not items:
            return items
        m = min(v for _, v in items)
        return items if m >= 0 else [(w, v - m) for w, v in items]

    def top_table(freqs: dict[str, float], k: int) -> pd.DataFrame:
        top_items = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return pd.DataFrame(top_items, columns=["Word", "Weight"])

    # Vectorized cosine similarity
    def top_by_centroid_similarity(centroid_vec: np.ndarray, candidate_words: list[str], k: int):
        if not candidate_words:
            return []
        mat = w2v_model.wv[candidate_words]  # shape (n, d)
        mat_norm = np.linalg.norm(mat, axis=1) + 1e-12
        c = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-12)
        sims = (mat @ c) / mat_norm
        order = np.argsort(-sims)[:k]
        return [(candidate_words[i], float(sims[i])) for i in order]

    # ---- Compute per-emotion weights by method
    weights_per_emotion: dict[str, dict[str, float]] = {}

    # A) Word2Vec centroid (semantic)
    if method.startswith("Word2Vec centroid (semantic)"):
        if w2v_model is None:
            st.error("Word2Vec model not found. Choose another method or train embeddings first.")
            st.stop()

        st.info("Cosine similarity to each emotion’s unweighted Word2Vec centroid (semantic).")
        # Weighted mean by per-emotion token counts (efficient & equivalent to repetition)
        centroids = {}
        for emo in EMOTIONS:
            toks_counts = [(w, c) for w, c in emo_counts[emo].items() if w in WV_KEYS]
            if not toks_counts:
                centroids[emo] = None
                continue
            mat = np.vstack([w2v_model.wv[w] for w, _ in toks_counts]).astype("float32")
            ws = np.array([c for _, c in toks_counts], dtype="float32").reshape(-1, 1)
            centroids[emo] = (mat * ws).sum(axis=0) / (ws.sum() + 1e-12)

        for emo in EMOTIONS:
            c = centroids[emo]
            if c is None:
                weights_per_emotion[emo] = {}
                continue
            if limit_to_emotion_vocab:
                allow = [w for w, cnt in emo_counts[emo].items() if cnt >= min_freq_per_emotion and w in WV_KEYS and token_ok(w)]
            else:
                allow = [w for w in w2v_model.wv.key_to_index if token_ok(w)]
            allow = allow[:max_vocab_for_similarity]
            top_items = top_by_centroid_similarity(c, allow, cloud_words_cap)
            top_items = shift_to_nonnegative(top_items)
            weights_per_emotion[emo] = dict(top_items)

    # B) Word2Vec centroid (SIF)
    elif method.startswith("Word2Vec centroid (SIF)"):
        if w2v_model is None:
            st.error("Word2Vec model not found. Train embeddings first.")
            st.stop()

        st.info("Cosine similarity to **SIF-weighted** centroids (Word2Vec-only).")
        # Build SIF from TRAIN if available; else from df_counts_source
        a = 1e-3
        if use_train_only_for_counts and tr_idx is not None:
            train_docs = df_clean.loc[tr_idx, "clean_text"].astype(str).tolist()
        else:
            train_docs = df_counts_source["clean_text"].astype(str).tolist()

        train_freq = Counter(t for d in train_docs for t in d.split())
        total_train_tokens = sum(train_freq.values())

        def sif_w(tok: str) -> float:
            return a / (a + train_freq.get(tok, 0) / max(total_train_tokens, 1))

        def sif_centroid(counter: Counter) -> np.ndarray | None:
            toks = [(w, c) for w, c in counter.items() if w in WV_KEYS]
            if not toks:
                return None
            mat = np.vstack([w2v_model.wv[w] for w, _ in toks]).astype("float32")
            ws = np.array([sif_w(w) * c for w, c in toks], dtype="float32").reshape(-1, 1)
            return (mat * ws).sum(axis=0) / (ws.sum() + 1e-12)

        centroids = {emo: sif_centroid(emo_counts[emo]) for emo in EMOTIONS}

        for emo in EMOTIONS:
            c = centroids[emo]
            if c is None:
                weights_per_emotion[emo] = {}
                continue
            if limit_to_emotion_vocab:
                allow = [w for w, cnt in emo_counts[emo].items() if cnt >= min_freq_per_emotion and w in WV_KEYS and token_ok(w)]
            else:
                allow = [w for w in w2v_model.wv.key_to_index if token_ok(w)]
            allow = allow[:max_vocab_for_similarity]
            top_items = top_by_centroid_similarity(c, allow, cloud_words_cap)
            top_items = shift_to_nonnegative(top_items)
            weights_per_emotion[emo] = dict(top_items)

    # C) Contrastive log-odds (counts)
    else:
        st.info("Contrastive log-odds with +1 smoothing (discriminative vs other emotions).")
        global_counts = Counter()
        for e in EMOTIONS:
            global_counts.update(emo_counts[e])

        for emo in EMOTIONS:
            in_counts = emo_counts[emo]
            out_counts = global_counts.copy()
            for w, c in in_counts.items():
                out_counts[w] -= c

            in_total = sum(in_counts.values())
            out_total = sum(out_counts.values())

            allowed = set(in_counts.keys()) if limit_to_emotion_vocab else set(global_counts.keys())
            allowed = {w for w in allowed if in_counts.get(w, 0) >= min_freq_per_emotion}

            scores = {}
            for w in allowed:
                a_ = in_counts[w] + 1
                b_ = (in_total - in_counts[w]) + 1
                c_ = out_counts[w] + 1
                d_ = (out_total - out_counts[w]) + 1
                scores[w] = float(np.log(a_ / b_) - np.log(c_ / d_))

            # Optional polarity gating
            if use_vader:
                if emo == "Positive" and POS_SET:
                    scores = {w: s for w, s in scores.items() if w in POS_SET}
                elif emo == "Negative" and NEG_SET:
                    scores = {w: s for w, s in scores.items() if w in NEG_SET}
                elif emo == "Neutral" and NEU_SET is not None:
                    scores = {w: s for w, s in scores.items() if w in NEU_SET}

            top_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:cloud_words_cap]
            top_items = shift_to_nonnegative(top_items)
            weights_per_emotion[emo] = dict(top_items)

    # ---- Render per emotion + tables
    st.markdown("---")
    cols = st.columns(len(EMOTIONS))
    combined_tables = []
    for col, emo in zip(cols, EMOTIONS):
        with col:
            title = f"{emo} — {method}"
            freqs = weights_per_emotion.get(emo, {})
            render_cloud(freqs, title, EMO2CMAP.get(emo, "viridis"))
            top_df = top_table(freqs, top_n)
            st.dataframe(top_df, use_container_width=True)
            # add emotion column for combined export
            if not top_df.empty:
                tdf = top_df.copy()
                tdf.insert(0, "Emotion", emo)
                combined_tables.append(tdf)

    # Combined CSV download (all emotions)
    if combined_tables:
        all_df = pd.concat(combined_tables, ignore_index=True)
        st.download_button(
            "Download ALL emotions Top-N (CSV)",
            data=all_df.to_csv(index=False).encode("utf-8"),
            file_name=f"wordcloud_tables_{method.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

    st.markdown("### Quick Interpretation")
    st.write(
        "- **Positive**: praise/adjectives (e.g., *delicious, amazing, wonderful*).  \n"
        "- **Negative**: complaint/defect (e.g., *awful, rancid, disappointing*).  \n"
        "- **Neutral**: transactional/description (*package, shipped, ingredients*)."
    )

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
        "Negative": r"Negative.jpg",
        "Neutral":  r"Neutral.jpg",
        "Positive": r"Positive.jpg",
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





#  ROUTER
PAGES = {
    "HomePage": page_home,
    "Data Load": page_data_load,
    "Preprocess & Labels": page_preprocess,
    "Post‑Cleaning Diagnostics": page_diagnostics,
    "Embeddings (Word2Vec)": page_word2vec,
    "Modeling and Results (RF & XGBoost)": page_modeling_and_results,
    "Word Clouds": page_wordclouds,
    "Prediction Page":page_predictions,
}

# Sidebar logo (optional; wrap in try so it doesn't crash if missing)
try:
    st.sidebar.image(
        "C:/Users/Francesca Manu/PycharmProjects/Text_Group_Project/Sidebar2.png",
        use_column_width=False,
        width=300,
    )
except Exception:
    pass

# Sidebar navigation
choice = st.sidebar.selectbox("Please Select Page", list(PAGES.keys()))
PAGES[choice]()
