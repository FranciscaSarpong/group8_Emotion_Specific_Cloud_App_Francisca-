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
    roc_curve, confusion_matrix, balanced_accuracy_score  # ADD THIS HERE
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

# ADD PLOTLY IMPORT FOR INTERACTIVE VISUALIZATIONS
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Streamlit safety
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass


class EnsembleClassifier:
    def __init__(self, rf_model, xgb_model, weights=None):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.weights = weights or [0.5, 0.5]

    def predict_proba(self, X):
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        return self.weights[0] * rf_proba + self.weights[1] * xgb_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


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
    """Read from a local absolute path (csv/parquet) with memory optimization"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    p = path.lower()

    # Memory-efficient loading for large files
    if p.endswith(".csv"):
        file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB

        if file_size > 100:  # If file > 100MB, use chunked loading
            st.info(f"Large file detected ({file_size:.1f}MB). Using chunked loading...")

            chunk_list = []
            chunk_size = 10000

            for chunk in pd.read_csv(path, usecols=usecols, nrows=nrows,
                                     encoding="utf-8", chunksize=chunk_size):
                chunk_list.append(safe_coerce_types(chunk))

                # Progress indicator for large files
                if len(chunk_list) % 10 == 0:
                    st.write(f"Processed {len(chunk_list) * chunk_size:,} rows...")

            df = pd.concat(chunk_list, ignore_index=True)
        else:
            df = pd.read_csv(path, usecols=usecols, nrows=nrows, encoding="utf-8")
    else:  # parquet
        df = pd.read_parquet(path, columns=usecols if usecols else None)

    return safe_coerce_types(df).reset_index(drop=True)


def comprehensive_evaluation_metrics(y_true, y_pred, y_proba, label_names):
    """Enhanced evaluation with confidence intervals and additional metrics"""
    from sklearn.metrics import classification_report, cohen_kappa_score

    # Basic metrics
    metrics = {
        'classification_report': classification_report(y_true, y_pred, target_names=label_names, output_dict=True),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }

    # Per-class confidence intervals
    try:
        from scipy import stats
        for i, class_name in enumerate(label_names):
            class_mask = (y_true == i)
            if class_mask.sum() > 0:
                class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
                n = class_mask.sum()
                if n > 0:
                    ci_low, ci_high = stats.proportion_confint(class_acc * n, n, alpha=0.05)
                    metrics[f'{class_name}_accuracy_ci'] = (ci_low, ci_high)
    except ImportError:
        st.info("Install scipy for confidence intervals: pip install scipy")

    # Prediction confidence statistics
    max_probas = np.max(y_proba, axis=1)
    metrics['prediction_confidence'] = {
        'mean': np.mean(max_probas),
        'std': np.std(max_probas),
        'min': np.min(max_probas),
        'max': np.max(max_probas)
    }

    return metrics


# PAGES

def page_home():
    st.title(" Emotion-Specific Word Cloud from Amazon Reviews (Word2Vec)")
    st.caption(PAGE_HELP.get("Home", ""))

    # Create columns to position logo on the right
    col1, col2 = st.columns([2, 1])

    with col2:
        # Home page logo (use a relative path or put image in /assets)
        try:
            st.image(
                "Group82.png",
                width=300,
                caption="Emotion-Specific Word Cloud from Amazon Reviews"
            )
        except Exception:
            st.info("Logo image not found. Place 'Group82.png' in the same directory.")

    with col1:
        st.markdown("""
        ### **Goal**
        Build a pipeline that analyzes Amazon Fine Food Reviews, detects **emotions** (Positive / Neutral / Negative),
        and visualizes vocabulary per emotion via **word clouds**, using **Word2Vec** + **Random Forest vs XGBoost**.

        ### **Navigation**
        Use the sidebar to move step-by-step: Data Load → Preprocessing → Embeddings → Modeling → Evaluation → Word Clouds → Prediction.
        """)

    # Team
    st.markdown("---")
    st.markdown("###  Project Team")
    team = [
        ("George Owell", "22256146", "http://emotionbasedwordcloud-kaqsykwxudaljgan3srpws.streamlit.app"),
        ("Francisca Manu Sarpong", "22255796", "https://emotio.streamlit.app/"),
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
        for _, _, role in team: st.markdown(role if role else "Yet to Deploy...")

    st.info("🚀 Start with **Data Load** in the sidebar.")


def page_data_load():
    st.title(" Data Load")
    st.caption(PAGE_HELP["Data Load"])
    st.markdown("Load from **local path** or **file uploader**. We'll cache the DataFrame and show quick EDA.")

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
    st.markdown("####  Load from local path")
    local_path = st.text_input("Absolute path (.csv or .parquet)",
                               value="Reviewsample.csv")
    btn_path = st.button(" Load from path")

    # --- Uploader loader ---
    st.markdown("####  Or upload a file")
    up = st.file_uploader("Upload file", type=["csv", "parquet"])
    btn_up = st.button(" Load from upload")

    df = None
    try:
        if btn_path:
            if not local_path.strip():
                st.error("Please provide a valid file path.")
                st.stop()

            df = load_reviews_from_path(
                local_path.strip(),
                usecols=selected_cols if selected_cols else None,
                nrows=int(nrows) if (first_n and local_path.lower().endswith(".csv")) else None
            )
        elif btn_up:
            if up is None:
                st.error("Please upload a CSV or Parquet file.")
                st.stop()
            df = load_reviews_uploaded(
                up,
                usecols=selected_cols if selected_cols else None,
                nrows=int(nrows) if (first_n and up.name.lower().endswith(".csv")) else None
            )
        else:
            st.stop()  # wait for a button press
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while loading data: {e}")
        st.info(" **Troubleshooting tips:**")
        st.write("- Check if the file path is correct")
        st.write("- Ensure the file is not corrupted")
        st.write("- Try with a smaller number of rows first")
        st.stop()

    if df is None or df.empty:
        st.error("Loaded an empty DataFrame. Check the path/file and selected columns.")
        st.stop()

    # Cache to session for later pages
    st.session_state["df"] = df
    st.success(f"✅ Loaded: **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")

    # --- Quick EDA ---
    st.subheader(" Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader(" Column Types & Missingness")
    info = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str),
        "Non-Null": df.notnull().sum(),
        "Nulls": df.isnull().sum(),
        "Null %": (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(info, use_container_width=True)

    st.subheader(" Duplicates")
    duplicate_count = df.duplicated().sum()
    st.write(f"Exact duplicate rows: **{duplicate_count:,}**")
    if duplicate_count > 0:
        st.warning(f"Found {duplicate_count:,} duplicate rows. Consider removing them in preprocessing.")

    if "Score" in df.columns:
        st.subheader(" Score & Emotion Distribution")
        # Handle potential issues with Score column
        valid_scores = df["Score"].dropna()
        if len(valid_scores) > 0:
            try:
                st.bar_chart(df["Score"].value_counts(dropna=False).sort_index())
                # Safe conversion to emotion
                emo = valid_scores.astype(int).map(score_to_emotion)
                emo_counts = emo.value_counts()
                st.bar_chart(emo_counts)

                # Show emotion distribution stats
                st.write("**Emotion Distribution:**")
                for emotion, count in emo_counts.items():
                    percentage = (count / len(emo)) * 100
                    st.write(f"- {emotion}: {count:,} ({percentage:.1f}%)")
            except Exception as e:
                st.warning(f"Could not process Score column: {e}")
        else:
            st.warning("No valid scores found in the Score column.")

    if "Time" in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df["Time"]):
                tmin, tmax = df["Time"].min(), df["Time"].max()
                if pd.notnull(tmin) and pd.notnull(tmax):
                    st.subheader("📅 Time Coverage")
                    st.write(f"**Date range:** {tmin.date()} → {tmax.date()}")

                    # Show time distribution
                    time_counts = df["Time"].dt.year.value_counts().sort_index()
                    if len(time_counts) > 0:
                        st.write("**Reviews by year:**")
                        st.bar_chart(time_counts)
            else:
                st.info("Time column found but not in datetime format. Will be converted during preprocessing.")
        except Exception as e:
            st.warning(f"Could not process Time column: {e}")

    # Additional stats
    st.subheader(" Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", f"{len(df.columns)}")
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    with col4:
        completeness = (df.notnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")

    st.info("✅ Dataset cached. Next step: **Preprocess & Labels** page.")


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
    fallback_to_text = st.checkbox("Fallback to Text when Summary is missing/empty", value=True)

    # ---- Cleaning options ----
    st.markdown("#### Cleaning Options")
    c1, c2, c3 = st.columns(3)
    with c1:
        do_lower = st.checkbox("Lowercase", True)
        rm_html = st.checkbox("Remove HTML-like tags", True)
        rm_punct = st.checkbox("Remove punctuation (after expanding negations)", True)
    with c2:
        rm_digits = st.checkbox("Remove digits", True)
        collapse_ws = st.checkbox("Collapse extra spaces", True)
        drop_empty = st.checkbox("Drop empty/NA cleaned rows", True)
    with c3:
        use_nltk_stop = st.checkbox("Use NLTK stopwords", False)  # CHANGED: Default to False
        do_lemmatize = st.checkbox("Lemmatize (WordNet)", False)  # CHANGED: Default to False
        do_stem = st.checkbox("Stem (Porter) — not recommended for Word2Vec", False)

    # ADDED: Warning about aggressive preprocessing
    if use_nltk_stop:
        st.warning("⚠️ Using NLTK stopwords may remove important sentiment words like 'good', 'bad', 'not'")

    if do_lemmatize:
        st.info("ℹ️ Lemmatization can help, but may change sentiment words (e.g., 'better' → 'good')")

    # Guardrail: Word2Vec ≠ stemming
    if do_stem:
        st.warning("⚠️ Stemming may harm Word2Vec semantics; consider lemmatization only.")

    # Row limits & hygiene
    st.markdown("#### Processing Options")
    col_a, col_b = st.columns(2)
    with col_a:
        limit_rows = st.checkbox("Process only first N rows", True)
    with col_b:
        n_limit = st.number_input("N rows (if limited)", min_value=5_000, value=20_000, step=5_000)
    rm_dups = st.checkbox("Remove exact duplicate rows before cleaning", value=False)

    # ADDED: Emotion mapping configuration
    st.markdown("#### Emotion Mapping Configuration")
    st.info("Configure how star ratings (1-5) map to emotions")

    col_emo1, col_emo2, col_emo3 = st.columns(3)
    with col_emo1:
        neg_threshold = st.selectbox("Negative threshold (≤)", [1, 2], index=1)
        st.caption("Stars 1-2 → Negative")
    with col_emo2:
        neu_exact = st.selectbox("Neutral (exactly)", [3], index=0)
        st.caption("Stars 3 → Neutral")
    with col_emo3:
        pos_threshold = st.selectbox("Positive threshold (≥)", [4, 5], index=0)
        st.caption("Stars 4-5 → Positive")

    # ---- NLTK assets (lazy) ----
    @st.cache_resource(show_spinner=False)
    def _ensure_nltk_assets():
        import nltk
        try:
            nltk.data.find("corpora/stopwords")
        except:
            nltk.download("stopwords", quiet=True)
        try:
            nltk.data.find("corpora/wordnet")
            nltk.data.find("corpora/omw-1.4")
        except:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

    # FIXED: More conservative stopword list
    FALLBACK_STOPWORDS = set("""
    a an the and or but if while with without within into onto from to for of on in out by up down over under again further
    is are was were be been being do does did doing have has had having this that these those it its i me my we our you your
    he him his she her they them their what which who whom where when why how all any both each few more most other some such
    only own same so than too very can will just should now
    """.split())

    # CRITICAL: Keep important sentiment words and negators
    SENTIMENT_WORDS = {"good", "bad", "great", "terrible", "excellent", "awful", "amazing", "horrible",
                       "wonderful", "disappointing", "fantastic", "poor", "best", "worst", "love", "hate",
                       "delicious", "disgusting", "perfect", "horrible", "awesome", "terrible"}

    NEGATORS = {"not", "no", "never", "nor", "cannot", "can_not", "dont", "doesn", "didn", "won", "wouldn"}

    def get_stopwords():
        if use_nltk_stop:
            try:
                _ensure_nltk_assets()
                from nltk.corpus import stopwords as sw
                sw_set = set(sw.words("english"))
                # CRITICAL: Remove sentiment words and negators from stopwords
                sw_set = sw_set - SENTIMENT_WORDS - NEGATORS
                st.info(f"Removed {len(SENTIMENT_WORDS)} sentiment words and {len(NEGATORS)} negators from stopwords")
            except Exception:
                st.warning("NLTK stopwords unavailable; using fallback.")
                sw_set = set(FALLBACK_STOPWORDS)
        else:
            sw_set = set(FALLBACK_STOPWORDS)

        # Double-check: ensure critical words stay
        sw_set = sw_set - SENTIMENT_WORDS - NEGATORS
        return sw_set

    # --- small normalizer helpers ---
    CONTRACTIONS = [
        (r"won't", "will not"),
        (r"can't", "cannot"),  # CHANGED: Use "cannot" instead of "can not"
        (r"ain't", "is not"),
        (r"n['']t\b", " not"),
        (r"y['']all", "you all"),
        (r"gonna", "going to"),
        (r"wanna", "want to"),
        (r"don't", "do not"),  # ADDED: More contractions
        (r"didn't", "did not"),
        (r"doesn't", "does not"),
        (r"wouldn't", "would not"),
        (r"shouldn't", "should not"),
        (r"couldn't", "could not"),
    ]

    def expand_contractions(s: str) -> str:
        for pat, rep in CONTRACTIONS:
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        return s

    def preprocess_text(text: str, stop_set: set) -> str:
        """
        Normalize summary text with negation-aware cleaning.
        FIXED: More conservative preprocessing to preserve sentiment.
        """
        s = str(text)

        if do_lower:
            s = s.lower()

        # Expand contractions FIRST so we preserve negations
        s = expand_contractions(s)

        if rm_html:
            s = re.sub(r"<.*?>", " ", s)

        # MODIFIED: More conservative punctuation removal
        if rm_punct:
            # Keep apostrophes in contractions, remove other punctuation
            s = re.sub(r"[^\w\s']", " ", s)

        if rm_digits:
            s = re.sub(r"\d+", " ", s)

        if collapse_ws:
            s = re.sub(r"\s+", " ", s).strip()

        toks = s.split()
        if not toks:
            return ""

        # CRITICAL: More conservative filtering
        # Keep tokens that are:
        # 1. Longer than 1 character
        # 2. NOT in stopwords OR are sentiment words OR are negators
        toks = [t for t in toks if len(t) > 1 and
                (t not in stop_set or t in SENTIMENT_WORDS or t in NEGATORS)]

        # ADDED: Check if we're removing too many tokens
        original_count = len(s.split())
        final_count = len(toks)
        if original_count > 0 and final_count / original_count < 0.3:
            # If we're removing more than 70% of tokens, something might be wrong
            pass  # Could add warning here if needed

        if do_lemmatize or do_stem:
            try:
                _ensure_nltk_assets()
                if do_lemmatize:
                    from nltk.stem import WordNetLemmatizer
                    lem = WordNetLemmatizer()
                    # FIXED: Preserve sentiment words during lemmatization
                    new_toks = []
                    for t in toks:
                        if t in SENTIMENT_WORDS or t in NEGATORS:
                            new_toks.append(t)  # Don't lemmatize sentiment words
                        else:
                            new_toks.append(lem.lemmatize(t))
                    toks = new_toks
                if do_stem:
                    from nltk.stem import PorterStemmer
                    ps = PorterStemmer()
                    # FIXED: Preserve sentiment words during stemming
                    new_toks = []
                    for t in toks:
                        if t in SENTIMENT_WORDS or t in NEGATORS:
                            new_toks.append(t)  # Don't stem sentiment words
                        else:
                            new_toks.append(ps.stem(t))
                    toks = new_toks
            except Exception:
                st.warning("Lemmatizer/stemmer unavailable; skipping.")

        return " ".join(toks)

    # FIXED: Updated score_to_emotion function with configurable thresholds
    def score_to_emotion(score: int) -> str:
        """Map Amazon 1–5 stars → emotion label with configurable thresholds."""
        try:
            s = int(score)
        except (ValueError, TypeError):
            return "Unknown"

        if s <= neg_threshold:
            return "Negative"
        if s == neu_exact:
            return "Neutral"
        if s >= pos_threshold:
            return "Positive"
        return "Unknown"  # Shouldn't happen with default settings

    if st.button("Run Preprocessing", type="primary"):
        with st.spinner("Processing data..."):
            work = df.copy()
            if limit_rows:
                work = work.head(int(n_limit)).copy()
                st.info(f"Processing first {len(work):,} rows...")

            if rm_dups:
                before = len(work)
                work = work.drop_duplicates().reset_index(drop=True)
                removed = before - len(work)
                if removed > 0:
                    st.info(f"Removed {removed:,} duplicate rows.")

            # ---- Ensure required columns ----
            need_cols = ["Score", "Summary"]
            missing_cols = [c for c in need_cols if c not in work.columns]
            if missing_cols:
                st.error(f"Required columns missing: {missing_cols}. Found: {list(work.columns)}.")
                st.stop()

            # Convert Score column safely
            try:
                work["Score"] = pd.to_numeric(work["Score"], errors="coerce").astype("Int64")
                # Check for too many NaN scores
                nan_scores = work["Score"].isna().sum()
                if nan_scores > len(work) * 0.5:  # More than 50% NaN
                    st.warning(
                        f"Warning: {nan_scores:,} ({nan_scores / len(work) * 100:.1f}%) scores could not be converted to numbers.")
            except Exception as e:
                st.error(f"Error processing Score column: {e}")
                st.stop()

            work["Summary"] = work["Summary"].astype(str)

            # Choose source text
            if use_summary_only:
                if fallback_to_text and "Text" in work.columns:
                    text_fallback = work["Text"].astype(str)
                    summary_empty = work["Summary"].fillna("").str.strip().eq("")
                    src = np.where(summary_empty, text_fallback, work["Summary"])
                    work["__source_text"] = pd.Series(src, index=work.index).astype(str)
                    fallback_count = summary_empty.sum()
                    if fallback_count > 0:
                        st.info(f"Used Text fallback for {fallback_count:,} empty summaries.")
                else:
                    work["__source_text"] = work["Summary"]
            else:
                if "Text" in work.columns:
                    work["__source_text"] = (
                                work["Summary"].fillna("") + " " + work["Text"].astype(str).fillna("")).str.strip()
                else:
                    work["__source_text"] = work["Summary"]

            stop_set = get_stopwords()
            st.info(f"Using {len(stop_set):,} stopwords (preserving sentiment words and negators).")

            # ADDED: Show what words we're preserving
            with st.expander("📝 View Preserved Words"):
                st.write("**Sentiment words preserved:**", sorted(list(SENTIMENT_WORDS)))
                st.write("**Negation words preserved:**", sorted(list(NEGATORS)))

            st.markdown("#### Cleaning Text...")
            if len(work) > 10000:
                st.info("Processing large dataset... This may take a moment.")

            with st.spinner("Normalizing text (negation-aware)..."):
                work["clean_text"] = work["__source_text"].apply(lambda x: preprocess_text(x, stop_set))

            # ENHANCED: Better cleaning diagnostics
            empty_before = (work["__source_text"].fillna("").str.strip() == "").sum()
            empty_after = (work["clean_text"].str.strip() == "").sum()

            # Check token preservation
            original_tokens = work["__source_text"].str.split().str.len().sum()
            cleaned_tokens = work["clean_text"].str.split().str.len().sum()
            preservation_rate = cleaned_tokens / original_tokens if original_tokens > 0 else 0

            st.info(f"Text cleaning results: {empty_before:,} → {empty_after:,} empty texts")
            st.info(f"Token preservation: {preservation_rate:.1%} ({cleaned_tokens:,}/{original_tokens:,} tokens kept)")

            if preservation_rate < 0.3:
                st.warning("⚠️ Low token preservation rate - consider less aggressive preprocessing")

            if drop_empty:
                before = len(work)
                work = work[work["clean_text"].str.len() > 0].reset_index(drop=True)
                removed = before - len(work)
                if removed > 0:
                    st.info(f"Dropped {removed:,} rows with empty cleaned text.")

            # ENHANCED: Map to Emotion with validation
            work["Emotion"] = work["Score"].apply(score_to_emotion)

            # Check for unknown emotions
            unknown_count = (work["Emotion"] == "Unknown").sum()
            if unknown_count > 0:
                st.warning(f"Found {unknown_count:,} reviews with unknown emotions (likely invalid scores)")

            # CRITICAL: Check for severe class imbalance
            emotion_counts = work["Emotion"].value_counts()
            total_valid = emotion_counts.sum()

            st.markdown("#### ⚠️ CLASS BALANCE CHECK")
            max_class_pct = emotion_counts.max() / total_valid
            min_class_pct = emotion_counts.min() / total_valid
            imbalance_ratio = emotion_counts.max() / emotion_counts.min()

            if max_class_pct > 0.7:
                st.error(
                    f"🚨 **SEVERE CLASS IMBALANCE DETECTED!** {emotion_counts.idxmax()} represents {max_class_pct:.1%} of data")
                st.error("This will cause the model to predict everything as the majority class!")
                st.info("**Recommendations:**")
                st.write("• Use SMOTE during model training")
                st.write("• Use class weights during model training")
                st.write("• Consider getting more balanced data")
            elif imbalance_ratio > 3:
                st.warning(
                    f"⚠️ **Moderate class imbalance:** {imbalance_ratio:.1f}:1 ratio between max and min classes")
                st.info("Consider using class balancing techniques during training")
            else:
                st.success("✅ **Good class balance** detected")

            # Show statistics about text lengths
            lengths = work["clean_text"].str.split().str.len()
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            with col_stats1:
                st.metric("Final Rows", f"{len(work):,}")
            with col_stats2:
                st.metric("Avg Words/Review", f"{lengths.mean():.1f}")
            with col_stats3:
                st.metric("Median Words", f"{lengths.median():.0f}")
            with col_stats4:
                st.metric("Unique Emotions", f"{work['Emotion'].nunique()}")

            # Show emotion distribution
            st.subheader("Emotion Distribution")

            # Create three columns for better layout
            col_chart, col_stats, col_examples = st.columns([2, 1, 2])

            with col_chart:
                st.bar_chart(emotion_counts)

            with col_stats:
                st.write("**Distribution:**")
                for emotion, count in emotion_counts.items():
                    percentage = (count / len(work)) * 100
                    st.write(f"• {emotion}: {count:,} ({percentage:.1f}%)")

            with col_examples:
                st.write("**Sample cleaned texts:**")
                for emotion in emotion_counts.index:
                    sample = work[work["Emotion"] == emotion]["clean_text"].iloc[0]
                    st.write(f"**{emotion}:** {sample[:60]}...")

            # ENHANCED: Better preview with sentiment analysis
            st.subheader("Preview (first 12)")
            cols = [c for c in ["Score", "Emotion", "Summary", "clean_text"] if c in work.columns]
            preview_df = work[cols].head(12).copy()

            # Truncate long text for better display
            if "clean_text" in preview_df.columns:
                preview_df["clean_text_preview"] = preview_df["clean_text"].str[:80] + "..."
                preview_df = preview_df.drop("clean_text", axis=1)
            if "Summary" in preview_df.columns:
                preview_df["Summary"] = preview_df["Summary"].str[:60] + "..."

            st.dataframe(preview_df, use_container_width=True)

            # ADDED: Sample sentiment words check
            st.markdown("#### 🔍 Sentiment Word Preservation Check")

            # Check if sentiment words are being preserved
            all_cleaned_text = " ".join(work["clean_text"].tolist())
            preserved_sentiment = [w for w in SENTIMENT_WORDS if w in all_cleaned_text]
            preserved_negators = [w for w in NEGATORS if w in all_cleaned_text]

            col_sent1, col_sent2 = st.columns(2)
            with col_sent1:
                st.write(f"**Sentiment words found:** {len(preserved_sentiment)}/{len(SENTIMENT_WORDS)}")
                if preserved_sentiment:
                    st.write(f"Found: {', '.join(preserved_sentiment[:10])}")
                else:
                    st.error("🚨 NO sentiment words found in cleaned text!")

            with col_sent2:
                st.write(f"**Negation words found:** {len(preserved_negators)}/{len(NEGATORS)}")
                if preserved_negators:
                    st.write(f"Found: {', '.join(preserved_negators)}")
                else:
                    st.error("🚨 NO negation words found in cleaned text!")

            # Cache to session
            final_df = work.drop(columns=["__source_text"], errors='ignore')
            st.session_state["df_clean"] = final_df
            st.session_state["preprocess_text"] = preprocess_text
            st.session_state["preproc_stopwords"] = stop_set

            # Download section
            st.markdown("### Download Cleaned Dataset")
            csv_data = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download cleaned CSV",
                data=csv_data,
                file_name="amazon_reviews_cleaned.csv",
                mime="text/csv"
            )

            st.success("✅ Preprocessing complete — df_clean ready (cleaned on **Summary**).")
            st.info("Next step: **Post-Cleaning Diagnostics** page.")


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
    -  Corpus shape, token length distribution  
    -  Vocabulary size & lexical richness  
    -  Top tokens (overall & per emotion)  
    -  Emotion distribution  
    - ️ Short-review & duplicate guardrails  
    """)

    # 0) Load cleaned data
    df = st.session_state.get("df_clean")
    if df is None or "clean_text" not in df.columns or "Emotion" not in df.columns:
        st.error("No cleaned dataset found. Please run **Preprocess & Emotion Mapping** first.")
        st.stop()
    df = df.copy()

    # Optional sampling for very large datasets (keeps page responsive)
    with st.expander(" Performance options"):
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
        st.info(f" Diagnosing on a random sample of **{len(df):,}** rows (out of {total_rows:,}).")

    # 1) Corpus shape & token lengths
    st.subheader(" Corpus Shape & Hygiene")
    rows = len(df)
    nn_clean = df["clean_text"].notna().sum()
    nn_emot = df["Emotion"].notna().sum()

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.write(f"**Total rows:** {rows:,}")
        st.write(f"**Non-null clean_text:** {nn_clean:,}")
        st.write(f"**Non-null Emotion:** {nn_emot:,}")

    # Check for data quality issues
    with col_info2:
        if nn_clean < rows:
            missing_clean = rows - nn_clean
            st.warning(f"⚠️ {missing_clean:,} rows missing clean_text")
        if nn_emot < rows:
            missing_emot = rows - nn_emot
            st.warning(f"⚠️ {missing_emot:,} rows missing emotion")
        if nn_clean == rows and nn_emot == rows:
            st.success("✅ All rows have complete data")

    # token lengths (fast split count)
    lengths = df["clean_text"].fillna("").str.split().str.len()

    # Handle edge case of empty dataset
    if len(lengths) == 0 or lengths.max() == 0:
        st.warning("⚠️ No valid tokens found in the dataset.")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Median tokens/review", int(np.median(lengths)))
    m2.metric("Mean tokens/review", f"{np.mean(lengths):.2f}")
    m3.metric("Max tokens/review", int(np.max(lengths)))

    # helpful quantiles to understand tails
    try:
        q10, q50, q90, q99 = np.quantile(lengths, [0.1, 0.5, 0.9, 0.99])
        st.caption(f"📏 Length quantiles — 10%: {q10:.0f}, 50%: {q50:.0f}, 90%: {q90:.0f}, 99%: {q99:.0f}")
    except Exception as e:
        st.warning(f"Could not calculate quantiles: {e}")

    # Plot length distribution
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(lengths, bins=min(80, max(10, int(lengths.max()))), kde=True, ax=ax)
        ax.set_title("Distribution of Review Lengths (tokens)", fontsize=14)
        ax.set_xlabel("Tokens per review")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not create length distribution plot: {e}")

    # 2) Lexical richness
    st.subheader(" Vocabulary & Lexical Richness")

    @st.cache_data(show_spinner=False)
    def _token_series(clean_col: pd.Series) -> pd.Series:
        return clean_col.fillna("").str.split()

    tokens_series = _token_series(df["clean_text"])

    @st.cache_data(show_spinner=False)
    def _vocab_counter(tokens: pd.Series) -> Counter:
        return Counter(t for row in tokens for t in row if len(t) > 0)  # Filter empty tokens

    vocab_counter = _vocab_counter(tokens_series)
    vocab_size = len(vocab_counter)
    total_tokens = int(lengths.sum())

    if total_tokens == 0:
        st.warning("⚠️ No tokens found for vocabulary analysis.")
        st.stop()

    ttr = (vocab_size / total_tokens) if total_tokens > 0 else 0.0
    herdan_c = (math.log(vocab_size + 1) / math.log(total_tokens + 1)) if total_tokens > 0 else 0.0
    hapax_prop = sum(1 for _, c in vocab_counter.items() if c == 1) / max(vocab_size, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vocabulary size", f"{vocab_size:,}")
    c2.metric("Total tokens", f"{total_tokens:,}")
    c3.metric("Type–Token Ratio (TTR)", f"{ttr:.4f}")
    c4.metric("Herdan's C", f"{herdan_c:.4f}")

    # Add interpretation of metrics
    st.caption(f" Hapax proportion (frequency = 1): **{hapax_prop:.3f}**")

    # Add lexical richness interpretation
    if ttr < 0.1:
        richness = "Low (repetitive vocabulary)"
    elif ttr < 0.3:
        richness = "Moderate"
    else:
        richness = "High (diverse vocabulary)"
    st.caption(f" Lexical richness: **{richness}**")

    # 3) Emotion distribution
    st.subheader(" Emotion Distribution")
    emo_counts = df["Emotion"].value_counts()

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.bar_chart(emo_counts)
    with col_table:
        emo_df = emo_counts.rename("Count").to_frame()
        emo_df["Percentage"] = (emo_df["Count"] / emo_df["Count"].sum() * 100).round(1)
        st.dataframe(emo_df, use_container_width=True)

    # Check for class imbalance
    min_class = emo_counts.min()
    max_class = emo_counts.max()
    imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

    if imbalance_ratio > 10:
        st.warning(
            f"⚠️ Significant class imbalance detected (ratio: {imbalance_ratio:.1f}:1). Consider using class weights or SMOTE.")
    elif imbalance_ratio > 3:
        st.info(f" Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1). Monitor model performance.")

    # 4) Top tokens overall (frequency) + per emotion
    st.subheader(" Top Tokens — Overall (frequency)")
    top_overall_k = st.slider("Show top N tokens", 10, 50, 20, 5, key="diag_top_overall_k")

    if vocab_size > 0:
        top_overall = pd.DataFrame(vocab_counter.most_common(top_overall_k),
                                   columns=["Token", "Frequency"])

        col_table, col_chart = st.columns([1, 1])
        with col_table:
            st.dataframe(top_overall, use_container_width=True)

        with col_chart:
            try:
                fig2, ax2 = plt.subplots(figsize=(7, 6))
                sns.barplot(y="Token", x="Frequency", data=top_overall, ax=ax2)
                ax2.set_title(f"Top {top_overall_k} Tokens (Overall)")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                plt.close(fig2)
            except Exception as e:
                st.warning(f"Could not create token frequency plot: {e}")
    else:
        st.warning("No tokens found for analysis.")

    # Per-emotion token view (helps spot leakage/bias)
    st.subheader(" Top Tokens — Per Emotion")
    per_k = st.slider("Top N per emotion", 5, 40, 15, 5, key="diag_top_per_emo")

    emotions = sorted(df["Emotion"].dropna().unique())
    if len(emotions) > 0:
        cols = st.columns(min(3, len(emotions)))
        for i, emo in enumerate(emotions):
            with cols[i % len(cols)]:
                emo_mask = df["Emotion"] == emo
                emo_tokens = tokens_series[emo_mask]
                c = Counter(t for row in emo_tokens for t in row if len(t) > 0)

                if len(c) > 0:
                    top_e = pd.DataFrame(c.most_common(per_k), columns=["Token", "Frequency"])
                    st.markdown(f"**{emo}** ({emo_mask.sum():,} reviews)")
                    st.dataframe(top_e, use_container_width=True)
                else:
                    st.markdown(f"**{emo}**")
                    st.info("No tokens found")

    # 5) Optional: overall TF-IDF signal check (diagnostic only)
    with st.expander(" Optional: Overall TF-IDF Signal Check"):
        max_feats = st.slider("Max features", 1_000, 12_000, 4_000, 500,
                              help="Limit vocabulary to keep this fast.")
        token_pattern = r"(?u)\b\w+\b"  # keep all word chars; your clean_text already removed most noise

        @st.cache_data(show_spinner=False)
        def _tfidf_overall(clean_text: pd.Series, max_features: int):
            try:
                vec = TfidfVectorizer(max_features=max_features, token_pattern=token_pattern)
                X = vec.fit_transform(clean_text.fillna(""))
                vocab = np.array(vec.get_feature_names_out())
                mean_scores = np.asarray(X.mean(axis=0)).ravel()
                return vocab, mean_scores
            except Exception as e:
                st.error(f"TF-IDF analysis failed: {e}")
                return np.array([]), np.array([])

        vocab, mean_scores = _tfidf_overall(df["clean_text"], int(max_feats))

        if len(vocab) > 0:
            k = st.slider("Top N tokens to display", 10, min(40, len(vocab)), 20, 5, key="diag_tfidf_topk")
            order = np.argsort(mean_scores)[::-1][:k]
            tfidf_overall = pd.DataFrame({"Token": vocab[order], "Mean TF-IDF": mean_scores[order]})
            st.dataframe(tfidf_overall, use_container_width=True)
        else:
            st.warning("Could not perform TF-IDF analysis.")

    # 6) Guardrail: short reviews
    st.subheader("⚠️ Short Reviews (≤ threshold tokens)")
    thr = st.slider("Threshold (tokens)", 1, 10, 2, 1, key="diag_short_thr")
    short_mask = lengths <= thr
    n_short = int(short_mask.sum())

    col_short1, col_short2 = st.columns(2)
    with col_short1:
        st.metric("Short reviews", f"{n_short:,}")
        st.metric("Percentage", f"{(n_short / len(df) * 100):.1f}%")

    with col_short2:
        if n_short > 0:
            st.warning(f"Found {n_short:,} reviews with ≤ {thr} tokens.")
            if n_short > len(df) * 0.1:  # More than 10%
                st.error("⚠️ High proportion of short reviews may indicate preprocessing issues.")
        else:
            st.success("✅ No short reviews found.")

    with st.expander("👀 Preview some short reviews"):
        if n_short > 0:
            short_sample = df.loc[short_mask, ["Score", "Emotion", "clean_text"]].head(20)
            st.dataframe(short_sample, use_container_width=True)
        else:
            st.info("No short reviews to display.")

    apply_short = st.checkbox(" Remove these short reviews and update session",
                              value=False, key="diag_drop_short")
    # We do NOT mutate df in place; only write back if confirmed:
    if apply_short and n_short > 0:
        df2 = df.loc[~short_mask].reset_index(drop=True)
        st.session_state["df_clean"] = df2
        st.success(f"✅ Removed {n_short:,} rows. Session updated. Re-run diagnostics if you wish.")

    # 7) Optional: duplicate detection (exact + stable hash)
    with st.expander(" Optional: Duplicates / near-duplicates"):
        st.caption("Uses exact `clean_text` and a **stable MD5** content hash (Python's builtin hash is not stable).")

        # A) exact duplicates by cleaned text
        dup_mask_exact = df["clean_text"].duplicated(keep="first")
        n_dup_exact = int(dup_mask_exact.sum())

        # B) stable content hash (md5 of cleaned text)
        @st.cache_data(show_spinner=False)
        def _content_hashes(series: pd.Series) -> pd.Series:
            return series.fillna("").map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())

        hashes = _content_hashes(df["clean_text"])
        dup_mask_hash = hashes.duplicated(keep="first")
        n_dup_hash = int(dup_mask_hash.sum())

        # Display duplicate statistics
        col_dup1, col_dup2 = st.columns(2)
        with col_dup1:
            st.metric("Exact duplicates", f"{n_dup_exact:,}")
            st.metric("Hash duplicates", f"{n_dup_hash:,}")

        with col_dup2:
            if n_dup_exact > 0:
                st.warning(f"Found {n_dup_exact:,} exact duplicates")
            if n_dup_hash > 0 and n_dup_hash != n_dup_exact:
                st.info(f"Found {n_dup_hash:,} hash-based duplicates")

        if n_dup_exact > 0:
            st.subheader("Sample duplicate reviews:")
            sample_dups = df.loc[dup_mask_exact, ["Emotion", "clean_text"]].head(10)
            st.dataframe(sample_dups, use_container_width=True)

        # Drop options (separate; pick one)
        col_drop1, col_drop2 = st.columns(2)
        with col_drop1:
            drop_exact = st.checkbox("️ Drop exact `clean_text` duplicates and update session",
                                     value=False, key="diag_drop_dups_exact")
            if drop_exact and n_dup_exact > 0:
                df2 = df.loc[~dup_mask_exact].reset_index(drop=True)
                st.session_state["df_clean"] = df2
                st.success(f"✅ Dropped {n_dup_exact:,} exact duplicates. Session updated.")

        with col_drop2:
            drop_hash = st.checkbox("️ Drop stable-hash duplicates and update session",
                                    value=False, key="diag_drop_dups_hash")
            if drop_hash and n_dup_hash > 0:
                df2 = df.loc[~dup_mask_hash].reset_index(drop=True)
                st.session_state["df_clean"] = df2
                st.success(f"✅ Dropped {n_dup_hash:,} hash-identified duplicates. Session updated.")

    # Summary section
    st.markdown("---")
    st.subheader(" Diagnostic Summary")

    issues = []
    if nn_clean < rows:
        issues.append(f"Missing clean_text: {rows - nn_clean:,}")
    if nn_emot < rows:
        issues.append(f"Missing emotions: {rows - nn_emot:,}")
    if n_short > len(df) * 0.1:
        issues.append(f"High proportion of short reviews: {n_short:,}")
    if imbalance_ratio > 10:
        issues.append(f"Severe class imbalance: {imbalance_ratio:.1f}:1")
    if n_dup_exact > len(df) * 0.05:
        issues.append(f"High duplication rate: {n_dup_exact:,}")

    if issues:
        st.warning("⚠️ **Issues detected:**")
        for issue in issues:
            st.write(f"• {issue}")
        st.info("Consider addressing these issues before proceeding to modeling.")
    else:
        st.success("✅ **Dataset looks good!** Ready for embedding and modeling.")

    st.info(" Diagnostics complete. If you applied any guardrails, re-run this page to see updated stats.")


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

    st.title(" Embeddings (Word2Vec)")
    st.caption("Train on TRAIN only; build review vectors; cache for modeling. Optional full SIF + L2 norm.")

    # ---------- Guards ----------
    df_clean = st.session_state.get("df_clean")
    if df_clean is None or "clean_text" not in df_clean.columns or "Emotion" not in df_clean.columns:
        st.error("Run **Preprocess & Emotion Mapping** first.")
        st.stop()

    # Tokens
    work = df_clean[["clean_text", "Emotion", "Score"]].copy()
    work["tokens"] = work["clean_text"].astype(str).apply(str.split)

    # Check if we have valid tokens
    total_tokens = work["tokens"].apply(len).sum()
    if total_tokens == 0:
        st.error("No tokens found in cleaned text. Please check preprocessing.")
        st.stop()

    st.info(f" Dataset: {len(work):,} reviews with {total_tokens:,} total tokens")

    # ---------- Hyper-parameters & options ----------
    st.markdown("####  Word2Vec Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        vector_size = int(st.number_input("vector_size", 50, 600, 200, 25))
        window = int(st.number_input("window", 2, 15, 5, 1))
        min_count = int(st.number_input("min_count", 1, 20, 5, 1))
    with c2:
        sg_choice = st.selectbox("Architecture", ["Skip-gram (sg=1)", "CBOW (sg=0)"], index=0)
        epochs = int(st.number_input("epochs", 3, 50, 10, 1))
        negative = int(st.number_input("negative sampling (0=off)", 0, 20, 10, 1))
    with c3:
        test_size = float(st.slider("Test size (hold-out)", 0.1, 0.4, 0.2, 0.05))
        seed = int(st.number_input("random_state / seed", 0, 9999, 42, 1))
        use_sif = st.checkbox("Use SIF token weighting", value=False,
                              help="Smooth Inverse Frequency weighting on tokens (TRAIN-only stats).")

    # Advanced options
    with st.expander(" Advanced Options (recommended defaults)"):
        sample = float(st.number_input("Downsampling of very frequent words (sample)", 0.0, 0.01, 1e-3, 1e-4,
                                       help="Higher → more aggressive subsampling of very frequent words. 1e-3 is a strong default."))
        remove_pc = st.checkbox("Remove 1st principal component (full SIF)", value=use_sif,
                                help="Standard SIF: subtract projection on top PC fitted on TRAIN embeddings.")
        do_l2 = st.checkbox("L2-normalize review vectors", value=True)

        # NEW: Add Doc2Vec option
        use_doc2vec = st.checkbox(" Also train Doc2Vec model (experimental)", value=False,
                                  help="Train Doc2Vec alongside Word2Vec for better document representations")

    sg = 1 if "sg=1" in sg_choice else 0
    st.caption("💡 Tip: With vocab ≈30–40k, min_count=5, vector_size=200, sg=1, sample≈1e-3 are strong defaults.")

    # ---------- Stratified split ----------
    try:
        trn, tst = train_test_split(
            work, test_size=test_size, random_state=seed, stratify=work["Emotion"]
        )
    except ValueError as e:
        st.error(f"Could not create stratified split: {e}")
        st.info("This might happen if some emotion classes have very few samples.")
        st.stop()

    st.info(f" Training Word2Vec on TRAIN only: **{len(trn):,}** documents (Test: **{len(tst):,}**)")

    # Show emotion distribution in splits
    col_train, col_test = st.columns(2)
    with col_train:
        st.write("**TRAIN emotions:**")
        train_dist = trn["Emotion"].value_counts()
        for emo, count in train_dist.items():
            st.write(f"• {emo}: {count:,}")
    with col_test:
        st.write("**TEST emotions:**")
        test_dist = tst["Emotion"].value_counts()
        for emo, count in test_dist.items():
            st.write(f"• {emo}: {count:,}")

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

    # NEW: Doc2Vec training function
    @st.cache_resource(show_spinner=True)
    def train_doc2vec(corpus_tokens, vec_size, epochs, seed, min_count):
        try:
            from gensim.models import Doc2Vec
            from gensim.models.doc2vec import TaggedDocument

            tagged_docs = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(corpus_tokens)]
            model = Doc2Vec(
                tagged_docs,
                vector_size=vec_size,
                window=5,
                min_count=min_count,
                workers=min(4, os.cpu_count() or 1),
                epochs=epochs,
                seed=seed
            )
            return model
        except ImportError:
            st.error("Doc2Vec requires gensim. Install with: pip install gensim")
            return None

    with st.spinner(" Training Word2Vec..."):
        w2v = train_w2v(
            trn["tokens"].tolist(), vector_size, window, min_count, sg, epochs, negative, sample, seed
        )

    vocab_size = len(w2v.wv.key_to_index)
    st.success(f"✅ Trained Word2Vec. Vocabulary size (after min_count={min_count}): **{vocab_size:,}**")

    # Train Doc2Vec if requested
    doc2vec_model = None
    if use_doc2vec:
        with st.spinner("🏋️ Training Doc2Vec..."):
            doc2vec_model = train_doc2vec(trn["tokens"].tolist(), vector_size, epochs, seed, min_count)
            if doc2vec_model:
                st.session_state["doc2vec_model"] = doc2vec_model
                st.success("✅ Doc2Vec model trained successfully!")

    # ---------- Nearest-neighbor sanity check ----------
    st.markdown("####  Nearest Words (Sanity Check)")
    default_terms = [w for w in ["good", "bad", "love", "terrible", "taste", "delicious", "awful"] if w in w2v.wv]
    options_list = list(w2v.wv.key_to_index.keys())

    col_terms, col_results = st.columns([1, 2])
    with col_terms:
        probe_terms = st.multiselect(
            "Pick words to inspect",
            options=options_list[:min(5000, len(options_list))],  # Limit for performance
            default=default_terms[:5]  # Limit default selection
        )

    with col_results:
        if probe_terms:
            for term in probe_terms[:3]:  # Limit to 3 terms for display
                try:
                    sims = w2v.wv.most_similar(term, topn=6)
                    similar_words = ", ".join([f"{w} ({s:.2f})" for w, s in sims])
                    st.write(f"**{term}** → {similar_words}")
                except KeyError:
                    st.write(f"**{term}** → Not in vocabulary")

    # Optional tiny intrinsic check
    def _cos(a, b):
        if a not in w2v.wv or b not in w2v.wv:
            return None
        va, vb = w2v.wv[a], w2v.wv[b]
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))

    # Semantic sanity checks
    if {"good", "great", "terrible"} <= set(w2v.wv.key_to_index):
        c1 = _cos("good", "great")
        c2 = _cos("good", "terrible")
        if c1 is not None and c2 is not None:
            st.caption(f" Semantic sanity: cos(good, great)={c1:.3f} vs cos(good, terrible)={c2:.3f}")
            if c1 > c2:
                st.success("✅ Good semantic relationships detected!")
            else:
                st.warning("⚠️ Unexpected semantic relationships. Consider adjusting parameters.")

    # ---------- Build review vectors (mean or SIF) ----------
    st.markdown("####  Building Document Vectors")
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
    with st.spinner("🔄 Computing document vectors..."):
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
            st.caption("✅ Applied full SIF: removed 1st principal component fitted on TRAIN.")
        except Exception as e:
            st.warning(f"Could not apply PC removal: {e}. Proceeding without it.")

    # Optional: L2-normalize
    if do_l2:
        norms = np.linalg.norm(X_emb, axis=1, keepdims=True)
        X_emb = X_emb / (norms + 1e-12)
        st.caption("✅ Applied L2 normalization to review vectors.")

    # ---------- Quality Assessments ----------
    st.markdown("#### Embedding Quality Assessment")

    # QA stats
    zero_rows = int((X_emb == 0).all(axis=1).sum())
    col_qa1, col_qa2, col_qa3 = st.columns(3)

    with col_qa1:
        st.metric("Zero vectors", f"{zero_rows:,}")
        st.metric("Zero vector %", f"{zero_rows / len(work):.2%}")

    with col_qa2:
        avg_norm = np.mean(np.linalg.norm(X_emb, axis=1))
        std_norm = np.std(np.linalg.norm(X_emb, axis=1))
        st.metric("Avg vector norm", f"{avg_norm:.3f}")
        st.metric("Std vector norm", f"{std_norm:.3f}")

    with col_qa3:
        non_zero_dims = np.mean(np.sum(X_emb != 0, axis=1))
        st.metric("Avg non-zero dims", f"{non_zero_dims:.1f}")
        st.metric("Vector utilization", f"{non_zero_dims / dim:.2%}")

    if zero_rows / len(work) > 0.1:  # More than 10% zero vectors
        st.warning(f"⚠️ High zero-vector rate ({zero_rows / len(work):.2%}). Consider lowering `min_count`.")

    # Token coverage (in-vocab) overall & per emotion
    def token_coverage(rows_tokens) -> tuple[int, int]:
        total, invocab = 0, 0
        for toks in rows_tokens:
            total += len(toks)
            invocab += sum(1 for t in toks if t in w2v.wv)
        return invocab, total

    inv, tot = token_coverage(work["tokens"])
    st.write(f"** Token coverage overall**: {inv:,}/{tot:,} = {(inv / max(tot, 1)):.2%}")

    # Per-emotion coverage
    emotions = sorted(work["Emotion"].unique())
    cov_cols = st.columns(len(emotions))
    for i, emo in enumerate(emotions):
        with cov_cols[i]:
            inv_e, tot_e = token_coverage(work.loc[work["Emotion"] == emo, "tokens"])
            coverage_pct = inv_e / max(tot_e, 1)
            st.metric(
                f"{emo} coverage",
                f"{coverage_pct:.2%}",
                help=f"{inv_e:,}/{tot_e:,} tokens in-vocab"
            )
            if coverage_pct < 0.7:  # Less than 70% coverage
                st.caption("⚠️ Low coverage")

    # Show most down-weighted (frequent) tokens under SIF for transparency
    if use_sif:
        with st.expander(" Most frequent tokens in TRAIN (lowest SIF weights)"):
            top_n = int(st.slider("Show top N frequent tokens", 10, 100, 30, 5))
            items = sorted(train_token_freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            rows = [{"Token": t, "Freq(TRAIN)": c, "SIF weight": f"{sif_weight(t):.4f}"} for t, c in items]
            sif_df = pd.DataFrame(rows)
            st.dataframe(sif_df, use_container_width=True)

            # Show average SIF weights by frequency buckets
            st.write("**SIF weight distribution:**")
            high_freq = [sif_weight(t) for t, c in items[:10]]
            mid_freq = [sif_weight(t) for t, c in items[10:20]]
            low_freq = [sif_weight(t) for t, c in items[20:30]]

            st.write(f"• High freq (top 10): avg weight = {np.mean(high_freq):.4f}")
            st.write(f"• Mid freq (10-20): avg weight = {np.mean(mid_freq):.4f}")
            st.write(f"• Low freq (20-30): avg weight = {np.mean(low_freq):.4f}")

    # Encode labels
    label_map = {lbl: idx for idx, lbl in enumerate(sorted(work["Emotion"].unique()))}
    y_labels = work["Emotion"].map(label_map).values

    # Final summary
    st.markdown("####  Summary")
    col_sum1, col_sum2 = st.columns(2)
    with col_sum1:
        st.write(f"**Embeddings shape:** {X_emb.shape}")
        st.write(f"**Labels map:** {label_map}")
        st.write(f"**Train/Test split:** {len(trn):,}/{len(tst):,}")

    with col_sum2:
        embedding_desc = f"Word2Vec({'SIF' if use_sif else 'mean'}{'-PCrm' if (use_sif and remove_pc) else ''}{'-L2' if do_l2 else ''})"
        st.write(f"**Embedding type:** {embedding_desc}")
        st.write(f"**Vocabulary size:** {vocab_size:,}")
        if doc2vec_model:
            st.write("**Doc2Vec:** ✅ Available")

    # ---------- Persist to session ----------
    st.session_state["w2v_model"] = w2v
    st.session_state["X_emb"] = X_emb
    st.session_state["y_labels"] = y_labels
    st.session_state["label_map"] = label_map
    st.session_state["train_index"] = trn.index.to_numpy()
    st.session_state["test_index"] = tst.index.to_numpy()
    st.session_state["embedding_used"] = embedding_desc

    st.success(" Per-review embeddings are ready ✅ — proceed to **Modeling & Results**.")
    st.caption(" Note: Multi-threaded W2V can have tiny non-determinism across runs; a fixed seed reduces this.")


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
    st.write(f"**Features:** {X.shape}  |  **Classes:** {classes_sorted}")
    st.caption(f"**Embedding used:** {emb_used}")

    # --------- TRAIN / TEST split (fixed from Word2Vec page) ---------
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # Display class distribution with imbalance warning
    col_train_dist, col_test_dist = st.columns(2)
    with col_train_dist:
        train_counts = dict(Counter(y_tr))
        st.write(f"**Train counts:** {train_counts}")

        # Check for severe imbalance in training data
        total_train = sum(train_counts.values())
        max_class_pct = max(train_counts.values()) / total_train
        if max_class_pct > 0.7:
            st.error(f"🚨 **SEVERE IMBALANCE:** {max_class_pct:.1%} majority class!")
            st.error("**This will cause positive prediction bias!**")
            st.info("**SOLUTION:** Use SMOTE or Class weights below ⬇️")

    with col_test_dist:
        test_counts = dict(Counter(y_te))
        st.write(f"**Test counts:** {test_counts}")

    # --------- Controls with enhanced guidance ---------
    with st.expander("🎛️ Training Controls", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_splits = int(st.number_input("CV folds (StratifiedKFold on TRAIN)", 3, 10, 5, 1))
            random_state = int(st.number_input("random_state", 0, 9999, 42, 1))
        with c2:
            # CRITICAL: Make SMOTE the default for imbalanced data
            default_imb = "SMOTE" if max_class_pct > 0.6 else "None"
            imb = st.selectbox("Imbalance handling", ["SMOTE", "Class weights", "None"],
                               index=["SMOTE", "Class weights", "None"].index(default_imb),
                               help="SMOTE and class weights applied only on TRAIN folds and final TRAIN fit.")

            if imb == "None" and max_class_pct > 0.6:
                st.warning("⚠️ **Recommended:** Use SMOTE or Class weights for imbalanced data!")

        with c3:
            # RF params
            rf_n_estimators = int(st.number_input("RF n_estimators", 100, 2000, 400, 50))
            rf_max_depth = int(st.number_input("RF max_depth (0=None)", 0, 200, 0, 1))
            rf_min_samples_leaf = int(st.number_input("RF min_samples_leaf", 1, 20, 1, 1))

    with st.expander("⚡ XGBoost Hyper-parameters", expanded=True):
        if not XGB_OK:
            st.warning("XGBoost not installed — only Random Forest will run.")
            st.info("Install with: `pip install xgboost`")
        else:
            col_xgb1, col_xgb2 = st.columns(2)
            with col_xgb1:
                xgb_n_estimators = int(st.number_input("xgb n_estimators", 100, 2000, 600, 50))
                xgb_lr = float(st.number_input("learning_rate", 0.01, 0.5, 0.1, 0.01))
                xgb_max_depth = int(st.number_input("max_depth", 2, 20, 6, 1))
            with col_xgb2:
                xgb_subsample = float(st.slider("subsample", 0.5, 1.0, 0.8, 0.05))
                xgb_colsample = float(st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05))
                xgb_reg_lambda = float(st.number_input("lambda (L2)", 0.0, 10.0, 1.0, 0.1))

            use_early_stop = st.checkbox("Use early stopping (uses 15% of TRAIN as val)", value=True)

    # ADDED: Imbalance handling explanation
    if imb != "None":
        with st.expander("ℹ️ About Imbalance Handling"):
            if imb == "SMOTE":
                st.markdown("""
                **SMOTE (Synthetic Minority Oversampling Technique):**
                - Creates synthetic examples of minority classes (Negative/Neutral)
                - Balances training data without duplicating existing samples
                - Applied only during training, not on test data
                - **Best for severe imbalance (77% positive like yours)**
                """)
            elif imb == "Class weights":
                st.markdown("""
                **Class Weights:**
                - Gives higher importance to minority class errors during training
                - No synthetic data generation
                - Faster than SMOTE but sometimes less effective
                - **Good for moderate imbalance**
                """)

    # helper: sample weights for class weighting
    def sample_weights(y_vec):
        classes, counts = np.unique(y_vec, return_counts=True)
        total = len(y_vec)
        w = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        return np.array([w[yi] for yi in y_vec])

    # --------- Train button with prominent styling ---------
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        train_button = st.button("🚀 Train & Evaluate Models", type="primary", use_container_width=True)

    if train_button:
        # Show selected configuration
        st.info(
            f"**Training Configuration:** {imb} imbalance handling, {n_splits}-fold CV, Random state: {random_state}")

        with st.spinner("Training models..."):
            # Optionally rebalance TRAIN
            rf_class_weight = None
            sw_tr = None
            Xtr_fit, ytr_fit = X_tr, y_tr  # local copies for final fit

            if imb == "Class weights":
                rf_class_weight = "balanced"
                sw_tr = sample_weights(y_tr)
                st.success("✅ Using balanced class weights for RF and sample weights for XGB.")
            elif imb == "SMOTE":
                if SMOTE_OK:
                    sm = SMOTE(random_state=random_state)
                    Xtr_fit, ytr_fit = sm.fit_resample(X_tr, y_tr)

                    # Show SMOTE results
                    original_counts = Counter(y_tr)
                    smote_counts = Counter(ytr_fit)

                    st.success(f"✅ SMOTE applied on TRAIN: {X_tr.shape} → {Xtr_fit.shape}")

                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.write("**Before SMOTE:**")
                        for cls, count in original_counts.items():
                            st.write(f"• {classes_sorted[cls]}: {count}")
                    with col_after:
                        st.write("**After SMOTE:**")
                        for cls, count in smote_counts.items():
                            st.write(f"• {classes_sorted[cls]}: {count}")
                else:
                    st.error("❌ SMOTE unavailable (install `imbalanced-learn`). Continuing without SMOTE.")
                    st.code("pip install imbalanced-learn")

            # --------- CV on TRAIN only ---------
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            def cv_eval(clf, uses_sample_weight=False):
                precs, recs, f1s, aucs = [], [], [], []
                for fold, (tr_i, va_i) in enumerate(skf.split(X_tr, y_tr)):
                    XtrF, XvaF = X_tr[tr_i], X_tr[va_i]
                    ytrF, yvaF = y_tr[tr_i], y_tr[va_i]

                    # SMOTE on fold-train only
                    if imb == "SMOTE" and SMOTE_OK:
                        sm = SMOTE(random_state=random_state)
                        try:
                            XtrF, ytrF = sm.fit_resample(XtrF, ytrF)
                        except Exception as e:
                            st.warning(f"SMOTE failed on fold {fold + 1}: {e}")

                    swF = sample_weights(ytrF) if (imb == "Class weights" and uses_sample_weight) else None

                    try:
                        if uses_sample_weight and swF is not None:
                            clf.fit(XtrF, ytrF, sample_weight=swF)
                        else:
                            clf.fit(XtrF, ytrF)

                        proba = np.clip(clf.predict_proba(XvaF), 1e-8, 1 - 1e-8)
                        preds = clf.predict(XvaF)

                        # Calculate metrics
                        precs.append(precision_score(yvaF, preds, average="macro", zero_division=0))
                        recs.append(recall_score(yvaF, preds, average="macro", zero_division=0))
                        f1s.append(f1_score(yvaF, preds, average="macro", zero_division=0))

                        # AUC calculation
                        try:
                            yva_bin = label_binarize(yvaF, classes=np.arange(n_classes))
                            if yva_bin.shape[1] == 1:  # Binary case
                                auc = roc_auc_score(yvaF, proba[:, 1])
                            else:
                                auc = roc_auc_score(yva_bin, proba, average="macro", multi_class="ovr")
                        except Exception:
                            auc = np.nan
                        aucs.append(auc)

                    except Exception as e:
                        st.warning(f"Error in CV fold {fold + 1}: {e}")
                        precs.append(0)
                        recs.append(0)
                        f1s.append(0)
                        aucs.append(np.nan)

                return float(np.nanmean(precs)), float(np.nanmean(recs)), float(np.nanmean(f1s)), float(
                    np.nanmean(aucs))

            # Initialize result containers
            metrics_rows = []
            conf_matrices = {}
            roc_curves = {}
            feature_importance = {}

            # ----- Random Forest -----
            st.markdown("#### 🌲 Training Random Forest")
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

            # CV evaluation
            with st.spinner("Running Random Forest CV..."):
                rf_cvP, rf_cvR, rf_cvF1, rf_cvAUC = cv_eval(rf, uses_sample_weight=False)

            # Fit on (optionally rebalanced) TRAIN and eval on TEST
            rf.fit(Xtr_fit, ytr_fit)
            rf_preds = rf.predict(X_te)
            rf_prob = np.clip(rf.predict_proba(X_te), 1e-8, 1 - 1e-8)
            rf_acc = accuracy_score(y_te, rf_preds)

            # Calculate AUC
            try:
                y_te_bin = label_binarize(y_te, classes=np.arange(n_classes))
                if y_te_bin.shape[1] == 1:  # Binary case
                    rf_auc_macro = roc_auc_score(y_te, rf_prob[:, 1])
                    fpr_rf, tpr_rf, _ = roc_curve(y_te, rf_prob[:, 1])
                else:
                    rf_auc_macro = roc_auc_score(y_te_bin, rf_prob, average="macro", multi_class="ovr")
                    fpr_rf, tpr_rf, _ = roc_curve(y_te_bin.ravel(), rf_prob.ravel())
            except Exception as e:
                st.warning(f"ROC calculation failed for RF: {e}")
                rf_auc_macro = np.nan
                fpr_rf, tpr_rf = np.array([0, 1]), np.array([0, 1])

            # ADDED: Quick RF results preview
            rf_pred_counts = Counter(rf_preds)
            st.write("**RF Predictions on Test Set:**")
            col_rf1, col_rf2, col_rf3 = st.columns(3)
            for i, (cls_idx, count) in enumerate(rf_pred_counts.items()):
                col = [col_rf1, col_rf2, col_rf3][i % 3]
                cls_name = classes_sorted[cls_idx]
                pct = count / len(rf_preds) * 100
                col.metric(cls_name, f"{count}", f"{pct:.1f}%")

            # Store RF results
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

            # Save RF model
            st.session_state["rf_model"] = rf

            # ----- XGBoost -----
            xgb = None
            if XGB_OK:
                st.markdown("#### ⚡ Training XGBoost")
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
                    verbosity=0
                )

                # CV evaluation
                with st.spinner("Running XGBoost CV..."):
                    xgb_cvP, xgb_cvR, xgb_cvF1, xgb_cvAUC = cv_eval(xgb, uses_sample_weight=(imb == "Class weights"))

                # Training with early stopping
                if use_early_stop and len(Xtr_fit) > 100:
                    try:
                        Xtr2, Xva2, ytr2, yva2 = train_test_split(
                            Xtr_fit, ytr_fit, test_size=0.15, stratify=ytr_fit, random_state=random_state
                        )
                        sw2 = sample_weights(ytr2) if (imb == "Class weights") else None

                        eval_result = {}
                        xgb.fit(
                            Xtr2, ytr2,
                            sample_weight=sw2,
                            eval_set=[(Xva2, yva2)],
                            eval_metric="mlogloss",
                            callbacks=[],  # Remove verbose callback
                            early_stopping_rounds=30
                        )
                        st.info(f"Early stopping applied - stopped at iteration {xgb.best_iteration}")
                    except Exception as e:
                        st.warning(f"Early stopping failed: {e}. Training without early stopping.")
                        sw2 = sample_weights(ytr_fit) if (imb == "Class weights") else None
                        xgb.fit(Xtr_fit, ytr_fit, sample_weight=sw2)
                else:
                    sw2 = sample_weights(ytr_fit) if (imb == "Class weights") else None
                    xgb.fit(Xtr_fit, ytr_fit, sample_weight=sw2)

                # Test evaluation
                xgb_preds = xgb.predict(X_te)
                xgb_prob = np.clip(xgb.predict_proba(X_te), 1e-8, 1 - 1e-8)
                xgb_acc = accuracy_score(y_te, xgb_preds)

                # ADDED: Quick XGB results preview
                xgb_pred_counts = Counter(xgb_preds)
                st.write("**XGB Predictions on Test Set:**")
                col_xgb1, col_xgb2, col_xgb3 = st.columns(3)
                for i, (cls_idx, count) in enumerate(xgb_pred_counts.items()):
                    col = [col_xgb1, col_xgb2, col_xgb3][i % 3]
                    cls_name = classes_sorted[cls_idx]
                    pct = count / len(xgb_preds) * 100
                    col.metric(cls_name, f"{count}", f"{pct:.1f}%")

                # Calculate AUC for XGBoost
                try:
                    if y_te_bin.shape[1] == 1:  # Binary case
                        xgb_auc_macro = roc_auc_score(y_te, xgb_prob[:, 1])
                        fpr_xgb, tpr_xgb, _ = roc_curve(y_te, xgb_prob[:, 1])
                    else:
                        xgb_auc_macro = roc_auc_score(y_te_bin, xgb_prob, average="macro", multi_class="ovr")
                        fpr_xgb, tpr_xgb, _ = roc_curve(y_te_bin.ravel(), xgb_prob.ravel())
                except Exception as e:
                    st.warning(f"ROC calculation failed for XGBoost: {e}")
                    xgb_auc_macro = np.nan
                    fpr_xgb, tpr_xgb = np.array([0, 1]), np.array([0, 1])

                # Store XGBoost results
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

                # Save XGBoost model
                st.session_state["xgb_model"] = xgb
            else:
                st.info("XGBoost not available; skipping XGB metrics.")

            # ----- Ensemble Model -----
            if XGB_OK and rf is not None and xgb is not None:
                st.markdown("#### 🤝 Training Ensemble Model")

                # Optimize ensemble weights using cross-validation on training set
                ensemble_weights = st.session_state.get("ensemble_weights", [0.5, 0.5])

                with st.expander("🔧 Ensemble Weight Optimization"):
                    optimize_weights = st.checkbox("Optimize ensemble weights", value=True)
                    if optimize_weights:
                        st.info("Finding optimal ensemble weights using grid search...")
                        best_score = 0
                        best_weights = [0.5, 0.5]

                        # Use a validation split from training data for weight optimization
                        try:
                            X_train_ens, X_val_ens, y_train_ens, y_val_ens = train_test_split(
                                X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=random_state
                            )

                            # Retrain models on the ensemble training set
                            rf_temp = RandomForestClassifier(**rf_kwargs)
                            rf_temp.fit(X_train_ens, y_train_ens)

                            xgb_temp = XGBClassifier(
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
                                verbosity=0
                            )
                            xgb_temp.fit(X_train_ens, y_train_ens)

                            # Grid search for weights
                            for w1 in np.arange(0.1, 1.0, 0.1):
                                w2 = 1.0 - w1
                                ensemble_temp = EnsembleClassifier(rf_temp, xgb_temp, [w1, w2])
                                val_preds = ensemble_temp.predict(X_val_ens)
                                val_score = f1_score(y_val_ens, val_preds, average="macro")

                                if val_score > best_score:
                                    best_score = val_score
                                    best_weights = [w1, w2]

                            st.success(
                                f"✅ Best weights found: RF={best_weights[0]:.2f}, XGB={best_weights[1]:.2f} (F1={best_score:.4f})")
                            ensemble_weights = best_weights

                        except Exception as e:
                            st.warning(f"Weight optimization failed: {e}. Using default weights.")

                # Create ensemble with optimized weights
                ensemble = EnsembleClassifier(rf, xgb, ensemble_weights)
                ensemble_preds = ensemble.predict(X_te)
                ensemble_prob = ensemble.predict_proba(X_te)
                ensemble_acc = accuracy_score(y_te, ensemble_preds)

                # ADDED: Quick Ensemble results preview
                ens_pred_counts = Counter(ensemble_preds)
                st.write("**Ensemble Predictions on Test Set:**")
                col_ens1, col_ens2, col_ens3 = st.columns(3)
                for i, (cls_idx, count) in enumerate(ens_pred_counts.items()):
                    col = [col_ens1, col_ens2, col_ens3][i % 3]
                    cls_name = classes_sorted[cls_idx]
                    pct = count / len(ensemble_preds) * 100
                    col.metric(cls_name, f"{count}", f"{pct:.1f}%")

                # Calculate ensemble AUC
                try:
                    if y_te_bin.shape[1] == 1:  # Binary case
                        ensemble_auc_macro = roc_auc_score(y_te, ensemble_prob[:, 1])
                        fpr_ens, tpr_ens, _ = roc_curve(y_te, ensemble_prob[:, 1])
                    else:
                        ensemble_auc_macro = roc_auc_score(y_te_bin, ensemble_prob, average="macro", multi_class="ovr")
                        fpr_ens, tpr_ens, _ = roc_curve(y_te_bin.ravel(), ensemble_prob.ravel())
                except Exception as e:
                    st.warning(f"ROC calculation failed for Ensemble: {e}")
                    ensemble_auc_macro = np.nan
                    fpr_ens, tpr_ens = np.array([0, 1]), np.array([0, 1])

                # Add ensemble metrics to results
                metrics_rows.append({
                    "Model": "Ensemble (RF+XGB)",
                    "CV_Precision": np.nan,  # Could implement ensemble CV
                    "CV_Recall": np.nan,
                    "CV_F1": np.nan,
                    "CV_ROC-AUC": np.nan,
                    "Test_Precision": precision_score(y_te, ensemble_preds, average="macro", zero_division=0),
                    "Test_Recall": recall_score(y_te, ensemble_preds, average="macro", zero_division=0),
                    "Test_F1": f1_score(y_te, ensemble_preds, average="macro", zero_division=0),
                    "Test_ROC-AUC": ensemble_auc_macro,
                    "Test_Accuracy": ensemble_acc
                })

                conf_matrices["Ensemble"] = confusion_matrix(y_te, ensemble_preds, labels=np.arange(n_classes))
                roc_curves["Ensemble"] = (fpr_ens, tpr_ens, ensemble_auc_macro)

                # Save ensemble model
                st.session_state["ensemble_model"] = ensemble
                st.session_state["ensemble_weights"] = ensemble_weights

            # Persist models & results
            results_bundle = {
                "metrics": metrics_rows,
                "conf_matrices": conf_matrices,
                "labels": classes_sorted,
                "roc_curves": roc_curves,
                "feature_importance": feature_importance,
                "embedding_used": emb_used
            }
            st.session_state["results"] = results_bundle

            # --------- Results Presentation ---------
            st.markdown("---")
            st.subheader("📊 Results Summary (CV on TRAIN vs Held-out TEST)")

            res_df = pd.DataFrame(metrics_rows).set_index("Model")
            st.dataframe(res_df.style.format("{:.4f}"), use_container_width=True)

            # Winner by Test_F1 then Test_ROC-AUC
            winner = max(res_df.index, key=lambda m: (res_df.loc[m, "Test_F1"], res_df.loc[m, "Test_ROC-AUC"]))
            st.success(f"🏆 **Overall winner (TEST):** {winner}")

            # CRITICAL: Check if imbalance handling worked
            best_f1 = res_df["Test_F1"].max()
            worst_f1 = res_df["Test_F1"].min()
            improvement = ((best_f1 - worst_f1) / worst_f1) * 100 if worst_f1 > 0 else 0

            if best_f1 > 0.6:
                st.success(f"✅ **Good F1 Score:** {best_f1:.3f} - Imbalance handling appears to be working!")
            elif best_f1 > 0.4:
                st.warning(f"⚠️ **Moderate F1 Score:** {best_f1:.3f} - Consider trying different imbalance methods")
            else:
                st.error(f"❌ **Low F1 Score:** {best_f1:.3f} - Imbalance handling may not be sufficient")
                st.info("**Try:** Different imbalance method, more balanced data, or feature engineering")

            st.info(f"📈 Best model improves F1 by {improvement:.1f}% over worst model")

            # Download results
            st.download_button(
                "📥 Download metrics (CSV)",
                data=res_df.to_csv().encode("utf-8"),
                file_name="model_metrics.csv",
                mime="text/csv"
            )

            # Confusion matrices
            st.subheader("🔀 Confusion Matrices — Held-out TEST")
            n = len(conf_matrices)
            if n > 0:
                fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
                if n == 1:
                    axes = [axes]

                colors = {"Random Forest": "Blues", "XGBoost": "Greens", "Ensemble": "Oranges"}

                for ax, (mname, cm) in zip(axes, conf_matrices.items()):
                    sns.heatmap(
                        np.asarray(cm),
                        annot=True,
                        fmt="d",
                        cmap=colors.get(mname, "viridis"),
                        ax=ax,
                        xticklabels=classes_sorted,
                        yticklabels=classes_sorted,
                        cbar_kws={'label': 'Count'}
                    )
                    ax.set_title(f"{mname}", fontsize=12)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # ADDED: Confusion matrix interpretation
                st.markdown("**💡 How to Read Confusion Matrices:**")
                st.info("• **Diagonal cells (correct predictions)** should be bright/high numbers\n"
                        "• **Off-diagonal cells (errors)** should be dark/low numbers\n"
                        "• **If bottom row is all bright** = predicting everything as positive (bad!)")

            # ROC curves
            st.subheader("📈 ROC Curves (macro AUC)")
            if roc_curves:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                colors = ['blue', 'green', 'orange', 'red']

                for i, (mname, (fpr, tpr, auc_macro)) in enumerate(roc_curves.items()):
                    fpr = np.asarray(fpr)
                    tpr = np.asarray(tpr)
                    color = colors[i % len(colors)]

                    if np.isfinite(auc_macro):
                        label = f"{mname} (AUC={auc_macro:.3f})"
                    else:
                        label = f"{mname} (AUC=n/a)"

                    ax2.plot(fpr, tpr, label=label, color=color, linewidth=2)

                ax2.plot([0, 1], [0, 1], "k--", label="Random guess", alpha=0.5)
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("ROC Curves Comparison")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                plt.close(fig2)

            # Feature importances (top embedding dims)
            st.subheader("🎯 Feature Importance (Top Embedding Dimensions)")

            def plot_top_importances(model_name: str, fi_dict: dict, top_k: int = 15):
                if not fi_dict or "importance" not in fi_dict:
                    st.info(f"No importances available for {model_name}")
                    return

                imp = np.asarray(fi_dict["importance"])
                dims = np.array(fi_dict.get("features", [f"dim_{i}" for i in range(len(imp))]))
                order = np.argsort(imp)[::-1][:top_k]
                top_df = pd.DataFrame({"Feature": dims[order], "Importance": imp[order]})

                st.markdown(f"**{model_name} — Top {top_k} dimensions**")

                col_table, col_chart = st.columns([1, 1])
                with col_table:
                    st.dataframe(top_df, use_container_width=True)

                with col_chart:
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    sns.barplot(data=top_df, y="Feature", x="Importance", ax=ax)
                    ax.set_title(f"{model_name}: Top {top_k} Importances")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

            # Display feature importances
            importance_models = [m for m in feature_importance.keys()]
            if importance_models:
                for model in importance_models:
                    plot_top_importances(model, feature_importance[model])
                    st.markdown("---")

            # Interactive visualizations (FIXED - only one chart)
            st.subheader("📊 Interactive Performance Analysis")
            if PLOTLY_OK:
                try:
                    # Create interactive metrics comparison
                    fig_metrics = go.Figure()

                    test_metrics = ['Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC-AUC']
                    colors = ['blue', 'red', 'green', 'orange']

                    for i, metric in enumerate(test_metrics):
                        fig_metrics.add_trace(go.Scatter(
                            x=res_df.index,
                            y=res_df[metric],
                            mode='markers+lines',
                            name=metric.replace('Test_', ''),
                            line=dict(width=3, color=colors[i % len(colors)]),
                            marker=dict(size=10)
                        ))

                    fig_metrics.update_layout(
                        title="Model Performance Comparison",
                        xaxis_title="Model",
                        yaxis_title="Score",
                        hovermode='x unified',
                        height=500
                    )

                    st.plotly_chart(fig_metrics, use_container_width=True)

                except Exception as e:
                    st.warning(f"Interactive visualization failed: {e}")
            else:
                st.info("💡 Install plotly for interactive visualizations: `pip install plotly`")

            # Advanced evaluation metrics
            st.subheader("🔬 Advanced Evaluation Metrics")
            try:
                # Use the comprehensive evaluation function if available
                advanced_metrics = comprehensive_evaluation_metrics(y_te, rf_preds, rf_prob, classes_sorted)

                col_adv1, col_adv2 = st.columns(2)

                with col_adv1:
                    st.write("**Additional Metrics (Random Forest):**")
                    st.write(f"• Cohen's Kappa: {advanced_metrics['cohen_kappa']:.4f}")
                    st.write(f"• Balanced Accuracy: {advanced_metrics['balanced_accuracy']:.4f}")

                with col_adv2:
                    st.write("**Prediction Confidence:**")
                    conf_stats = advanced_metrics['prediction_confidence']
                    st.write(f"• Mean: {conf_stats['mean']:.4f}")
                    st.write(f"• Std: {conf_stats['std']:.4f}")
                    st.write(f"• Range: {conf_stats['min']:.4f} - {conf_stats['max']:.4f}")

                # Per-class confidence intervals (if available)
                ci_metrics = {k: v for k, v in advanced_metrics.items() if k.endswith('_accuracy_ci')}
                if ci_metrics:
                    st.write("**Per-class Accuracy Confidence Intervals (95%):**")
                    for class_ci, (low, high) in ci_metrics.items():
                        class_name = class_ci.replace('_accuracy_ci', '')
                        st.write(f"• {class_name}: [{low:.3f}, {high:.3f}]")

            except (NameError, Exception) as e:
                st.info("Advanced metrics function not available or failed")

            # Model selection recommendations
            st.subheader("🎯 Model Selection Recommendations")

            if len(res_df) > 1:
                # Find best model for different criteria
                best_f1_model = res_df['Test_F1'].idxmax()
                best_auc_model = res_df['Test_ROC-AUC'].idxmax()
                best_precision_model = res_df['Test_Precision'].idxmax()
                best_recall_model = res_df['Test_Recall'].idxmax()

                recommendations = []
                if best_f1_model == best_auc_model:
                    recommendations.append(f"**{best_f1_model}** excels in both F1-score and ROC-AUC")
                else:
                    recommendations.append(f"**{best_f1_model}** has the best F1-score")
                    recommendations.append(f"**{best_auc_model}** has the best ROC-AUC")

                if best_precision_model != best_f1_model:
                    recommendations.append(
                        f"**{best_precision_model}** has the highest precision (fewer false positives)")

                if best_recall_model != best_f1_model:
                    recommendations.append(f"**{best_recall_model}** has the highest recall (fewer false negatives)")

                for rec in recommendations:
                    st.info(rec)

                # Business context recommendations
                st.markdown("**Use case recommendations:**")
                st.write(
                    "• **High precision needed:** Choose the model with highest precision to minimize false positives")
                st.write("• **High recall needed:** Choose the model with highest recall to minimize false negatives")
                st.write("• **Balanced performance:** Choose the model with highest F1-score")
                st.write("• **Probability calibration important:** Choose the model with highest ROC-AUC")

            # ADDED: Success validation
            st.markdown("---")
            st.subheader("✅ Training Validation")

            # Check if the models are actually predicting different classes
            all_predictions = []
            if rf_preds is not None:
                all_predictions.extend(rf_preds)
            if 'xgb_preds' in locals() and xgb_preds is not None:
                all_predictions.extend(xgb_preds)
            if 'ensemble_preds' in locals() and ensemble_preds is not None:
                all_predictions.extend(ensemble_preds)

            unique_predictions = set(all_predictions)

            if len(unique_predictions) == 1:
                pred_class = classes_sorted[list(unique_predictions)[0]]
                st.error(f"🚨 **STILL PREDICTING ONLY {pred_class}!**")
                st.error("**The imbalance handling didn't work. Try:**")
                st.write("• Different SMOTE parameters")
                st.write("• More aggressive class weights")
                st.write("• Different emotion mapping (e.g., only 5-star = Positive)")
                st.write("• More balanced training data")
            elif len(unique_predictions) == n_classes:
                st.success(
                    f"✅ **SUCCESS!** Models are predicting all {n_classes} classes: {[classes_sorted[i] for i in unique_predictions]}")
            else:
                st.warning(
                    f"⚠️ **Partial Success:** Models are predicting {len(unique_predictions)}/{n_classes} classes: {[classes_sorted[i] for i in unique_predictions]}")

            # Training summary
            st.markdown("---")
            st.subheader("📋 Training Summary")

            summary_info = {
                "Models Trained": len(metrics_rows),
                "Best Test F1": f"{res_df['Test_F1'].max():.4f}",
                "Cross-Validation Folds": n_splits,
                "Imbalance Handling": imb,
                "Train/Test Split": f"{len(X_tr)}/{len(X_te)}",
                "Feature Dimensions": X.shape[1],
                "Classes": len(classes_sorted)
            }

            col_sum1, col_sum2 = st.columns(2)
            items = list(summary_info.items())
            mid = len(items) // 2

            with col_sum1:
                for key, value in items[:mid]:
                    st.write(f"**{key}:** {value}")

            with col_sum2:
                for key, value in items[mid:]:
                    st.write(f"**{key}:** {value}")

            st.success("✅ **Training complete!** Results cached for next steps.")
            st.info(" **Next step:** Go to **Word Clouds** page to visualize emotion-specific vocabulary.")


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

    st.title(" Emotion-Specific Word Clouds")
    st.caption("Word2Vec-centric clouds with leakage-safe option and deterministic layout.")

    # ---- Enhanced Guards with Better Error Messages ----
    df_clean = st.session_state.get("df_clean")
    if df_clean is None or "clean_text" not in df_clean.columns or "Emotion" not in df_clean.columns:
        st.error("❌ No cleaned dataset found. Please run **Preprocess & Labels** first.")
        st.info("💡 The word clouds need preprocessed text data to analyze emotion-specific vocabulary.")
        st.stop()

    # Check if models are trained (helps users understand the pipeline)
    models_trained = any([
        st.session_state.get("rf_model") is not None,
        st.session_state.get("xgb_model") is not None,
        st.session_state.get("ensemble_model") is not None
    ])

    if not models_trained:
        st.warning(
            "⚠️ **No trained models detected.** Word clouds work independently, but trained models help validate results.")
        st.info(
            "💡 **Tip:** Train models first in **Modeling & Results** to see if emotion classification is working properly.")

    EMOTIONS = [e for e in ["Negative", "Neutral", "Positive"] if e in df_clean["Emotion"].unique()]
    if not EMOTIONS:
        st.error("❌ No emotion labels available.")
        st.stop()

    # Show data distribution
    emotion_counts = df_clean["Emotion"].value_counts()
    col_dist1, col_dist2 = st.columns([2, 1])

    with col_dist1:
        st.markdown("####  Data Distribution")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(df_clean)) * 100
            st.write(f"• **{emotion}**: {count:,} ({percentage:.1f}%)")

    with col_dist2:
        # Warn about imbalance affecting word clouds
        max_pct = emotion_counts.max() / len(df_clean)
        if max_pct > 0.7:
            st.error(" **Severe class imbalance detected!**")
            st.caption("This may affect word cloud quality - minority classes may have fewer distinctive words.")

    w2v_model = st.session_state.get("w2v_model")  # required for centroid methods
    WV_KEYS = set(w2v_model.wv.key_to_index) if w2v_model is not None else set()

    # Check Word2Vec availability
    if w2v_model is None:
        st.warning("⚠️ **No Word2Vec model found.** Only contrastive log-odds method will be available.")
        st.info("💡 Train Word2Vec embeddings first for semantic-based word clouds.")

    EMO2CMAP = {"Negative": "Reds", "Neutral": "Blues", "Positive": "Greens"}

    # ---- Enhanced Controls with Better UX ----
    st.markdown("---")
    st.markdown("####  Word Cloud Configuration")

    # Method selection with intelligent defaults
    available_methods = ["Contrastive log-odds (counts)"]
    if w2v_model is not None:
        available_methods = [
            "Word2Vec centroid (semantic)",
            "Word2Vec centroid (SIF)",
            "Contrastive log-odds (counts)"
        ]

    c_top = st.columns(2)
    with c_top[0]:
        method = st.radio(
            "🔬 Weighting method",
            available_methods,
            index=0,
            help="Choose how to weight words: semantic similarity, SIF weighting, or frequency-based contrast"
        )

        # Method explanation
        if "semantic" in method.lower():
            st.info(" **Semantic method**: Groups words by meaning similarity using Word2Vec")
        elif "sif" in method.lower():
            st.info(" **SIF method**: Down-weights common words for more nuanced vocabulary")
        else:
            st.info(" **Log-odds method**: Shows statistically distinctive words for each emotion")

    with c_top[1]:
        use_train_only_for_counts = st.checkbox(
            "🛡️ Use TRAIN only for counts/centroids (avoid leakage)",
            value=True,
            help="Use only training data to compute centroids and frequencies (recommended)"
        )

        # Show leakage explanation
        if use_train_only_for_counts:
            st.success("✅ **Leakage prevention ON** - Using only training data")
        else:
            st.warning("⚠️ **Using full dataset** - May cause overfitting")

    # Source df for counting/centroids
    tr_idx = st.session_state.get("train_index")
    df_counts_source = (
        df_clean.loc[tr_idx].reset_index(drop=True)
        if (use_train_only_for_counts and tr_idx is not None)
        else df_clean
    )

    # Enhanced data source info
    source_info = f"Using {'TRAIN split' if use_train_only_for_counts and tr_idx is not None else 'FULL dataset'} ({len(df_counts_source):,} reviews)"
    st.caption(f" {source_info}")

    # Main parameters with better organization
    st.markdown("#### ⚙️ Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        top_n = st.slider(" Top-N table / CSV", 20, 200, 60, 5,
                          help="Number of top words to show in tables and CSV export")
        cloud_words_cap = st.slider(" Words used in each cloud", 60, 300, 180, 20,
                                    help="Maximum number of words to display in each word cloud")
    with c2:
        background = st.selectbox(" Background", ["white", "black"], index=0)
        seed = st.number_input("🎲 Word cloud seed (deterministic)", 0, 9999, 42, 1,
                               help="Random seed for reproducible word cloud layouts")
    with c3:
        if w2v_model is not None:
            max_vocab_for_similarity = st.slider(" Max vocab for similarity (per emotion)", 500, 20000, 5000, 500,
                                                 help="Limit vocabulary size for semantic similarity computation")
        else:
            st.info("Word2Vec parameters disabled (no model found)")

    # Enhanced filtering options
    st.markdown("####  Filtering Options")
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        limit_to_emotion_vocab = st.checkbox(" Use only tokens appearing in that emotion's reviews", value=True,
                                             help="Restrict to words that actually appear in each emotion's texts")
        min_freq_per_emotion = st.number_input(" Min frequency in emotion", 1, 100, 10, 1,
                                               help="Minimum times a word must appear in an emotion to be included")
    with col_filter2:
        global_min_df = st.number_input(
            " Global document frequency (min #reviews containing token)",
            1, 1000, 20, 1,
            help="Helps remove rare/garbled tokens. Higher values = cleaner results.",
        )

    # Advanced options with better organization
    with st.expander(" Advanced Word Cloud Options"):
        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            st.markdown("**Aspect Analysis**")
            use_aspect_analysis = st.checkbox(" Enable Aspect-Based Analysis", value=False)
            if use_aspect_analysis:
                aspects = st.multiselect(
                    "Select aspects to analyze",
                    ["taste", "price", "packaging", "delivery", "quality", "service", "texture", "aroma"],
                    default=["taste", "quality"],
                    help="Focus on specific product aspects"
                )

        with col_adv2:
            st.markdown("**Temporal Analysis**")
            use_temporal_analysis = st.checkbox(" Show emotion trends over time", value=False)
            if use_temporal_analysis and "Time" in df_clean.columns:
                time_granularity = st.selectbox("Time granularity", ["Month", "Quarter", "Year"])
            elif use_temporal_analysis:
                st.warning("No 'Time' column found in data")

    # Enhanced brand/product filtering
    st.markdown("####  Hide brand/product/common words (visual-only)")
    default_stoplist = "amazon, starbucks, folgers, keurig, nespresso, kitkat, trader, joes, walmart, costco, coffee, food"
    custom_stop = st.text_input("Comma-separated words to exclude", value=default_stoplist,
                                help="These words will be hidden from word clouds but not from analysis")
    HIDE = set(w.strip().lower() for w in custom_stop.split(",") if w.strip())

    if HIDE:
        st.caption(f" Hiding {len(HIDE)} words: {', '.join(sorted(list(HIDE)[:5]))}{'...' if len(HIDE) > 5 else ''}")

    # Enhanced VADER polarity filtering
    with st.expander(" Optional: Polarity gate (VADER) for Positive/Negative/Neutral", expanded=False):
        use_vader = st.checkbox(" Gate by VADER polarity", value=False,
                                help="Filter words by their sentiment polarity scores")
        if use_vader:
            st.info(" VADER filtering will restrict words to those matching each emotion's expected polarity")
            col_vader1, col_vader2, col_vader3 = st.columns(3)
            with col_vader1:
                pos_thresh = st.slider("Positive threshold (>=)", 0.1, 2.0, 0.5, 0.1)
            with col_vader2:
                neg_thresh = st.slider("Negative threshold (<=)", -2.0, -0.1, -0.5, 0.1)
            with col_vader3:
                neu_band = st.slider("Neutral band (|valence| ≤)", 0.1, 1.0, 0.2, 0.1)

    # Enhanced domain stopwords
    use_domain_stop = st.checkbox(" Apply domain stop-list", value=True,
                                  help="Remove common domain-specific words that don't add insight")
    DOMAIN_STOP = {
        "like", "taste", "product", "one", "would", "good", "great", "get", "make", "really", "time", "much", "also",
        "food", "coffee", "tea", "amazon", "buy", "use", "used", "got", "well", "bit", "little", "thing", "things",
        "even", "also", "back", "order", "will", "way", "first", "come", "new", "still"
    }
    DOMAIN_STOP_TUPLE = tuple(sorted(DOMAIN_STOP))  # cache-friendly

    if use_domain_stop:
        st.caption(f" Domain stoplist: {', '.join(sorted(list(DOMAIN_STOP)[:10]))}... ({len(DOMAIN_STOP)} total)")

    TOKEN_RE = re.compile(r"^[a-z]+$")

    def token_ok(w: str) -> bool:
        return 3 <= len(w) <= 15 and TOKEN_RE.match(w) is not None and (
            w not in DOMAIN_STOP if use_domain_stop else True)

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

    # Build document frequency with progress
    with st.spinner("Computing document frequencies..."):
        GLOBAL_DF = build_global_docfreq_from_texts(
            df_counts_source["clean_text"].astype(str).tolist(), use_domain_stop, DOMAIN_STOP_TUPLE
        )

    # Enhanced vocabulary info
    vocab_info = f"🔍 Found {len(GLOBAL_DF):,} unique tokens"
    filtered_vocab = {k: v for k, v in GLOBAL_DF.items() if v >= global_min_df}
    vocab_info += f" → {len(filtered_vocab):,} after min frequency filter (≥{global_min_df})"
    st.info(vocab_info)

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
            return set(), set(), set()

    POS_SET, NEG_SET, NEU_SET = (set(), set(), set())
    if use_vader:
        POS_SET, NEG_SET, NEU_SET = vader_sets(pos_thresh, neg_thresh, neu_band)
        st.success(f"🎭 VADER sets loaded: Positive({len(POS_SET)}), Negative({len(NEG_SET)}), Neutral({len(NEU_SET)})")

    # ---- Aspect-based analysis helper
    def extract_aspect_sentences(text, aspect_keywords):
        """Extract sentences containing aspect keywords"""
        sentences = text.split('.')
        aspect_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in aspect_keywords):
                aspect_sentences.append(sentence.strip())
        return ' '.join(aspect_sentences)

    # ---- Per-emotion filtered counts with enhanced reporting
    def get_emotion_tokens(emo: str) -> list[str]:
        docs = df_counts_source.loc[df_counts_source["Emotion"] == emo, "clean_text"].astype(str)
        return [t for doc in docs for t in doc.split()]

    st.markdown("---")
    st.markdown("####  Computing emotion-specific token counts...")

    emo_counts: dict[str, Counter] = {}
    total_valid_tokens = 0

    for emo in EMOTIONS:
        toks = [t for t in get_emotion_tokens(emo) if token_ok(t) and GLOBAL_DF.get(t, 0) >= global_min_df]
        emo_counts[emo] = Counter(toks)
        total_valid_tokens += sum(emo_counts[emo].values())

        unique_tokens = len(emo_counts[emo])
        total_tokens = sum(emo_counts[emo].values())

        # Enhanced reporting with color coding
        if unique_tokens < 50:
            st.error(f"• **{emo}**: ⚠️ Only {unique_tokens:,} unique tokens - may produce poor word clouds")
        elif unique_tokens < 200:
            st.warning(
                f"• **{emo}**: ⚠️ {unique_tokens:,} unique tokens from {total_tokens:,} total - moderate vocabulary")
        else:
            st.success(f"• **{emo}**: ✅ {unique_tokens:,} unique tokens from {total_tokens:,} total - good vocabulary")

    if total_valid_tokens == 0:
        st.error("❌ No valid tokens found for any emotion. Try reducing filtering parameters.")
        st.stop()

    # ---- Enhanced Helper Functions ----
    def render_cloud(freqs: dict[str, float], title: str, cmap: str):
        freqs = {w: v for w, v in freqs.items() if w.lower() not in HIDE and v > 0}
        if not freqs:
            st.info(f"ℹ️ No terms to display for **{title}** after filtering.")
            return

        try:
            # Enhanced word cloud with better styling
            wc = WordCloud(
                width=1200, height=600, background_color=background,
                collocations=False, colormap=cmap, random_state=int(seed),
                max_words=min(len(freqs), cloud_words_cap),
                relative_scaling=0.6, min_font_size=12,
                max_font_size=100, prefer_horizontal=0.8
            ).generate_from_frequencies(freqs)

            fig, ax = plt.subplots(figsize=(14, 7), dpi=120)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontsize=18, fontweight='bold', pad=25)

            # Add subtitle with word count
            ax.text(0.5, -0.05, f"({len(freqs)} words)", transform=ax.transAxes,
                    ha='center', fontsize=12, style='italic')

            st.pyplot(fig)

            # Enhanced download with better filename
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2, dpi=150)
            plt.close(fig)  # Important: close to prevent memory leaks

            emotion_name = title.split(' —')[0]
            filename = f"wordcloud_{emotion_name.lower()}_{method.split('(')[0].strip().replace(' ', '_')}.png"

            st.download_button(
                label=f" Download '{emotion_name}' PNG",
                data=buf.getvalue(),
                file_name=filename,
                mime="image/png",
            )
        except Exception as e:
            st.error(f"❌ Error creating word cloud for {title}: {e}")

    def shift_to_nonnegative(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
        if not items:
            return items
        m = min(v for _, v in items)
        return items if m >= 0 else [(w, v - m + 0.001) for w, v in items]  # Add small constant

    def top_table(freqs: dict[str, float], k: int) -> pd.DataFrame:
        top_items = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:k]
        df = pd.DataFrame(top_items, columns=["Word", "Weight"])
        df.index = range(1, len(df) + 1)  # 1-based indexing
        return df

    # Vectorized cosine similarity with error handling
    def top_by_centroid_similarity(centroid_vec: np.ndarray, candidate_words: list[str], k: int):
        if not candidate_words:
            return []
        try:
            mat = w2v_model.wv[candidate_words]  # shape (n, d)
            mat_norm = np.linalg.norm(mat, axis=1) + 1e-12
            c = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-12)
            sims = (mat @ c) / mat_norm
            order = np.argsort(-sims)[:k]
            return [(candidate_words[i], float(sims[i])) for i in order]
        except Exception as e:
            st.warning(f"⚠️ Error in similarity computation: {e}")
            return []

    # ---- Compute per-emotion weights by method with enhanced progress ----
    st.markdown("---")
    st.markdown(f"#### 🔬 Computing weights using: **{method}**")

    weights_per_emotion: dict[str, dict[str, float]] = {}

    # A) Word2Vec centroid (semantic)
    if method.startswith("Word2Vec centroid (semantic)"):
        if w2v_model is None:
            st.error("❌ Word2Vec model not found. Choose another method or train embeddings first.")
            st.stop()

        st.info(" Computing cosine similarity to each emotion's unweighted Word2Vec centroid (semantic).")

        # Enhanced centroid computation with validation
        centroids = {}
        for emo in EMOTIONS:
            toks_counts = [(w, c) for w, c in emo_counts[emo].items() if w in WV_KEYS]
            if not toks_counts:
                centroids[emo] = None
                st.error(f"❌ No valid tokens found for {emo} emotion in Word2Vec vocabulary")
                continue

            st.info(f"🔍 {emo}: Using {len(toks_counts)} words for centroid computation")
            mat = np.vstack([w2v_model.wv[w] for w, _ in toks_counts]).astype("float32")
            ws = np.array([c for _, c in toks_counts], dtype="float32").reshape(-1, 1)
            centroids[emo] = (mat * ws).sum(axis=0) / (ws.sum() + 1e-12)

        for emo in EMOTIONS:
            c = centroids[emo]
            if c is None:
                weights_per_emotion[emo] = {}
                continue
            if limit_to_emotion_vocab:
                allow = [w for w, cnt in emo_counts[emo].items() if
                         cnt >= min_freq_per_emotion and w in WV_KEYS and token_ok(w)]
            else:
                allow = [w for w in w2v_model.wv.key_to_index if token_ok(w)]
            allow = allow[:max_vocab_for_similarity]

            if not allow:
                st.warning(f"⚠️ No candidate words found for {emo} after filtering")
                weights_per_emotion[emo] = {}
                continue

            top_items = top_by_centroid_similarity(c, allow, cloud_words_cap)
            top_items = shift_to_nonnegative(top_items)
            weights_per_emotion[emo] = dict(top_items)

    # B) Word2Vec centroid (SIF)
    elif method.startswith("Word2Vec centroid (SIF)"):
        if w2v_model is None:
            st.error("❌ Word2Vec model not found. Train embeddings first.")
            st.stop()

        st.info(" Computing cosine similarity to **SIF-weighted** centroids (Word2Vec-only).")

        # Enhanced SIF computation with validation
        a = 1e-3
        if use_train_only_for_counts and tr_idx is not None:
            train_docs = df_clean.loc[tr_idx, "clean_text"].astype(str).tolist()
        else:
            train_docs = df_counts_source["clean_text"].astype(str).tolist()

        train_freq = Counter(t for d in train_docs for t in d.split())
        total_train_tokens = sum(train_freq.values())

        st.info(f" SIF computed from {len(train_docs):,} documents with {total_train_tokens:,} total tokens")

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
                allow = [w for w, cnt in emo_counts[emo].items() if
                         cnt >= min_freq_per_emotion and w in WV_KEYS and token_ok(w)]
            else:
                allow = [w for w in w2v_model.wv.key_to_index if token_ok(w)]
            allow = allow[:max_vocab_for_similarity]

            if not allow:
                st.warning(f"⚠️ No candidate words found for {emo} after filtering")
                weights_per_emotion[emo] = {}
                continue

            top_items = top_by_centroid_similarity(c, allow, cloud_words_cap)
            top_items = shift_to_nonnegative(top_items)
            weights_per_emotion[emo] = dict(top_items)

    # C) Contrastive log-odds (counts) - Enhanced with better validation
    else:
        st.info(" Computing contrastive log-odds with +1 smoothing (discriminative vs other emotions).")
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

            if in_total == 0 or out_total == 0:
                st.error(f"❌ No valid counts for {emo} emotion")
                weights_per_emotion[emo] = {}
                continue

            allowed = set(in_counts.keys()) if limit_to_emotion_vocab else set(global_counts.keys())
            allowed = {w for w in allowed if in_counts.get(w, 0) >= min_freq_per_emotion}

            if not allowed:
                st.warning(f"⚠️ No words meet frequency threshold for {emo}")
                weights_per_emotion[emo] = {}
                continue

            scores = {}
            for w in allowed:
                a_ = in_counts[w] + 1
                b_ = (in_total - in_counts[w]) + 1
                c_ = max(out_counts[w], 0) + 1  # Ensure non-negative
                d_ = (out_total - max(out_counts[w], 0)) + 1
                scores[w] = float(np.log(a_ / b_) - np.log(c_ / d_))

            # Optional polarity gating with reporting
            if use_vader:
                original_count = len(scores)
                if emo == "Positive" and POS_SET:
                    scores = {w: s for w, s in scores.items() if w in POS_SET}
                elif emo == "Negative" and NEG_SET:
                    scores = {w: s for w, s in scores.items() if w in NEG_SET}
                elif emo == "Neutral" and NEU_SET:
                    scores = {w: s for w, s in scores.items() if w in NEU_SET}

                filtered_count = len(scores)
                if filtered_count < original_count:
                    st.info(
                        f" {emo}: VADER filtering reduced vocabulary from {original_count} to {filtered_count} words")

            top_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:cloud_words_cap]
            top_items = shift_to_nonnegative(top_items)
            weights_per_emotion[emo] = dict(top_items)

    # ---- Enhanced Rendering Section ----
    st.markdown("---")
    st.markdown("###  Emotion-Specific Word Clouds")

    # Check if we have valid results with better error reporting
    total_words = sum(len(weights) for weights in weights_per_emotion.values())
    if total_words == 0:
        st.error("❌ No words found for any emotion. Try adjusting your filtering parameters:")
        st.write("• Reduce min_freq_per_emotion")
        st.write("• Reduce global_min_df")
        st.write("• Disable domain stoplist")
        st.write("• Check if Word2Vec model has sufficient vocabulary")
        st.stop()

    st.success(f"✅ Generated {total_words} total weighted words across {len(EMOTIONS)} emotions")

    # Enhanced rendering with better layout
    if len(EMOTIONS) == 3:
        cols = st.columns(3)
    elif len(EMOTIONS) == 2:
        cols = st.columns(2)
    else:
        cols = st.columns(len(EMOTIONS))

    combined_tables = []

    for col, emo in zip(cols, EMOTIONS):
        with col:
            title = f"{emo} — {method.split('(')[0].strip()}"
            freqs = weights_per_emotion.get(emo, {})

            if freqs:
                # Show word count before rendering
                st.markdown(f"**🎯 {emo} ({len(freqs)} words)**")

                render_cloud(freqs, title, EMO2CMAP.get(emo, "viridis"))

                # Enhanced table with ranking
                top_df = top_table(freqs, min(top_n, len(freqs)))
                st.markdown(f"**📊 Top {len(top_df)} words:**")

                # Color-code the table based on emotion
                if emo == "Positive":
                    st.dataframe(top_df, use_container_width=True)
                elif emo == "Negative":
                    st.dataframe(top_df, use_container_width=True)
                else:  # Neutral
                    st.dataframe(top_df, use_container_width=True)

                # add emotion column for combined export
                if not top_df.empty:
                    tdf = top_df.copy()
                    tdf.insert(0, "Emotion", emo)
                    tdf.insert(1, "Rank", range(1, len(tdf) + 1))
                    combined_tables.append(tdf)

                # Show sample high-weight words
                if len(freqs) >= 5:
                    top_5_words = list(freqs.keys())[:5]
                    st.caption(f"🔝 **Top words:** {', '.join(top_5_words)}")
            else:
                st.error(f"❌ No words found for {emo}")
                st.info("Try reducing filtering parameters for this emotion")

    # Enhanced CSV download and statistics
    if combined_tables:
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        all_df = pd.concat(combined_tables, ignore_index=True)

        # Enhanced download with method info in filename
        method_short = method.split('(')[0].strip().replace(' ', '_').lower()
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        filename = f"wordcloud_analysis_{method_short}_{timestamp}.csv"

        st.download_button(
            " Download Complete Analysis (CSV)",
            data=all_df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            help="Download all emotions and rankings in one CSV file"
        )

        # Enhanced statistics with insights
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        with col_stats1:
            st.metric("Total words", len(all_df))
        with col_stats2:
            st.metric("Avg weight", f"{all_df['Weight'].mean():.4f}")
        with col_stats3:
            st.metric("Unique words", all_df['Word'].nunique())
        with col_stats4:
            overlap_pct = (1 - all_df['Word'].nunique() / len(all_df)) * 100
            st.metric("Word overlap", f"{overlap_pct:.1f}%")

        # Word overlap analysis
        st.markdown("####  Cross-Emotion Word Analysis")

        word_counts = all_df['Word'].value_counts()
        overlapping_words = word_counts[word_counts > 1]

        if len(overlapping_words) > 0:
            st.warning(f"⚠️ **{len(overlapping_words)} words appear in multiple emotions**")

            with st.expander("View overlapping words"):
                overlap_df = overlapping_words.head(20).reset_index()
                overlap_df.columns = ['Word', 'Emotions_Count']

                # Show which emotions each word appears in
                for _, row in overlap_df.iterrows():
                    word = row['Word']
                    emotions_with_word = all_df[all_df['Word'] == word]['Emotion'].tolist()
                    st.write(f"• **{word}**: {', '.join(emotions_with_word)}")
        else:
            st.success("✅ **No word overlap** - Each word is unique to its emotion!")

        # Top discriminative words
        unique_words = word_counts[word_counts == 1].index.tolist()
        if unique_words:
            st.markdown("####  Most Discriminative Words (Emotion-Specific)")

            for emo in EMOTIONS:
                emo_unique = all_df[(all_df['Emotion'] == emo) & (all_df['Word'].isin(unique_words))]
                if not emo_unique.empty:
                    top_unique = emo_unique.head(5)['Word'].tolist()
                    st.write(f"**{emo}**: {', '.join(top_unique)}")

    # Enhanced interpretation and insights
    st.markdown("---")
    st.markdown("### 💡 Interpretation Guide")

    col_interp1, col_interp2 = st.columns(2)
    with col_interp1:
        st.markdown("""
        **Emotion Characteristics:**
        - 😊 **Positive**: praise, quality adjectives (*delicious, amazing, wonderful, excellent*)
        - 😞 **Negative**: complaints, defects (*awful, rancid, disappointing, terrible*)
        - 😐 **Neutral**: transactional, descriptive (*package, shipped, ingredients, received*)
        """)

    with col_interp2:
        st.markdown("""
        **Method Differences:**
        -  **Semantic**: Groups semantically similar words
        -  **SIF**: Down-weights very common words
        -  **Log-odds**: Emphasizes discriminative words
        """)

    # Method-specific insights with enhanced explanations
    st.markdown("####  Method-Specific Insights")

    if "semantic" in method.lower():
        st.info(
            " **Semantic method**: Words are grouped by meaning similarity. Expect coherent semantic clusters and related concepts.")
        st.write(" **What to look for**: Synonyms, related concepts, semantic fields")
    elif "sif" in method.lower():
        st.info(
            " **SIF method**: Common words are down-weighted. Expect more nuanced, less frequent terms that are semantically meaningful.")
        st.write(" **What to look for**: Sophisticated vocabulary, domain-specific terms, descriptive adjectives")
    else:
        st.info(
            " **Log-odds method**: Shows words that are statistically distinctive for each emotion compared to others.")
        st.write(" **What to look for**: Unique markers, discriminative terms, emotion-specific vocabulary")

    # Quality assessment
    st.markdown("#### ✅ Quality Assessment")

    quality_checks = []

    # Check 1: Vocabulary diversity
    total_unique_words = all_df['Word'].nunique() if combined_tables else 0
    if total_unique_words > 100:
        quality_checks.append("✅ **Good vocabulary diversity** (>100 unique words)")
    elif total_unique_words > 50:
        quality_checks.append("⚠️ **Moderate vocabulary diversity** (50-100 unique words)")
    else:
        quality_checks.append("❌ **Low vocabulary diversity** (<50 unique words)")

    # Check 2: Emotion balance
    if combined_tables:
        emotion_word_counts = all_df.groupby('Emotion').size()
        min_words = emotion_word_counts.min()
        max_words = emotion_word_counts.max()
        balance_ratio = min_words / max_words if max_words > 0 else 0

        if balance_ratio > 0.7:
            quality_checks.append("✅ **Well-balanced emotions** (similar word counts)")
        elif balance_ratio > 0.4:
            quality_checks.append("⚠️ **Moderately balanced emotions**")
        else:
            quality_checks.append("❌ **Imbalanced emotions** (large word count differences)")

    # Check 3: Word overlap
    if combined_tables:
        overlap_rate = (len(all_df) - all_df['Word'].nunique()) / len(all_df) if len(all_df) > 0 else 0
        if overlap_rate < 0.2:
            quality_checks.append("✅ **Low word overlap** (emotions well-separated)")
        elif overlap_rate < 0.4:
            quality_checks.append("⚠️ **Moderate word overlap**")
        else:
            quality_checks.append("❌ **High word overlap** (emotions poorly separated)")

    for check in quality_checks:
        st.write(check)

    # Recommendations based on results
    st.markdown("####  Recommendations")

    recommendations = []

    if total_unique_words < 50:
        recommendations.append("• **Reduce filtering**: Lower min_freq_per_emotion or global_min_df")
        recommendations.append("• **Check preprocessing**: Ensure important words aren't being removed")

    if combined_tables and overlap_rate > 0.4:
        recommendations.append("• **Try different method**: Log-odds typically gives better separation")
        recommendations.append("• **Increase frequency thresholds**: Focus on more distinctive words")

    if len([emo for emo in EMOTIONS if len(weights_per_emotion.get(emo, {})) < 20]) > 0:
        recommendations.append("• **Check class balance**: Some emotions may have too few examples")
        recommendations.append("• **Verify preprocessing**: Ensure emotion-specific vocabulary is preserved")

    if not recommendations:
        recommendations.append("✅ **Results look good!** The word clouds will provide meaningful insights.")

    for rec in recommendations:
        st.write(rec)

    # Final success message with next steps
    st.markdown("---")
    if combined_tables and len(combined_tables) == len(EMOTIONS):
        st.success(" **Word clouds generated successfully!** Use the download buttons to save images and data.")

        # Check if models are trained for next step guidance
        if models_trained:
            st.info(
                " **Next step:** Go to **Prediction Page** to test your models on new reviews and see how the word clouds relate to actual predictions.")
        else:
            st.info(
                " **Recommended:** Go to **Modeling & Results** to train classification models, then test predictions to validate these word cloud insights.")
    else:
        st.warning("⚠️ **Partial success:** Some emotions may need parameter adjustment for better results.")

    # Debug information (if needed)
    with st.expander(" Debug Information"):
        st.write("**Configuration Summary:**")
        st.write(f"• Method: {method}")
        st.write(f"• Use train only: {use_train_only_for_counts}")
        st.write(f"• Data source: {len(df_counts_source):,} reviews")
        st.write(f"• Global vocabulary: {len(GLOBAL_DF):,} tokens")
        st.write(f"• Filtered vocabulary: {len(filtered_vocab):,} tokens")
        st.write(f"• Emotions found: {EMOTIONS}")

        if w2v_model:
            st.write(f"• Word2Vec vocabulary: {len(WV_KEYS):,} words")

        st.write("**Per-emotion results:**")
        for emo in EMOTIONS:
            word_count = len(weights_per_emotion.get(emo, {}))
            st.write(f"• {emo}: {word_count} words generated")


def page_predictions():
    """
    Enhanced Prediction Interface
    Predict emotion for new reviews (single text or uploaded CSV/TXT).
    Uses Word2Vec embeddings + the trained model saved in session_state.
    """

    st.title(" Emotion Predictions")
    st.caption("Test your trained models on new reviews - get instant emotion analysis with confidence scores")

    # ---- Enhanced Dependency Checks ----
    w2v = st.session_state.get("w2v_model")
    label_map = st.session_state.get("label_map")
    rf_model = st.session_state.get("rf_model")
    xgb_model = st.session_state.get("xgb_model")
    ensemble_model = st.session_state.get("ensemble_model")

    if w2v is None or label_map is None:
        st.error("❌ **Word2Vec and label map not found.** Please run **Embeddings (Word2Vec)** first.")
        st.info(" The prediction system needs Word2Vec embeddings to convert text into numerical vectors.")
        st.stop()

    if rf_model is None and xgb_model is None and ensemble_model is None:
        st.error("❌ **No trained model found.** Please run **Modeling & Results** first.")
        st.info(" You need at least one trained classifier (Random Forest, XGBoost, or Ensemble) to make predictions.")
        st.stop()

    idx2lbl = {v: k for k, v in label_map.items()}
    ordered_labels = [idx2lbl[i] for i in range(len(idx2lbl))]

    # Enhanced emoji and styling
    EMOJI = {"Negative": "😞", "Neutral": "😐", "Positive": "😀"}
    COLOR_MAP = {"Positive": "#28a745", "Negative": "#dc3545", "Neutral": "#6c757d"}
    IMAGE_PATHS = {
        "Negative": "Negative.jpg",
        "Neutral": "Neutral.jpg",
        "Positive": "Positive.jpg",
    }

    # ---- Enhanced Model Performance Display ----
    st.markdown("###  Model Performance Overview")

    def get_model_metrics(model_name: str) -> dict:
        """Get comprehensive metrics for a model"""
        res = st.session_state.get("results")
        if res and "metrics" in res:
            df = pd.DataFrame(res["metrics"])
            row = df.loc[df["Model"] == model_name]
            if not row.empty:
                return {
                    'accuracy': row["Test_Accuracy"].values[0] if "Test_Accuracy" in row.columns else None,
                    'f1': row["Test_F1"].values[0] if "Test_F1" in row.columns else None,
                    'precision': row["Test_Precision"].values[0] if "Test_Precision" in row.columns else None,
                    'recall': row["Test_Recall"].values[0] if "Test_Recall" in row.columns else None
                }
        return {}

    # Model selection with enhanced info
    st.markdown("####  Model Selection")
    model_options = []
    model_metrics = {}

    if rf_model is not None:
        model_options.append("Random Forest")
        model_metrics["Random Forest"] = get_model_metrics("Random Forest")
    if xgb_model is not None:
        model_options.append("XGBoost")
        model_metrics["XGBoost"] = get_model_metrics("XGBoost")
    if ensemble_model is not None:
        model_options.append("Ensemble (RF+XGB)")
        model_metrics["Ensemble (RF+XGB)"] = get_model_metrics("Ensemble (RF+XGB)")

    # Enhanced model selection with performance info
    col_model_select, col_model_info = st.columns([1, 2])

    with col_model_select:
        model_choice = st.radio("Choose model:", model_options, horizontal=False)

    with col_model_info:
        if model_choice in model_metrics and model_metrics[model_choice]:
            metrics = model_metrics[model_choice]
            st.markdown(f"**{model_choice} Performance:**")

            # Create metrics display
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                acc = metrics.get('accuracy')
                if acc is not None:
                    st.metric("Accuracy", f"{acc:.3f}")
            with col_m2:
                f1 = metrics.get('f1')
                if f1 is not None:
                    st.metric("F1-Score", f"{f1:.3f}")
            with col_m3:
                prec = metrics.get('precision')
                if prec is not None:
                    st.metric("Precision", f"{prec:.3f}")
            with col_m4:
                rec = metrics.get('recall')
                if rec is not None:
                    st.metric("Recall", f"{rec:.3f}")
        else:
            st.info(" Model metrics will appear here after training")

    # Initialize prediction history for monitoring
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    # Enhanced session metrics
    col_session1, col_session2, col_session3 = st.columns(3)

    with col_session1:
        if len(st.session_state.prediction_history) > 0:
            recent_predictions = st.session_state.prediction_history[-10:]
            avg_conf = np.mean([p['confidence'] for p in recent_predictions])
            st.metric("Session Avg Confidence", f"{avg_conf:.3f}")
        else:
            st.metric("Predictions Made", "0")

    with col_session2:
        if len(st.session_state.prediction_history) > 0:
            emotions = [p['prediction'] for p in st.session_state.prediction_history]
            most_common = max(set(emotions), key=emotions.count)
            st.metric("Most Predicted", most_common)
        else:
            st.metric("Most Predicted", "None")

    with col_session3:
        vocab_size = len(w2v.wv.key_to_index) if w2v else 0
        st.metric("Vocabulary Size", f"{vocab_size:,}")

    # ---- Enhanced Text Processing Functions ----
    preprocess_text_fn = st.session_state.get("preprocess_text")
    stop_set = st.session_state.get("preproc_stopwords", set())

    def _fallback_clean(s: str) -> str:
        """Enhanced fallback text cleaning"""
        s = str(s).lower()
        s = re.sub(r"<.*?>", " ", s)  # Remove HTML
        s = re.sub(r"[^\w\s]", " ", s)  # Remove punctuation
        s = re.sub(r"\d+", " ", s)  # Remove numbers
        s = re.sub(r"\s+", " ", s).strip()  # Collapse whitespace
        toks = [t for t in s.split() if len(t) > 1 and t not in stop_set]
        return " ".join(toks)

    def clean_text(raw: str) -> str:
        """Clean text using saved preprocessing function or fallback"""
        if callable(preprocess_text_fn):
            try:
                return preprocess_text_fn(raw, stop_set)
            except (TypeError, AttributeError):
                return _fallback_clean(raw)
        return _fallback_clean(raw)

    def doc_vec(tokens: list[str]) -> np.ndarray:
        """Convert tokens to document vector using Word2Vec"""
        vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
        if vecs:
            return np.mean(vecs, axis=0).astype("float32")
        else:
            return np.zeros(w2v.vector_size, dtype="float32")

    def vectorize_texts(texts: list[str]) -> np.ndarray:
        """Convert list of texts to feature matrix"""
        toks_list = [clean_text(t).split() for t in texts]
        return np.vstack([doc_vec(toks) for toks in toks_list])

    # Prediction function
    def predict_proba(X: np.ndarray) -> np.ndarray:
        if model_choice == "XGBoost":
            return xgb_model.predict_proba(X)
        elif model_choice == "Random Forest":
            return rf_model.predict_proba(X)
        else:  # Ensemble
            return ensemble_model.predict_proba(X)

    # ---- Enhanced Single Prediction Interface ----
    st.markdown("---")
    st.markdown("### 💬 Analyze Your Review")

    # Enhanced text input with smart examples
    txt = st.text_area(
        "Enter a review to analyze:",
        height=120,
        placeholder="Type your review here... The model will analyze the emotional sentiment.",
        help="Enter any product review or opinion text. The model works best on food-related content."
    )

    # Smart example system with actual prediction capability
    st.markdown("####  Try These Examples")

    example_texts = {
        " Strong Positive": "This coffee is absolutely delicious and amazing! The taste is wonderful and the aroma is fantastic. I love it so much and highly recommend this excellent product to everyone!",
        " Strong Negative": "This coffee tastes terrible and awful. Very disappointing quality and horrible flavor. The worst coffee I have ever tried. Complete waste of money and would never buy again.",
        " Neutral": "Coffee arrived yesterday. Package was sealed properly. Standard delivery time. Product as described in listing. No major issues.",
        " Mixed Sentiment": "The coffee taste is okay but the price is too expensive. Packaging was good though delivery took longer than expected."
    }

    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        for i, (label, text) in enumerate(list(example_texts.items())[:2]):
            if st.button(label, key=f"ex_{i}"):
                txt = text
                st.experimental_rerun()

    with col_ex2:
        for i, (label, text) in enumerate(list(example_texts.items())[2:], 2):
            if st.button(label, key=f"ex_{i}"):
                txt = text
                st.experimental_rerun()

    # Enhanced prediction interface
    col_pred1, col_pred2 = st.columns([1, 2])

    with col_pred1:
        predict_btn = st.button(" Analyze Emotion", type="primary", use_container_width=True)

    with col_pred2:
        if txt.strip():
            # Enhanced text analysis preview
            cleaned_text = clean_text(txt)
            tokens = cleaned_text.split()
            in_vocab = sum(1 for t in tokens if t in w2v.wv)

            st.markdown("**Text Analysis Preview:**")

            # Word count and complexity
            word_count = len(tokens)
            char_count = len(cleaned_text)

            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Words", word_count)
            with col_stats2:
                st.metric("Characters", char_count)
            with col_stats3:
                if tokens:
                    coverage = in_vocab / len(tokens)
                    st.metric("Vocab Coverage", f"{coverage:.1%}")

            # Coverage assessment with recommendations
            if tokens:
                if coverage < 0.3:
                    st.error(f"⚠️ **Very low vocabulary coverage** ({coverage:.1%})")
                    st.info("💡 **Tip:** Use more common words or longer text for better predictions")
                elif coverage < 0.5:
                    st.warning(f"⚠️ **Moderate vocabulary coverage** ({coverage:.1%})")
                    st.info("💡 **Suggestion:** Add more descriptive words for improved accuracy")
                else:
                    st.success(f"✅ **Good vocabulary coverage** ({coverage:.1%})")

    # ---- Enhanced Prediction Results ----
    if predict_btn:
        if not txt.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            try:
                with st.spinner(" Analyzing emotion..."):
                    # Enhanced prediction process
                    cleaned_text = clean_text(txt)
                    tokens = cleaned_text.split()

                    X = vectorize_texts([txt])

                    # Check for zero vector issue
                    if np.allclose(X, 0):
                        st.error("⚠️ **Warning:** No recognizable words found in vocabulary.")
                        st.info("💡 **Solutions:**")
                        st.write("• Try using more common words")
                        st.write("• Add more descriptive language")
                        st.write("• Check if text relates to food/product reviews")
                        return

                    p = predict_proba(X)[0]
                    pred_idx = int(np.argmax(p))
                    pred_lbl = idx2lbl[pred_idx]
                    conf = float(p[pred_idx])

                    # Store in enhanced prediction history
                    st.session_state.prediction_history.append({
                        'text': txt[:50] + "..." if len(txt) > 50 else txt,
                        'prediction': pred_lbl,
                        'confidence': conf,
                        'model': model_choice,
                        'timestamp': pd.Timestamp.now(),
                        'word_count': len(tokens),
                        'vocab_coverage': sum(1 for t in tokens if t in w2v.wv) / len(tokens) if tokens else 0
                    })

                # ---- Enhanced Results Display ----
                st.markdown("---")
                st.markdown("###  Prediction Results")

                col_result1, col_result2 = st.columns([2, 1])

                with col_result1:
                    # Main prediction with enhanced styling
                    emoji = EMOJI.get(pred_lbl, "🤔")
                    color = COLOR_MAP.get(pred_lbl, "#007bff")

                    # Dynamic result styling based on prediction
                    if pred_lbl == "Positive":
                        st.success(f"**Predicted Emotion:** {emoji} **{pred_lbl}**")
                    elif pred_lbl == "Negative":
                        st.error(f"**Predicted Emotion:** {emoji} **{pred_lbl}**")
                    else:
                        st.info(f"**Predicted Emotion:** {emoji} **{pred_lbl}**")

                    # Enhanced confidence display
                    st.markdown(f"**Confidence Score:** {conf:.1%}")

                    # Confidence bar visualization
                    conf_bar_html = f"""
                    <div style="background-color: #f0f0f0; border-radius: 10px; padding: 2px;">
                        <div style="background-color: {color}; width: {conf * 100:.1f}%; height: 20px; border-radius: 8px; text-align: center; line-height: 20px; color: white; font-weight: bold;">
                            {conf:.1%}
                        </div>
                    </div>
                    """
                    st.markdown(conf_bar_html, unsafe_allow_html=True)

                    # Enhanced probability breakdown
                    st.markdown("####  Detailed Probability Breakdown")
                    pairs = [(lbl, float(p[label_map[lbl]])) for lbl in ordered_labels]
                    pairs.sort(key=lambda t: t[1], reverse=True)

                    for i, (lbl, prob) in enumerate(pairs):
                        # Enhanced visualization with better bars
                        bar_width = max(5, int(prob * 100))  # Minimum 5% width for visibility

                        emotion_color = COLOR_MAP.get(lbl, "#007bff")
                        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "▫️"

                        # Create enhanced probability bar
                        prob_bar_html = f"""
                        <div style="margin: 5px 0;">
                            <div style="display: flex; align-items: center; margin-bottom: 2px;">
                                <span>{rank_emoji} <strong>{lbl}</strong>: {prob:.1%}</span>
                            </div>
                            <div style="background-color: #f0f0f0; border-radius: 5px; height: 15px;">
                                <div style="background-color: {emotion_color}; width: {bar_width}%; height: 15px; border-radius: 5px;"></div>
                            </div>
                        </div>
                        """
                        st.markdown(prob_bar_html, unsafe_allow_html=True)

                with col_result2:
                    # Enhanced emotion visualization
                    img_path = IMAGE_PATHS.get(pred_lbl)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"{pred_lbl} Emotion", width=200)
                    else:
                        # Enhanced placeholder with better styling
                        color = COLOR_MAP.get(pred_lbl, "#007bff")
                        placeholder_html = f"""
                        <div style="
                            width: 200px; 
                            height: 180px; 
                            background: linear-gradient(135deg, {color} 0%, {color}CC 100%); 
                            display: flex; 
                            flex-direction: column;
                            align-items: center; 
                            justify-content: center; 
                            border-radius: 20px;
                            margin: 0 auto;
                            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
                            border: 3px solid white;
                        ">
                            <span style="font-size: 72px; margin-bottom: 10px;">{emoji}</span>
                            <span style="color: white; font-weight: bold; font-size: 16px;">{pred_lbl}</span>
                            <span style="color: white; font-size: 14px; opacity: 0.9;">{conf:.1%} confidence</span>
                        </div>
                        """
                        st.markdown(placeholder_html, unsafe_allow_html=True)

                # ---- Enhanced Analysis Insights ----
                st.markdown("####  Analysis Insights")

                # Sentiment word detection with enhanced categories
                positive_words = {
                    "strong": ["excellent", "amazing", "wonderful", "fantastic", "incredible", "outstanding", "superb"],
                    "moderate": ["good", "great", "nice", "pleasant", "satisfying", "decent", "fine"],
                    "emotional": ["love", "adore", "enjoy", "appreciate", "perfect", "awesome", "delicious"]
                }

                negative_words = {
                    "strong": ["terrible", "awful", "horrible", "disgusting", "appalling", "dreadful", "atrocious"],
                    "moderate": ["bad", "poor", "disappointing", "unsatisfactory", "mediocre", "subpar"],
                    "emotional": ["hate", "despise", "worst", "horrible", "regret", "waste", "useless"]
                }

                found_positive = {}
                found_negative = {}

                for category, words in positive_words.items():
                    found = [w for w in words if w in cleaned_text.lower()]
                    if found:
                        found_positive[category] = found

                for category, words in negative_words.items():
                    found = [w for w in words if w in cleaned_text.lower()]
                    if found:
                        found_negative[category] = found

                # Enhanced insight display
                col_insight1, col_insight2 = st.columns(2)

                with col_insight1:
                    if found_positive:
                        st.markdown("** Positive Indicators:**")
                        for category, words in found_positive.items():
                            st.write(f"• **{category.title()}**: {', '.join(words)}")

                with col_insight2:
                    if found_negative:
                        st.markdown("** Negative Indicators:**")
                        for category, words in found_negative.items():
                            st.write(f"• **{category.title()}**: {', '.join(words)}")

                # Overall sentiment interpretation
                total_pos = sum(len(words) for words in found_positive.values())
                total_neg = sum(len(words) for words in found_negative.values())

                if total_pos == 0 and total_neg == 0:
                    st.info(" **Neutral language detected** - No strong emotional indicators found")
                elif total_pos > total_neg:
                    st.success(f" **Text leans positive** - {total_pos} positive vs {total_neg} negative indicators")
                elif total_neg > total_pos:
                    st.warning(f"⚠️ **Text leans negative** - {total_neg} negative vs {total_pos} positive indicators")
                else:
                    st.info("⚖️ **Mixed sentiment** - Equal positive and negative indicators")

                # Enhanced confidence interpretation
                st.markdown("####  Confidence Assessment")
                if conf >= 0.8:
                    st.success(" **High confidence** - The model is very sure about this classification")
                    st.info("✅ **Recommendation:** Trust this prediction")
                elif conf >= 0.6:
                    st.info("✅ **Moderate confidence** - The model is reasonably confident")
                    st.info("💭 **Recommendation:** Good prediction, consider context")
                elif conf >= 0.4:
                    st.warning(" **Low confidence** - The model is uncertain")
                    st.info(" **Recommendation:** Review all probabilities carefully")
                else:
                    st.error("❌ **Very low confidence** - The model is very uncertain")
                    st.info(" **Recommendation:** Try rephrasing or adding more context")

            except Exception as e:
                st.error(f"❌ **Prediction failed:** {e}")

                # Enhanced error handling with solutions
                st.markdown("**🔧 Possible Solutions:**")
                st.write("• **Text preprocessing issues**: Try simpler language")
                st.write("• **Model compatibility problems**: Retrain models if needed")
                st.write("• **Vocabulary mismatch**: Use more common words")

                # Enhanced debugging for errors
                with st.expander(" Technical Details"):
                    import traceback
                    st.code(traceback.format_exc())

    # ---- Enhanced Batch Prediction Section ----
    st.markdown("---")
    st.markdown("###  Batch Prediction (CSV Upload)")
    st.caption("Upload a CSV file to analyze multiple reviews at once")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="CSV should contain a text column with reviews to analyze"
    )

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"✅ **File loaded successfully:** {len(df_batch):,} rows, {len(df_batch.columns)} columns")

            # Enhanced column detection
            text_column_options = []
            for col in df_batch.columns:
                if any(keyword in col.lower() for keyword in ['text', 'review', 'comment', 'content', 'description']):
                    text_column_options.append(col)

            if not text_column_options:
                text_column_options = df_batch.columns.tolist()

            col_batch1, col_batch2 = st.columns(2)

            with col_batch1:
                text_col = st.selectbox("Select text column:", text_column_options)
                max_rows = st.number_input("Max rows to process:", 1, len(df_batch), min(1000, len(df_batch)), 100)

            with col_batch2:
                include_confidence = st.checkbox("Include confidence scores", value=True)
                include_all_probs = st.checkbox("Include all probabilities", value=False)

            # Enhanced preview
            if text_col:
                st.markdown("**Data Preview:**")
                preview_df = df_batch[[text_col]].head(3).copy()
                preview_df[f"{text_col}_preview"] = preview_df[text_col].astype(str).str[:100] + "..."
                st.dataframe(preview_df[[f"{text_col}_preview"]], use_container_width=True)

                if st.button(" Run Batch Analysis", type="primary"):
                    df_to_process = df_batch.head(max_rows) if max_rows < len(df_batch) else df_batch

                    with st.spinner(f"🔄 Analyzing {len(df_to_process):,} reviews..."):
                        try:
                            texts = df_to_process[text_col].astype(str).tolist()

                            # Enhanced progress tracking
                            if len(texts) > 100:
                                progress_bar = st.progress(0)
                                status_text = st.empty()

                                # Batch processing with progress updates
                                batch_size = 100
                                all_predictions = []
                                all_probabilities = []

                                for i in range(0, len(texts), batch_size):
                                    batch_texts = texts[i:i + batch_size]
                                    batch_X = vectorize_texts(batch_texts)
                                    batch_probs = predict_proba(batch_X)
                                    batch_preds = np.argmax(batch_probs, axis=1)

                                    all_predictions.extend(batch_preds)
                                    all_probabilities.extend(batch_probs)

                                    progress = min((i + batch_size) / len(texts), 1.0)
                                    progress_bar.progress(progress)
                                    status_text.text(
                                        f"Processed {min(i + batch_size, len(texts)):,} / {len(texts):,} reviews...")

                                progress_bar.progress(1.0)
                                status_text.text("✅ Analysis complete!")
                            else:
                                # Process all at once for smaller batches
                                X_batch = vectorize_texts(texts)
                                all_probabilities = predict_proba(X_batch)
                                all_predictions = np.argmax(all_probabilities, axis=1)

                            # Create enhanced results dataframe
                            results_df = df_to_process.copy()
                            results_df["Predicted_Emotion"] = [idx2lbl[i] for i in all_predictions]

                            if include_confidence:
                                results_df["Confidence"] = [prob.max() for prob in all_probabilities]

                            if include_all_probs:
                                for i, label in enumerate(ordered_labels):
                                    results_df[f"Prob_{label}"] = [prob[i] for prob in all_probabilities]

                            # Enhanced results summary
                            st.markdown("####  Batch Analysis Results")

                            pred_summary = pd.Series([idx2lbl[i] for i in all_predictions]).value_counts()

                            col_summary = st.columns(min(len(pred_summary), 4))
                            for i, (emotion, count) in enumerate(pred_summary.items()):
                                if i < len(col_summary):
                                    percentage = count / len(all_predictions) * 100
                                    emoji = EMOJI.get(emotion, "📊")
                                    col_summary[i].metric(
                                        f"{emoji} {emotion}",
                                        f"{count:,}",
                                        f"{percentage:.1f}%"
                                    )

                            # Enhanced results display
                            st.markdown("** Results Preview:**")
                            display_cols = [text_col, "Predicted_Emotion"]
                            if include_confidence:
                                display_cols.append("Confidence")

                            display_df = results_df[display_cols].head(10).copy()
                            display_df[f"{text_col}_short"] = display_df[text_col].astype(str).str[:80] + "..."
                            final_cols = [f"{text_col}_short", "Predicted_Emotion"]
                            if include_confidence:
                                final_cols.append("Confidence")

                            st.dataframe(display_df[final_cols], use_container_width=True)

                            # Enhanced download with metadata
                            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"emotion_predictions_{model_choice.lower().replace(' ', '_')}_{timestamp}.csv"

                            # Add metadata to results
                            results_df.attrs['model_used'] = model_choice
                            results_df.attrs['prediction_timestamp'] = timestamp

                            csv_data = results_df.to_csv(index=False).encode("utf-8")

                            st.download_button(
                                " Download Complete Results",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv",
                                help=f"Download all {len(results_df):,} predictions with metadata"
                            )

                            # Quality metrics for batch
                            if include_confidence:
                                avg_confidence = results_df["Confidence"].mean()
                                low_confidence_count = (results_df["Confidence"] < 0.6).sum()

                                st.markdown("####  Batch Quality Metrics")
                                col_qual1, col_qual2, col_qual3 = st.columns(3)

                                with col_qual1:
                                    st.metric("Average Confidence", f"{avg_confidence:.3f}")
                                with col_qual2:
                                    st.metric("High Confidence", f"{(results_df['Confidence'] >= 0.8).sum():,}")
                                with col_qual3:
                                    low_conf_pct = low_confidence_count / len(results_df) * 100
                                    st.metric("Low Confidence", f"{low_confidence_count:,} ({low_conf_pct:.1f}%)")

                        except Exception as e:
                            st.error(f"❌ Batch processing failed: {e}")
                            st.info("💡 **Try:** Reducing batch size or checking data format")

        except Exception as e:
            st.error(f"❌ Could not read CSV file: {e}")
            st.info(" **Requirements:** Valid CSV file with UTF-8 encoding")

    # ---- Enhanced Prediction History ----
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("###  Prediction History & Analytics")

        with st.expander(" View Session Analytics", expanded=False):
            history_df = pd.DataFrame(st.session_state.prediction_history)

            if len(history_df) > 0:
                # Enhanced analytics
                col_hist1, col_hist2, col_hist3, col_hist4 = st.columns(4)

                with col_hist1:
                    total_predictions = len(history_df)
                    st.metric("Total Predictions", total_predictions)

                with col_hist2:
                    avg_confidence = history_df['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")

                with col_hist3:
                    most_common_emotion = history_df['prediction'].mode().iloc[0] if not history_df.empty else "None"
                    st.metric("Most Common", most_common_emotion)

                with col_hist4:
                    if 'vocab_coverage' in history_df.columns:
                        avg_coverage = history_df['vocab_coverage'].mean()
                        st.metric("Avg Coverage", f"{avg_coverage:.1%}")

                # Prediction distribution chart
                if len(history_df) > 1:
                    st.markdown("** Prediction Distribution:**")
                    emotion_counts = history_df['prediction'].value_counts()

                    # Create a simple bar chart
                    chart_data = pd.DataFrame({
                        'Emotion': emotion_counts.index,
                        'Count': emotion_counts.values
                    })
                    st.bar_chart(chart_data.set_index('Emotion'))

                # Recent predictions table
                st.markdown("** Recent Predictions:**")
                recent_df = history_df.tail(10)[['text', 'prediction', 'confidence', 'model']].copy()
                recent_df.index = range(len(recent_df), 0, -1)  # Reverse chronological order
                st.dataframe(recent_df, use_container_width=True)

                # Download history
                history_csv = history_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    " Download Session History",
                    data=history_csv,
                    file_name=f"prediction_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                # Clear history option
                if st.button(" Clear History", help="Clear all prediction history for this session"):
                    st.session_state.prediction_history = []
                    st.success("✅ Prediction history cleared!")
                    st.experimental_rerun()

    # ---- Enhanced Debugging Panel ----
    with st.expander(" Advanced Debugging & Model Info"):
        st.markdown("###  Model Debugging Information")

        # Model information
        col_debug1, col_debug2 = st.columns(2)

        with col_debug1:
            st.markdown("** Current Session State:**")
            st.write(f"• **Word2Vec Model**: {'✅ Loaded' if w2v else '❌ Missing'}")
            st.write(f"• **Label Mapping**: {label_map}")
            st.write(f"• **Available Models**: {', '.join(model_options)}")
            st.write(f"• **Selected Model**: {model_choice}")

        with col_debug2:
            st.markdown("** Model Performance:**")
            if model_choice in model_metrics and model_metrics[model_choice]:
                metrics = model_metrics[model_choice]
                for metric_name, value in metrics.items():
                    if value is not None:
                        st.write(f"• **{metric_name.title()}**: {value:.4f}")
            else:
                st.write("• No metrics available")

        # Vocabulary analysis
        if w2v:
            st.markdown("** Vocabulary Analysis:**")
            vocab_size = len(w2v.wv.key_to_index)
            st.write(f"• **Total Vocabulary**: {vocab_size:,} words")

            # Sample vocabulary words
            sample_words = list(w2v.wv.key_to_index.keys())[:20]
            st.write(f"• **Sample Words**: {', '.join(sample_words[:10])}...")

            # Check for common sentiment words
            sentiment_words = ['good', 'bad', 'excellent', 'terrible', 'amazing', 'awful', 'love', 'hate']
            found_sentiment = [w for w in sentiment_words if w in w2v.wv.key_to_index]
            missing_sentiment = [w for w in sentiment_words if w not in w2v.wv.key_to_index]

            if found_sentiment:
                st.write(f"• **Found Sentiment Words**: {', '.join(found_sentiment)}")
            if missing_sentiment:
                st.write(f"• **Missing Sentiment Words**: {', '.join(missing_sentiment)}")

        # Preprocessing information
        st.markdown("** Text Preprocessing:**")
        st.write(
            f"• **Preprocessing Function**: {'✅ Available' if callable(preprocess_text_fn) else '❌ Using fallback'}")
        st.write(f"• **Stop Words**: {len(stop_set)} words")

        if len(stop_set) > 0:
            sample_stop = list(stop_set)[:10]
            st.write(f"• **Sample Stop Words**: {', '.join(sample_stop)}...")

    # ---- Enhanced Tips and Best Practices ----
    st.markdown("---")
    st.markdown("###  Tips for Better Predictions")

    col_tips1, col_tips2 = st.columns(2)

    with col_tips1:
        st.markdown("""
        ** For Better Accuracy:**
        - Use **complete sentences** with context
        - Include **descriptive adjectives** (delicious, terrible, etc.)
        - Write **food-related content** (model's specialty)
        - Avoid **excessive abbreviations** or slang
        - Add **emotional indicators** (love, hate, disappointed)
        """)

    with col_tips2:
        st.markdown("""
        ** Understanding Results:**
        - **High confidence (>80%)**: Trust the prediction
        - **Moderate confidence (60-80%)**: Good prediction
        - **Low confidence (<60%)**: Be cautious, check alternatives
        - **Mixed probabilities**: Text may have mixed sentiment
        - **Zero vector warning**: Add more common words
        """)

    # Model-specific guidance
    st.markdown("####  Model-Specific Guidance")

    if model_choice == "Random Forest":
        st.info(" **Random Forest**: Good for stable predictions, handles mixed sentiment well")
    elif model_choice == "XGBoost":
        st.info(" **XGBoost**: Often more accurate, better with complex patterns")
    else:
        st.info(" **Ensemble**: Combines both models for potentially better performance")

    # Domain-specific advice
    st.markdown("####  Domain-Specific Tips")
    st.warning("""
    ** This model was trained on Amazon Fine Food Reviews:**
    - **Best performance**: Food and beverage reviews
    - **Good performance**: Product reviews, service experiences
    - **Limited performance**: Technical content, abstract topics
    - **Example domains**: Restaurants, grocery products, kitchen appliances
    """)

    # ---- Enhanced Footer with Quick Actions ----
    st.markdown("---")

    col_footer1, col_footer2, col_footer3 = st.columns(3)

    with col_footer1:
        if st.button(" Refresh Models", help="Reload models from session state"):
            st.experimental_rerun()

    with col_footer2:
        if st.button(" View Model Metrics", help="Go to modeling results"):
            st.info(" Check the **Modeling & Results** page for detailed performance metrics")

    with col_footer3:
        if st.button(" View Word Clouds", help="Analyze emotion-specific vocabulary"):
            st.info(" Check the **Word Clouds** page to see emotion-specific vocabulary")

    # Final encouragement message
    st.success("""
     **Ready to analyze emotions!** 
    Enter your text above or upload a CSV file to get started. 
    The model will provide instant emotion classification with confidence scores.
    """)

    # Quick model validation test
    if st.checkbox(" Run Quick Model Validation", help="Test model with known examples"):
        test_cases = [
            ("This is absolutely amazing and delicious!", "Positive"),
            ("This tastes terrible and awful!", "Negative"),
            ("Product arrived on time, standard quality.", "Neutral")
        ]

        st.markdown("** Model Validation Results:**")

        for test_text, expected in test_cases:
            try:
                X_test = vectorize_texts([test_text])
                p_test = predict_proba(X_test)[0]
                pred_test = idx2lbl[np.argmax(p_test)]
                conf_test = p_test.max()

                if pred_test == expected:
                    st.success(f"✅ **'{test_text[:30]}...'** → {pred_test} ({conf_test:.2f}) ✓")
                else:
                    st.error(f"❌ **'{test_text[:30]}...'** → {pred_test} ({conf_test:.2f}) ≠ {expected}")
            except Exception as e:
                st.error(f"❌ Test failed: {e}")

        st.info("💡 If validation fails, consider retraining models with SMOTE or class weights")
#  ROUTER
PAGES = {
    "Home": page_home,
    "Data Load": page_data_load,
    "Preprocess & Labels": page_preprocess,
    "Post‑Cleaning Diagnostics": page_diagnostics,
    "Embeddings (Word2Vec)": page_word2vec,
    "Modeling and Results (RF & XGBoost)": page_modeling_and_results,
    "Word Clouds": page_wordclouds,
    "Prediction Page": page_predictions,
}

# Sidebar configuration
st.sidebar.markdown("---")

# Sidebar logo (optional; wrap in try so it doesn't crash if missing)
try:
    st.sidebar.image(
        "Sidebar2.png",
        use_column_width=False,
        width=280,
        caption="Emotion Analysis Pipeline"
    )
except Exception:
    st.sidebar.info("Place 'Sidebar2.png' in the project directory for the logo.")

st.sidebar.markdown("---")

# Navigation with progress tracking
st.sidebar.markdown("### Navigation")


# Check completion status for each step
def check_completion_status():
    """Check which steps have been completed"""
    status = {}

    # Step 1: Data is loaded
    status["data_loaded"] = st.session_state.get("df") is not None

    # Step 2: Preprocessing is done
    status["preprocessed"] = st.session_state.get("df_clean") is not None

    # Step 3: Diagnostics completed (if preprocessing is done, this is accessible)
    status["diagnostics"] = status["preprocessed"]  # Diagnostics is available after preprocessing

    # Step 4: Embeddings are trained
    status["embeddings"] = st.session_state.get("w2v_model") is not None

    # Step 5: Models are trained
    status["models"] = (st.session_state.get("rf_model") is not None or
                        st.session_state.get("xgb_model") is not None)

    # Step 6: Results exist
    status["results"] = st.session_state.get("results") is not None

    # Step 7: Word clouds can be generated (needs models and embeddings)
    status["wordclouds"] = status["embeddings"] and status["models"]

    # Step 8: Predictions can be made (needs models and embeddings)
    status["predictions"] = status["embeddings"] and status["models"]

    return status


# Get completion status
completion_status = check_completion_status()

# Create navigation with status indicators
page_status_map = {
    "Home": "READY",  # Always available
    "Data Load": "DONE" if completion_status["data_loaded"] else "START",
    "Preprocess & Labels": "DONE" if completion_status["preprocessed"] else (
        "READY" if completion_status["data_loaded"] else "WAIT"),
    "Post‑Cleaning Diagnostics": "DONE" if completion_status["diagnostics"] else (
        "READY" if completion_status["preprocessed"] else "WAIT"),
    "Embeddings (Word2Vec)": "DONE" if completion_status["embeddings"] else (
        "READY" if completion_status["preprocessed"] else "WAIT"),
    "Modeling and Results (RF & XGBoost)": "DONE" if completion_status["models"] else (
        "READY" if completion_status["embeddings"] else "WAIT"),
    "Word Clouds": "DONE" if completion_status["wordclouds"] else ("READY" if completion_status["models"] else "WAIT"),
    "Prediction Page": "DONE" if completion_status["predictions"] else (
        "READY" if completion_status["models"] else "WAIT"),
}

# Enhanced page selection with status
choice = st.sidebar.selectbox(
    "Select Page:",
    list(PAGES.keys()),
    format_func=lambda x: f"[{page_status_map[x]}] {x}",
    help="DONE = Completed, READY = Ready to start, WAIT = Prerequisites needed"
)

# Progress indicator for all 8 steps
st.sidebar.markdown("### Pipeline Progress")

# Count completed core steps (excluding Home page)
completed_steps = sum([
    completion_status["data_loaded"],  # Step 1
    completion_status["preprocessed"],  # Step 2
    completion_status["diagnostics"],  # Step 3
    completion_status["embeddings"],  # Step 4
    completion_status["models"],  # Step 5
    completion_status["results"],  # Step 6
    completion_status["wordclouds"],  # Step 7
    completion_status["predictions"],  # Step 8
])
total_steps = 8
progress = completed_steps / total_steps

st.sidebar.progress(progress)
st.sidebar.caption(f"**{completed_steps}/{total_steps}** pipeline steps completed ({progress:.0%})")

# Detailed step status
st.sidebar.markdown("**Step Status:**")

steps = [
    ("Data Load", completion_status["data_loaded"]),
    ("Preprocessing", completion_status["preprocessed"]),
    ("Diagnostics", completion_status["diagnostics"]),
    ("Embeddings", completion_status["embeddings"]),
    ("Modeling", completion_status["models"]),
    ("Results", completion_status["results"]),
    ("Word Clouds", completion_status["wordclouds"]),
    ("Predictions", completion_status["predictions"]),
]

for i, (step_name, completed) in enumerate(steps, 1):
    status_text = "DONE" if completed else "TODO"
    st.sidebar.caption(f"{i}. [{status_text}] {step_name}")

# Show next recommended step
st.sidebar.markdown("### What's Next?")
if not completion_status["data_loaded"]:
    st.sidebar.info("**Next:** Load your Amazon reviews dataset")
elif not completion_status["preprocessed"]:
    st.sidebar.info("**Next:** Clean and preprocess the text data")
elif not completion_status["diagnostics"]:
    st.sidebar.info("**Next:** Run diagnostics to validate data quality")
elif not completion_status["embeddings"]:
    st.sidebar.info("**Next:** Train Word2Vec embeddings")
elif not completion_status["models"]:
    st.sidebar.info("**Next:** Train RF & XGBoost models")
elif not completion_status["results"]:
    st.sidebar.info("**Next:** Evaluate model performance")
elif not completion_status["wordclouds"]:
    st.sidebar.info("**Next:** Generate emotion-specific word clouds")
elif not completion_status["predictions"]:
    st.sidebar.info("**Next:** Test predictions on new reviews")
else:
    st.sidebar.success("**Pipeline Complete!** All steps finished!")

st.sidebar.markdown("---")

# Quick stats (if data is available)
if completion_status["data_loaded"]:
    df = st.session_state.get("df")
    if df is not None:
        st.sidebar.markdown("### Dataset Info")
        st.sidebar.metric("Rows", f"{len(df):,}")
        st.sidebar.metric("Columns", len(df.columns))

        if completion_status["preprocessed"]:
            df_clean = st.session_state.get("df_clean")
            if df_clean is not None:
                emotions = df_clean["Emotion"].value_counts()
                st.sidebar.markdown("**Emotion Distribution:**")
                for emotion, count in emotions.items():
                    percentage = (count / len(df_clean)) * 100
                    st.sidebar.caption(f"• {emotion}: {count:,} ({percentage:.1f}%)")

# Model performance (if available)
if completion_status["models"]:
    results = st.session_state.get("results")
    if results and "metrics" in results:
        st.sidebar.markdown("### Best Model")
        metrics_df = pd.DataFrame(results["metrics"])
        if not metrics_df.empty and "Test_F1" in metrics_df.columns:
            best_idx = metrics_df["Test_F1"].idxmax()
            best_model = metrics_df.loc[best_idx]
            st.sidebar.metric("Model", best_model["Model"])
            st.sidebar.metric("F1 Score", f"{best_model['Test_F1']:.3f}")
            st.sidebar.metric("Accuracy", f"{best_model['Test_Accuracy']:.3f}")

st.sidebar.markdown("---")

# Help section
with st.sidebar.expander("Need Help?"):
    st.markdown("""
    **Pipeline Steps:**
    1. **Data Load** - Upload Amazon reviews dataset
    2. **Preprocess** - Clean text and map emotions  
    3. **Diagnostics** - Validate data quality
    4. **Embeddings** - Train Word2Vec model
    5. **Modeling** - Train RF & XGBoost classifiers
    6. **Results** - Evaluate model performance
    7. **Word Clouds** - Generate emotion visualizations
    8. **Predictions** - Test on new reviews

    **Tips:**
    - Follow steps in order for best results
    - Check status indicators before proceeding
    - Each step builds on previous ones
    - DONE = completed, READY = ready to start, WAIT = prerequisites needed
    """)

# Session management
with st.sidebar.expander("Session Management"):
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear All", help="Clear all session data and restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session cleared!")
            st.experimental_rerun()

    with col2:
        # Show session info
        session_size = len(st.session_state)
        st.metric("Session Items", session_size)

    # Show memory usage info
    if session_size > 0:
        key_info = []
        for key in st.session_state.keys():
            if hasattr(st.session_state[key], '__len__'):
                try:
                    size = len(st.session_state[key])
                    key_info.append(f"• {key}: {size:,} items")
                except:
                    key_info.append(f"• {key}: cached")
            else:
                key_info.append(f"• {key}: cached")

        if key_info:
            st.caption("**Session Contents:**")
            for info in key_info[:5]:  # Show first 5
                st.caption(info)
            if len(key_info) > 5:
                st.caption(f"... and {len(key_info) - 5} more items")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p><strong>Emotion Analysis Pipeline</strong></p>
    <p>8-Step Word2Vec + ML Classification</p>
    <p>Built with Streamlit & scikit-learn</p>
</div>
""", unsafe_allow_html=True)

# Run the selected page with error handling
try:
    PAGES[choice]()
except Exception as e:
    st.error(f"Error running page '{choice}': {e}")
    st.info("Try refreshing the page or clearing the session data if the error persists.")

    # Show traceback in debug mode
    if st.sidebar.checkbox("Show Debug Info"):
        import traceback

        st.code(traceback.format_exc())

        # Show session state for debugging
        st.markdown("**Session State Debug:**")
        for key, value in st.session_state.items():
            try:
                st.write(f"• **{key}**: {type(value).__name__}")
            except:
                st.write(f"• **{key}**: <unable to display>")