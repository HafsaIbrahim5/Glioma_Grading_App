import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder


# ------------------------------------------------------------
# Data loading and preprocessing
# ------------------------------------------------------------
@st.cache_data
def load_data(csv_path: str = "dataset.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        st.error(
            f"Dataset file `{csv_path}` not found. "
            f"Please make sure it exists in the same folder as this app."
        )
        st.stop()
    data = pd.read_csv(csv_path)
    return data


def _get_column_name(df: pd.DataFrame, candidates) -> str:
    normalized_cols = {c.strip().upper(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().upper()
        if key in normalized_cols:
            return normalized_cols[key]
    st.error(
        "Expected one of the following columns, but none were found in your dataset: "
        f"{candidates}\n\nAvailable columns: {list(df.columns)}"
    )
    st.stop()


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    data = df.copy()
    data = data.drop_duplicates()

    target_col = _get_column_name(
        data, ["GRADE", "Grade", "tumor_grade", "Glioma_grade"]
    )

    target_encoder = LabelEncoder()
    data[target_col] = target_encoder.fit_transform(data[target_col])

    for col in data.select_dtypes(include=["object", "category"]).columns:
        if col == target_col:
            continue
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    X = data.drop(target_col, axis=1)
    y = data[target_col]
    return X, y, target_encoder


@st.cache_resource
def train_model(random_state: int = 42):
    raw_data = load_data()
    X, y, target_encoder = preprocess_data(raw_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    base_model = LogisticRegression(max_iter=500, solver="lbfgs", multi_class="auto")
    param_grid = {"C": [0.1, 1.0, 10.0]}
    grid = GridSearchCV(base_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_res, y_train_res)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)

    metrics_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test,
        "classes_": target_encoder.classes_,
    }

    if len(target_encoder.classes_) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics_dict["y_proba"] = y_proba
        metrics_dict["roc_auc"] = roc_auc_score(y_test, y_proba)

    return model, metrics_dict, raw_data, target_encoder


# ------------------------------------------------------------
# Page configuration and custom CSS
# ------------------------------------------------------------
def set_page_config():
    st.set_page_config(
        page_title="Glioma Grading · ML Studio",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Color palette
    primary_color = "#A78BFA"  # soft violet
    accent_color = "#F472B6"  # pink
    secondary_color = "#2DD4BF"  # teal
    surface_color = "rgba(17, 24, 39, 0.85)"  # dark gray with transparency
    bg_color = "#030712"  # almost black

    st.markdown(
        f"""
        <style>
        /* Global background */
        .stApp {{
            background: linear-gradient(145deg, {bg_color} 0%, #111827 100%);
            color: #f3f4f6;
        }}

        /* Main container padding */
        .main .block-container {{
            padding-top: 1.8rem;
            padding-bottom: 1.5rem;
        }}

        /* Headline style */
        .main-title {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, {primary_color}, {accent_color}, {secondary_color});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }}
        .subtitle {{
            color: #9ca3af;
            font-size: 1rem;
            border-left: 4px solid {primary_color};
            padding-left: 1rem;
            margin-top: 0.2rem;
        }}

        /* Modern card design */
        .metric-card {{
            background: {surface_color};
            backdrop-filter: blur(12px);
            padding: 1.2rem 1rem;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.02);
            transition: all 0.2s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-4px);
            border-color: {primary_color}80;
            box-shadow: 0 30px 60px -12px {primary_color}40, 0 0 0 1px {primary_color}40;
        }}

        /* Streamlit metric overrides */
        [data-testid="stMetricValue"] {{
            color: white !important;
            font-weight: 700;
            font-size: 1.6rem !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: #9ca3af !important;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        /* Buttons */
        .stButton>button {{
            background: linear-gradient(145deg, {primary_color}, {accent_color});
            color: white;
            border: none;
            border-radius: 40px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            box-shadow: 0 15px 30px -10px {primary_color}80;
            transition: all 0.2s ease;
        }}
        .stButton>button:hover {{
            background: linear-gradient(145deg, {accent_color}, {secondary_color});
            box-shadow: 0 20px 40px -10px {accent_color}b0;
            transform: scale(1.02);
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] > div:first-child {{
            background: rgba(3, 7, 18, 0.9);
            backdrop-filter: blur(16px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .sidebar-title {{
            font-size: 1.4rem;
            font-weight: 700;
            background: linear-gradient(145deg, {primary_color}, {accent_color});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }}
        .sidebar-subtitle {{
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 1rem;
        }}
        .sidebar-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.4rem 1rem;
            border-radius: 40px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin: 0.2rem 0;
            transition: 0.2s;
        }}
        .sidebar-pill:hover {{
            border-color: {primary_color}80;
            background: rgba(255, 255, 255, 0.05);
        }}
        .sidebar-pill a {{
            color: #e5e7eb !important;
            text-decoration: none;
            font-weight: 500;
        }}
        .sidebar-pill-icon {{
            font-size: 1.2rem;
        }}

        /* Tables */
        thead tr th {{
            background: #1f2937 !important;
            color: #f3f4f6 !important;
        }}
        tbody tr td {{
            color: #d1d5db !important;
        }}

        /* Hide default Streamlit elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Additional flair */
        hr {{
            border-color: rgba(255,255,255,0.05);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# UI rendering functions
# ------------------------------------------------------------
def _stat_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.7rem; color:#9ca3af; margin-bottom:0.2rem;">{label}</div>
            <div style="font-size:1.6rem; font-weight:700; color:white;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(raw_data: pd.DataFrame):
    st.markdown(
        "<div class='main-title'>Glioma Grading · ML Studio</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#9ca3af; margin-top:-0.2rem; font-size:1.1rem;'>by Hafsa Ibrahim</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='subtitle'>
        End‑to‑end machine learning pipeline for brain tumor grading (glioma) 
        using clinical and molecular features. Built with Logistic Regression + SMOTE, 
        deployed as an interactive dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _stat_card("Total samples", f"{raw_data.shape[0]:,}")
    with col2:
        _stat_card("Features", f"{raw_data.shape[1] - 1}")
    with col3:
        _stat_card("Target", "Glioma Grade")
    with col4:
        _stat_card("Model", "Logistic Reg. + SMOTE")

    st.markdown("### 🔍 Dataset preview")
    st.dataframe(raw_data.head(10), use_container_width=True)


def render_eda(raw_data: pd.DataFrame):
    st.markdown("### 📊 Exploratory Data Analysis")

    st.write("**Summary statistics**")
    st.dataframe(raw_data.describe(include="all").T, use_container_width=True)

    grade_col = _get_column_name(raw_data, ["GRADE", "Grade", "tumor_grade"])
    st.markdown("#### Grade distribution")
    class_counts = raw_data[grade_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(
        x=class_counts.index.astype(str),
        y=class_counts.values,
        ax=ax,
        palette="rocket",
    )
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    st.pyplot(fig, use_container_width=True)

    numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.markdown("#### Feature correlations")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            raw_data[numeric_cols].corr(),
            cmap="vlag",
            annot=False,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        st.pyplot(fig, use_container_width=True)

    if len(numeric_cols) > 1:
        st.markdown("#### Distribution by grade")
        feature = st.selectbox(
            "Select a numeric feature:",
            options=[c for c in numeric_cols if c != grade_col],
        )
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(
            data=raw_data,
            x=feature,
            hue=grade_col,
            kde=True,
            palette="coolwarm",
            ax=ax,
        )
        st.pyplot(fig, use_container_width=True)


def render_model_performance(metrics_dict):
    st.markdown("### 📈 Model performance – Logistic Regression")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{metrics_dict['accuracy']*100:.2f}%")
    with col2:
        st.metric("Classes", ", ".join(map(str, metrics_dict["classes_"])))

    st.markdown("#### Confusion matrix")
    cm = metrics_dict["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, use_container_width=True)

    if "roc_auc" in metrics_dict:
        st.markdown("#### ROC curve (binary case)")
        y_test = metrics_dict["y_test"]
        y_proba = metrics_dict["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(fpr, tpr, label=f"AUC = {metrics_dict['roc_auc']:.3f}", lw=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig, use_container_width=True)

    st.markdown("#### Detailed report")
    report = pd.DataFrame(metrics_dict["classification_report"]).T
    st.dataframe(
        report.style.background_gradient(cmap="PuBu"), use_container_width=True
    )


def _build_manual_input_schema(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = raw_data.copy()
    grade_col = _get_column_name(df, ["GRADE", "Grade", "tumor_grade", "Glioma_grade"])
    return df.drop(columns=[grade_col])


def manual_input_form(raw_data: pd.DataFrame) -> pd.DataFrame:
    feature_df = _build_manual_input_schema(raw_data)
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    st.markdown("### 🧪 Single patient features")
    user_input = {}

    # Two columns for numeric sliders
    col_left, col_right = st.columns(2)
    half = len(numeric_cols) // 2 + 1

    with col_left:
        for col in numeric_cols[:half]:
            minv, maxv = float(feature_df[col].min()), float(feature_df[col].max())
            default = float(feature_df[col].median())
            user_input[col] = st.slider(col, minv, maxv, default)

    with col_right:
        for col in numeric_cols[half:]:
            minv, maxv = float(feature_df[col].min()), float(feature_df[col].max())
            default = float(feature_df[col].median())
            user_input[col] = st.slider(col, minv, maxv, default)

    for col in cat_cols:
        options = sorted(feature_df[col].dropna().unique().tolist())
        user_input[col] = st.selectbox(col, options)

    return pd.DataFrame([user_input])


def apply_same_preprocessing(
    input_df: pd.DataFrame, raw_data: pd.DataFrame
) -> pd.DataFrame:
    df = raw_data.copy()
    grade_col = _get_column_name(df, ["GRADE", "Grade", "tumor_grade", "Glioma_grade"])
    df = df.drop(columns=[grade_col])

    result = input_df.copy()
    for col in df.columns:
        if col not in result.columns:
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            le = LabelEncoder()
            le.fit(df[col])
            if result[col].iloc[0] not in le.classes_:
                le.classes_ = np.append(le.classes_, result[col].iloc[0])
            result[col] = le.transform(result[col])
    return result[df.columns]


def render_single_prediction(
    model, raw_data: pd.DataFrame, target_encoder: LabelEncoder
):
    input_df = manual_input_form(raw_data)
    X_input = apply_same_preprocessing(input_df, raw_data)

    st.markdown("---")
    if len(target_encoder.classes_) == 2:
        threshold = st.slider(
            "Decision threshold (positive class)", 0.1, 0.9, 0.5, 0.05
        )
    else:
        threshold = None

    if st.button("Predict grade", type="primary"):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            pred_idx = int(np.argmax(proba))
        else:
            pred_idx = int(model.predict(X_input)[0])
            proba = None

        label_text = target_encoder.inverse_transform([pred_idx])[0]

        st.markdown("#### ✅ Prediction result")
        st.write(f"**Predicted grade:** `{label_text}`")

        if proba is not None:
            st.write("**Class probabilities:**")
            prob_series = pd.Series(proba, index=target_encoder.classes_)
            st.bar_chart(prob_series)


def render_batch_prediction(
    model, raw_data: pd.DataFrame, target_encoder: LabelEncoder
):
    st.markdown("### 📁 Batch prediction (CSV upload)")
    st.write(
        "Upload a CSV file containing the same features as the training set "
        "(without the grade column)."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is None:
        st.stop()

    try:
        df_new = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    feature_df = _build_manual_input_schema(raw_data)
    missing_cols = [c for c in feature_df.columns if c not in df_new.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    X_new = apply_same_preprocessing(df_new[feature_df.columns], raw_data)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)
        preds = np.argmax(proba, axis=1)
    else:
        preds = model.predict(X_new)
        proba = None

    labels = target_encoder.inverse_transform(preds)

    result_df = df_new.copy()
    result_df["predicted_grade"] = labels
    if proba is not None:
        for i, cls in enumerate(target_encoder.classes_):
            result_df[f"prob_{cls}"] = proba[:, i]

    st.markdown("#### Sample of predictions")
    st.dataframe(result_df.head(20), use_container_width=True)

    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download full predictions (CSV)",
        data=csv_out,
        file_name="glioma_predictions.csv",
        mime="text/csv",
    )


def render_project_info():
    st.markdown("### 📌 About this project")

    st.markdown(
        """
        **Project summary**  
        This dashboard reproduces a classical machine learning notebook for glioma grading.
        The original notebook performed:
        - Data loading and inspection (`dataset.csv`)
        - Exploratory Data Analysis (distributions, correlations, class imbalance)
        - SMOTE oversampling to handle imbalance
        - Logistic Regression with basic hyperparameter tuning (GridSearchCV)
        - Evaluation via accuracy, confusion matrix, classification report, and ROC‑AUC
        - Packaging as an interactive Streamlit app with EDA and prediction interfaces

        **Why this matters**  
        Glioma grading is crucial for treatment planning. This project demonstrates a complete
        ML workflow, from data to deployment, in a clean, modern UI.

        **Author**  
        - **Hafsa Ibrahim** – AI & ML enthusiast  
        - [LinkedIn](https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/)  
        - [GitHub](https://github.com/HafsaIbrahim5)

        **Built with**  
        🐍 Python · 📊 Streamlit · 🤖 scikit-learn · 🧮 imbalanced-learn · 📈 matplotlib/seaborn
        """
    )


# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------
def main():
    set_page_config()

    with st.sidebar:
        st.markdown(
            "<div class='sidebar-title'>🧬 Glioma Grading</div>", unsafe_allow_html=True
        )
        st.markdown(
            "<div class='sidebar-subtitle'>interactive ML pipeline</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=[
                "Overview",
                "Data Exploration",
                "Model Performance",
                "Single Prediction",
                "Batch Prediction",
                "Project Info",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Connect**")
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; gap:0.3rem;">
                <div class="sidebar-pill"><span class="sidebar-pill-icon">🔗</span><a href="https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/" target="_blank">LinkedIn</a></div>
                <div class="sidebar-pill"><span class="sidebar-pill-icon">🐙</span><a href="https://github.com/HafsaIbrahim5" target="_blank">GitHub</a></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Built with**")
        st.markdown(
            """
            <div style="display:flex; gap:0.3rem; flex-wrap:wrap;">
                <span class="sidebar-pill">🐍 Python</span>
                <span class="sidebar-pill">📊 Streamlit</span>
                <span class="sidebar-pill">🤖 scikit-learn</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Load model and data once
    model, metrics_dict, raw_data, target_encoder = train_model()

    # Page routing
    if page == "Overview":
        render_overview(raw_data)
    elif page == "Data Exploration":
        render_eda(raw_data)
    elif page == "Model Performance":
        render_model_performance(metrics_dict)
    elif page == "Single Prediction":
        render_single_prediction(model, raw_data, target_encoder)
    elif page == "Batch Prediction":
        render_batch_prediction(model, raw_data, target_encoder)
    elif page == "Project Info":
        render_project_info()


if __name__ == "__main__":
    main()
