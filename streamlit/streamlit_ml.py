"""
terminal: streamlit run streamlit_ml.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Cell Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# load data and models
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    scaler       = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    model_rf     = joblib.load(os.path.join(BASE_DIR, "model_rf.pkl"))
    model_svm    = joblib.load(os.path.join(BASE_DIR, "model_svm.pkl"))
    model_knn    = joblib.load(os.path.join(BASE_DIR, "model_knn.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    label_names   = joblib.load(os.path.join(BASE_DIR, "label_names.pkl"))
    data          = joblib.load(os.path.join(BASE_DIR, "data_splits.pkl"))
    models = {
        "Random Forest": model_rf,
        "SVM":           model_svm,
        "KNN":           model_knn,
    }
    return scaler, models, feature_names, label_names, data

try:
    scaler, models, feature_names, label_names, data = load_artifacts()
except FileNotFoundError:
    st.error(
        "Modell-Dateien nicht gefunden. "
    )
    st.stop()

X_train = data["X_train"]; y_train = data["y_train"]
X_val   = data["X_val"];   y_val   = data["y_val"]
X_test  = data["X_test"];  y_test  = data["y_test"]

MODEL_PARAMS = {
    "Random Forest": "n_estimators=100, max_depth=20, min_samples_split=2",
    "SVM":           "kernel=rbf, C=10, gamma=scale",
    "KNN":           "n_neighbors=5, weights=distance, metric=euclidean",
}

# ─────────────────────────────────────────────────────────────────────────────
# functions
# ─────────────────────────────────────────────────────────────────────────────
def get_scores(model):
    return {
        "Train":      model.score(X_train, y_train),
        "Validation": model.score(X_val,   y_val),
        "Test":       model.score(X_test,  y_test),
    }

def make_confusion(model, normalized=False):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    if normalized:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    return cm, y_pred

def fig_confusion(cm, normalized, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    fmt = ".1%" if normalized else "d"
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax, linewidths=0.4, linecolor="#e0e0e0",
        vmin=0, vmax=1 if normalized else None,
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    return fig

def fig_per_class(y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(label_names))
    )
    x = np.arange(len(label_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x - width, precision, width, label="Precision", color="#4C72B0")
    b2 = ax.bar(x,         recall,    width, label="Recall",    color="#DD8452")
    b3 = ax.bar(x + width, f1,        width, label="F1-Score",  color="#55A868")
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics on Test Set", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

def fig_pr_curves(model):
    y_test_bin = label_binarize(y_test, classes=range(len(label_names)))
    y_proba    = model.predict_proba(X_test)
    fig, ax    = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))
    for idx, (name, color) in enumerate(zip(label_names, colors)):
        prec, rec, _ = precision_recall_curve(y_test_bin[:, idx], y_proba[:, idx])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, label=f"{name} (AUC={pr_auc:.2f})", linewidth=2, color=color)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves per Class", fontsize=13)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig

def fig_feature_importance(model):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color="#4C72B0",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig

def fig_support_vs_f1(y_pred):
    report = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True
    )
    support = [int(np.sum(y_train == i)) for i in range(len(label_names))]
    f1s     = [report[n]["f1-score"] for n in label_names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(support, f1s, s=120, alpha=0.75, color="#4C72B0", edgecolors="black")
    for i, n in enumerate(label_names):
        ax.annotate(n, (support[i], f1s[i]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    z = np.polyfit(support, f1s, 1)
    xs = sorted(support)
    ax.plot(xs, np.poly1d(z)(xs), "r--", alpha=0.5, label="Trend")
    ax.set_xlabel("Training Samples per Class", fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("Training Support vs. F1-Score", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

def fig_accuracy_comparison():
    rows = []
    for name, model in models.items():
        s = get_scores(model)
        rows.append({"Model": name, **s})
    df = pd.DataFrame(rows).set_index("Model")
    fig, ax = plt.subplots(figsize=(7, 4))
    x   = np.arange(len(df))
    w   = 0.25
    c   = ["#4C72B0", "#DD8452", "#55A868"]
    for i, col in enumerate(["Train", "Validation", "Test"]):
        bars = ax.bar(x + (i - 1) * w, df[col], w, label=col, color=c[i])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison: Train / Validation / Test Accuracy", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Blood Cell Classification")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Modells",
            "Feature Analysis",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Choose Model")
    selected_model_name = st.selectbox(
        "active Model",
        list(models.keys()),
        label_visibility="collapsed",
    )
    selected_model = models[selected_model_name]

    st.caption(f"Parameter: `{MODEL_PARAMS[selected_model_name]}`")

    st.divider()
    # st.caption(
    #     "Modelle trainiert und gespeichert mit `train_models.py`. "
    #     "Geladen via `joblib.load()`."
    # )

# ─────────────────────────────────────────────────────────────────────────────
# page 1: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.header("Overview")

    # Metrikkarten (alle 3 Modelle nebeneinander)
    cols = st.columns(3)
    for col, (name, model) in zip(cols, models.items()):
        scores = get_scores(model)
        with col:
            st.subheader(name)
            st.metric("Train Accuracy",      f"{scores['Train']:.3f}")
            st.metric("Validation Accuracy", f"{scores['Validation']:.3f}")
            st.metric("Test Accuracy",       f"{scores['Test']:.3f}")

    st.divider()

    st.subheader("Accuracy")
    st.pyplot(fig_accuracy_comparison())

    st.divider()

    st.subheader("Data set")
    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Train-Samples",  len(y_train))
    with info_cols[1]:
        st.metric("Val-Samples",    len(y_val))
    with info_cols[2]:
        st.metric("Test-Samples",   len(y_test))
    with info_cols[3]:
        st.metric("Features",       len(feature_names))

    st.subheader("Class-Distribution (Train)")
    class_counts = pd.Series(y_train).value_counts().sort_index()
    class_counts.index = [label_names[i] for i in class_counts.index]
    st.bar_chart(class_counts)

# ─────────────────────────────────────────────────────────────────────────────
# page 2: Models
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Models":
    st.header(f"Model: {selected_model_name}")
    st.caption(f"Hyperparameter: `{MODEL_PARAMS[selected_model_name]}`")

    cm_raw, y_pred = make_confusion(selected_model, normalized=False)
    cm_norm, _     = make_confusion(selected_model, normalized=True)

    # Confusion Matrices
    st.subheader("Confusion Matrix")
    show_norm = st.toggle("Show Normalisized", value=False)
    if show_norm:
        st.pyplot(fig_confusion(cm_norm, True, f"{selected_model_name} – Normalisized"))
    else:
        st.pyplot(fig_confusion(cm_raw,  False, f"{selected_model_name} – Absolut"))

    st.divider()

    # Per-Class metrics
    st.subheader("Per-Class Metrics")
    st.pyplot(fig_per_class(y_pred))

    st.divider()

    # Precision-Recall Kurven
    st.subheader("Precision-Recall curves")
    st.pyplot(fig_pr_curves(selected_model))

    st.divider()

    # Classification Report als Tabelle
    st.subheader("Classification Report")
    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T.round(3)
    st.dataframe(report_df, use_container_width=True)

    st.divider()

    # Support vs. F1
    st.subheader("Trainings-Support vs. F1-Score")
    st.pyplot(fig_support_vs_f1(y_pred))


# ─────────────────────────────────────────────────────────────────────────────
# page 3: Feature-Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Feature Analysis":
    st.header("Feature Analysis")

    st.subheader("Random Forest Feature Importance")

    rf_model = models["Random Forest"]
    st.pyplot(fig_feature_importance(rf_model))

    st.divider()

    importances = rf_model.feature_importances_
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.index += 1
    importance_df["Importance"] = importance_df["Importance"].round(4)

    st.subheader("Feature-Ranking")
    st.dataframe(importance_df, use_container_width=True, height=400)

    st.divider()

    st.subheader("Feature Distribution")
    selected_feature = st.selectbox(
        "choose Feature",
        importance_df["Feature"].tolist(),
    )
    feat_idx = list(feature_names).index(selected_feature)

    fig_feat, ax_feat = plt.subplots(figsize=(8, 4))
    for class_idx, class_name in enumerate(label_names):
        vals = X_train[y_train == class_idx, feat_idx]
        ax_feat.hist(vals, bins=30, alpha=0.5, label=class_name, density=True)
    ax_feat.set_xlabel(selected_feature, fontsize=11)
    ax_feat.set_ylabel(" ")
    ax_feat.set_title(f"Distribution of '{selected_feature}' per class", fontsize=13)
    ax_feat.legend(fontsize=8)
    ax_feat.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_feat)