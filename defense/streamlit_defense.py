

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Cell Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Load feature data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    path = os.path.join(BASE_DIR, "all_features.csv")
    return pd.read_csv(path)

df = load_features()

# Reduced dataframe for feature analysis (drop non-feature columns)
cols_to_drop = [c for c in ["Filepath", "Source", "augmented"] if c in df.columns]
df_reduced = df.drop(columns=cols_to_drop)

label_names   = sorted(df_reduced["Label"].unique().tolist())
feature_names = [c for c in df_reduced.columns if c != "Label"]

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – Page navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Blood Cell Classification")
    st.divider()
    page = st.radio(
        "Page",
        ["Data Visualization", "Feature Engineering", "Modelling"],
        label_visibility="collapsed",
    )
    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – Data Visualization
# ─────────────────────────────────────────────────────────────────────────────
if page == "Data Visualization":
    st.header("Data Visualization")

    tab1, tab2, tab3 = st.tabs(
        ["Dataset Overview", "Size Distribution", "Class Distribution"]
    )

    # ── Tab 1: Dataset Overview ──────────────────────────────────────────────
    with tab1:
        st.subheader("Dataset Overview")

        CELLS_DIR = os.path.join(BASE_DIR, "NormalCells")

        if not os.path.isdir(CELLS_DIR):
            st.warning(f"Folder 'NormalCells' not found at: {CELLS_DIR}")
        else:
            subfolders = sorted([
                d for d in os.listdir(CELLS_DIR)
                if os.path.isdir(os.path.join(CELLS_DIR, d))
            ])

            if st.button("Reload"):
                st.session_state["reload_seed"] = random.randint(0, 999999)

            seed = st.session_state.get("reload_seed", 42)
            random.seed(seed)

            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            axes = axes.flatten()

            for ax, folder in zip(axes, subfolders[:8]):
                folder_path = os.path.join(CELLS_DIR, folder)
                images = [
                    f for f in os.listdir(folder_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
                ]
                if images:
                    chosen = random.choice(images)
                    img = Image.open(os.path.join(folder_path, chosen))
                    ax.imshow(img, cmap="gray")
                else:
                    ax.text(0.5, 0.5, "No image found",
                            ha="center", va="center", transform=ax.transAxes)
                ax.set_title(folder, fontsize=10)
                ax.axis("off")

            plt.tight_layout()
            st.pyplot(fig)

    # ── Tab 2: Size Distribution ─────────────────────────────────────────────
    with tab2:
        st.subheader("Size Distribution")
        size_path = os.path.join(BASE_DIR, "size_dist.png")
        if os.path.isfile(size_path):
            st.image(size_path, use_container_width=True)
        else:
            st.warning(f"File 'size_dist.png' not found at: {size_path}")

    # ── Tab 3: Class Distribution ─────────────────────────────────────────────
    with tab3:
        st.subheader("Class Distribution")

        if "Source" not in df.columns or "Label" not in df.columns:
            st.warning("Columns 'Source' and/or 'Label' not found in all_features.csv.")
        else:
            # Checkboxes to toggle Original / Augmented
            col_chk1, col_chk2, _ = st.columns([1, 1, 6])
            with col_chk1:
                show_original  = st.checkbox("Original",  value=True)
            with col_chk2:
                show_augmented = st.checkbox("Augmented", value=True)

            # Precompute counts
            # labels_ordered = sorted(df["Label"].unique().tolist())
            labels_ordered = (
                df[df["Source"] == "Original"]
                .groupby("Label").size()
                .sort_values(ascending=False)
                .index.tolist()
            )


            orig_counts = (
                df[df["Source"] == "Original"]
                .groupby("Label").size()
                .reindex(labels_ordered, fill_value=0)
            )
            aug_counts = (
                df[df["Source"] == "Augmented"]
                .groupby("Label").size()
                .reindex(labels_ordered, fill_value=0)
            )
            total_orig = orig_counts.sum()
            total_aug  = aug_counts.sum()

            col_plot, col_table = st.columns([3, 2])

            with col_plot:
                x      = np.arange(len(labels_ordered))
                width  = 0.35
                fig, ax = plt.subplots(figsize=(9, 5))

                if show_original and show_augmented:
                    ax.bar(x - width / 2, orig_counts.values, width,
                           label="Original",  color="#4C72B0")
                    ax.bar(x + width / 2, aug_counts.values, width,
                           label="Augmented", color="#DD8452")
                elif show_original:
                    ax.bar(x, orig_counts.values, width * 1.4,
                           label="Original",  color="#4C72B0")
                elif show_augmented:
                    ax.bar(x, aug_counts.values, width * 1.4,
                           label="Augmented", color="#DD8452")

                ax.set_xticks(x)
                ax.set_xticklabels(labels_ordered, rotation=45, ha="right", fontsize=9)
                ax.set_ylabel("Count", fontsize=11)
                ax.set_title("Class Distribution by Source", fontsize=13)
                ax.legend(fontsize=10)
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

            with col_table:
                table_data = []
                for label in labels_ordered:
                    oc = int(orig_counts[label])
                    ac = int(aug_counts[label])
                    op = f"{oc / total_orig * 100:.1f}%" if total_orig > 0 else "—"
                    ap = f"{ac / total_aug  * 100:.1f}%" if total_aug  > 0 else "—"
                    table_data.append({
                        "Label":              label,
                        "Original (n)":       oc,
                        "Original (%)":       op,
                        "Augmented (n)":      ac,
                        "Augmented (%)":      ap,
                    })
                table_df = pd.DataFrame(table_data).set_index("Label")
                st.dataframe(table_df, use_container_width=True, height=340)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Feature Engineering":
    st.header("Feature Engineering")

    tab1, tab2, tab3 = st.tabs(
        ["Feature Distribution", "Feature Extraction", "In-depth Information"]
    )

    # ── Tab 1: Feature Distribution ──────────────────────────────────────────
    with tab1:
        st.subheader("Feature Distribution")

        selected_feature = st.selectbox(
            "Choose Feature",
            feature_names,
        )

        fig_feat, ax_feat = plt.subplots(figsize=(9, 4))

        for class_name in label_names:
            vals = df_reduced.loc[df_reduced["Label"] == class_name, selected_feature].dropna().values
            ax_feat.hist(vals, bins=30, alpha=0.5, label=class_name, density=True)

        ax_feat.set_xlabel(selected_feature, fontsize=11)
        ax_feat.set_ylabel("Density", fontsize=11)
        ax_feat.set_title(f"Distribution of '{selected_feature}' per Class", fontsize=13)
        ax_feat.legend(fontsize=8, loc="best")
        ax_feat.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_feat)

    # ── Tab 2: Feature Extraction ─────────────────────────────────────────────
    with tab2:
        st.subheader("Feature Extraction")

        img1_path = os.path.join(BASE_DIR, "features_biological_shape.png")
        img2_path = os.path.join(BASE_DIR, "features_correlationmatrix.png")

        col1, col2 = st.columns(2)
        with col1:
            if os.path.isfile(img1_path):
                st.image(img1_path, caption="Biological Shape Features", use_container_width=True)
            else:
                st.warning("'features_biological_shape.png' not found.")
        with col2:
            if os.path.isfile(img2_path):
                st.image(img2_path, caption="Feature Correlation Matrix", use_container_width=True)
            else:
                st.warning("'features_correlationmatrix.png' not found.")

    # ── Tab 3: In-depth Information ───────────────────────────────────────────
    with tab3:
        st.subheader("In-depth Information")

        sep_path = os.path.join(BASE_DIR, "features_separability.png")
        if os.path.isfile(sep_path):
            st.image(sep_path, caption="Feature Separability", use_container_width=True)
        else:
            st.warning("'features_separability.png' not found.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – Modelling
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Modelling":
    st.header("Modelling")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Random Forest", "SVM", "KNN", "Comparison"]
    )

    with tab1:
        st.subheader("Random Forest")
        st.info("To be added.")

    with tab2:
        st.subheader("SVM")
        st.info("To be added.")

    with tab3:
        st.subheader("KNN")
        st.info("To be added.")

    with tab4:
        st.subheader("Comparison")
        st.info("To be added.")