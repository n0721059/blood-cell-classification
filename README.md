# Blood Cell Classification: Normal vs. Leukemic Morphologies

This project develops deep learning models to distinguish **normal peripheral blood cells** from **pathological cells in Acute Myeloid Leukemia (AML)** using two expert-annotated public datasets. The goal is to support early, automated screening for hematological malignancies through morphological analysis of single-cell images.

## 🎯 Objective

Build a robust system capable of:
- **Classifying 8 normal blood cell types** (from healthy/clinical samples)
- **Detecting abnormal cells**, especially **myeloblasts** (hallmark of AML)
- Enabling **unsupervised clustering** to discover morphological patterns across health and disease
- Providing a **reproducible pipeline** for research and clinical prototyping

> 🔬 **Clinical Relevance**: Early detection of blasts in peripheral blood can accelerate leukemia diagnosis and reduce reliance on invasive bone marrow biopsies [Boldú et al., 2021].

---

## 📚 Scientific Background

### Normal Blood Cell Morphology
In healthy individuals, peripheral blood contains distinct white blood cell (WBC) types:
- **Granulocytes**: Neutrophils, Eosinophils, Basophils  
- **Agranulocytes**: Lymphocytes, Monocytes  
- **Immature forms**: Immature granulocytes (myelocytes, metamyelocytes, promyelocytes)  
- **Others**: Erythroblasts, Platelets  

These are routinely identified by pathologists during manual differential counts [Acevedo et al., 2019].

### Pathological Context: AML
In **Acute Myeloid Leukemia (AML)**, immature myeloid precursors (**blasts**) proliferate abnormally and appear in peripheral blood. Their presence is a key diagnostic criterion [Matek et al., 2019].

---

## 📁 Data Sources

### 1. **Normal Cells: Acevedo et al. (2019) – Mendeley Dataset**
- **Source**: [Mendeley Dataset](https://data.mendeley.com/datasets/snkd93bnjr/1)  
- **DOI**: [10.17632/snkd93bnjr.1](https://doi.org/10.17632/snkd93bnjr.1)  
- **Images**: 17,092 single-cell RGB images  
- **Classes**: 8 normal types  
  - Neutrophils, Eosinophils, Basophils  
  - Lymphocytes, Monocytes  
  - Immature granulocytes (IG)  
  - Erythroblasts, Platelets  
- **Acquisition**: CellaVision DM96 analyzer (360 × 363 px)  
- **Staining**: May-Grünwald-Giemsa  
- **Reference**: Acevedo et al., *Comput. Methods Programs Biomed.* 2019

### 2. **Abnormal + Control Cells: AML-Cytomorphology_LMU – TCIA**
- **Source**: [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/)  
- **DOI**: [10.7937/tcia.2019.36f5o9ld](https://doi.org/10.7937/tcia.2019.36f5o9ld)  
- **Images**: 18,365 single-cell images  
- **Subjects**: 100 AML patients + 100 non-malignant controls  
- **Key Abnormal Class**: `MYB` (Myeloblasts)  
- **Other Classes**: Atypical lymphocytes (`LYT`), immature forms, and normal-like cells  
- **Acquisition**: Precipoint M8 scanner (~400×400 px, 14.14 px/µm)  
- **Reference**: Matek et al., *Nat. Mach. Intell.* 2019

> ⚠️ **Note**: Labels differ between datasets. For initial experiments, we recommend **binary classification**:  
> - **Normal**: All Acevedo classes  
> - **Abnormal**: Only `MYB` (myeloblasts) from TCIA

---

## 🧪 Project Stages

### Phase 1: Exploratory Data Analysis (EDA)
- Analyze class distributions in both datasets  
- Verify image quality, size consistency, and label integrity  
- Visualize sample cells per class

### Phase 2: Data Harmonization & Balancing
- Extract only relevant classes (e.g., `MYB` for abnormal)  
- Apply **paper-accurate augmentation** (per Acevedo):  
  - Horizontal/vertical flips  
  - ±60° rotation  
  - Up to 4 synthetic images for minority classes  
- Balance datasets to avoid bias

### Phase 3: Model Development
- **Binary classifier**: Normal vs. Blast (`MYB`)  
- **Multi-class extension** (if label mapping is validated)  
- Use **transfer learning** with **VGG-16 or InceptionV3** (as in Acevedo)  
- Fine-tune on balanced dataset

### Phase 4: Evaluation & Interpretation
- Report **accuracy, precision, recall, F1**  
- Generate **confusion matrices**  
- Compare performance on **balanced vs. unbalanced** data (as in Fig. 18 of Acevedo)

### Phase 5: Unsupervised Clustering
- Apply **UMAP / t-SNE** on CNN features  
- Cluster cells across datasets to discover morphological continua  
- Validate clusters against expert labels

### Phase 6: Reporting & Reproducibility
- Document all steps in Jupyter notebooks  
- Share model cards and visualizations in `reports/`  
- Ensure full reproducibility via `requirements.txt` and `environment.yml`

---

## 🚀 Setup
See **[SETUP.md](SETUP.md)** for environment configuration and data download instructions.

> 💡 **Important**: The `data/` folder is **not tracked by Git**. Each team member must download datasets locally.

---

## 🤝 Collaboration Policy
- **Never push directly to `main`**
- Create feature branches: `git checkout -b feature/eda-normal`
- Open **Pull Requests** for review
- Only merge after team consensus

---

## 📖 Citations (Required)

If you use this project or its data, please cite:

```bibtex
@article{acevedo2019,
  title={Recognition of peripheral blood cell images using convolutional neural networks},
  author={Acevedo, Andrea and Alférez, Santiago and Merino, Anna and Puigví, Laura and Rodellar, José},
  journal={Computer Methods and Programs in Biomedicine},
  volume={180},
  pages={105020},
  year={2019},
  publisher={Elsevier}
}

@article{matek2019,
  title={Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks},
  author={Matek, Christian and Schwarz, Simone and Spiekermann, Karsten and Marr, Carsten},
  journal={Nature Machine Intelligence},
  volume={1},
  number={11},
  pages={538--544},
  year={2019},
  publisher={Nature Publishing Group}
}

@article{boldu2021,
  title={A deep learning model (ALNet) for the diagnosis of acute leukaemia lineage using peripheral blood cell images},
  author={Boldú, Laura and Merino, Anna and Acevedo, Andrea and Molina, Angel and Rodellar, José},
  journal={Computer Methods and Programs in Biomedicine},
  volume={202},
  pages={105999},
  year={2021},
  publisher={Elsevier}
}
