# Setup Guide

Follow these steps to configure your development environment.

## 🛠️ Requirements
- Python 3.10+
- `git`
- Internet access (to download datasets)

## 🧪 Using `venv` (Recommended)

### 1. Clone the repository
```bash
git clone https://github.com/n0721059/blood-cell-classification.git
cd blood-cell-classification
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
```

- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  venv\Scripts\Activate.ps1
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Register Jupyter kernel
```bash
python -m ipykernel install --user --name=blood-cell --display-name="Blood Cell Env"
```

### 5. Launch Jupyter
```bash
jupyter notebook
# or
jupyter lab
```

> In your notebook: `Kernel → Change kernel → Blood Cell Env`

## 💻 VS Code Tips
1. Open the project folder in VS Code
2. Install **Python** stable version and **Jupyter** extensions
3. Press `Cmd+Shift+P` → "Python: Select Interpreter" → choose `./venv/bin/python`
4. Open any `.ipynb` file — select the "Blood Cell Env" kernel

## 📥 Download Datasets

### Normal Cells
- Source: [Mendely Data Set](https://data.mendeley.com/datasets/snkd93bnjr/1)
- Action: Download → unzip into `data/raw/normal_cells/`

### Abnormal Cells
- Source: [The Cancer Imaging Archive](https://www.googledrive.com)
- Action: Download → unzip into `data/raw/abnormal_cells/`

> ✅ Final structure:
> ```
> data/raw/normal_cells/PBC_dataset_normal_DIB/basophil/...
> data/raw/abnormal_cells/BCCD/... 
> ```

## ⚠️ Important Notes
- The `data/` folder is **ignored by Git** — each team member must download it locally.
- Always **clear notebook outputs** before committing:
  - In Jupyter: `Kernel → Restart & Clear Output`
