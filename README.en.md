# SERSPeakFinder

A PyQt6 GUI for **detecting peak presence** in SERS (Raman) spectra.  
It covers the full workflow: split mapped data into per-coordinate CSVs, manual labeling, model training, batch detection, result comparison, and peak-region integration.

[한국어](README.md) | [English](README.en.md)

## Workflow

| Tab | Function |
|---|---|
| **Splite Plot Data** | Split raw txt/csv into per-coordinate `(x_y).csv` spectra |
| **Plot Classifier** | Manual labeling: peak (1) / no peak (0) (`A` / `D`) |
| **CNN Training** | Train binary classifier with live loss/accuracy plots |
| **Run (Peak Detection)** | Batch inference with a trained `.pth` model |
| **Compare with Prev. Plot** | Compare model vs manual labels and export a report |
| **Interpolation** | Savitzky–Golay filter + linear baseline interpolation for peak-area integration |

## Model

Despite the historical “CNN” wording in the UI, the network is a **fully connected MLP**, not a CNN.

- Input: spectrum CSV → `StandardScaler` → flattened 1D vector
- Architecture: Linear `input → 128 → 64 → 32 → 1` with ReLU / Dropout / Sigmoid
- Loss: BCELoss
- Implementation: `BinaryClassifier` in `touch_training.py`

Example pretrained weights: `small_ng_samp_m*.pth` (default config uses `m6`)

## Requirements

- Windows 10+
- Python 3.8+
- Packages: `PyQt6`, `numpy`, `pandas`, `torch`, `scipy`, `matplotlib`, `scikit-learn`, `natsort`, `torchsummary`, `qbstyles`

Uses CUDA when available; otherwise falls back to CPU.

## Install & Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install PyQt6 numpy pandas torch scipy matplotlib scikit-learn natsort torchsummary qbstyles
python main.py
```

Paths and hyperparameters are stored in `config.ini` and restored on the next launch.

## Key Files

| File | Description |
|---|---|
| `main.py` | GUI entry point, training/inference threads, tab logic |
| `ui_main.py` / `ui_main.ui` | Qt Designer UI |
| `touch_training.py` | Dataset, model, train/eval helpers |
| `matplotlibwidget.py` | Matplotlib canvas widget |
| `matplotlib_toolbar.py` | Plot toolbar |
| `config.ini` | Recent paths and parameters |
| `*.pth` | Trained model weights |
| `main.spec` | PyInstaller packaging config |

## Data Formats

- **Raw**: tab-separated txt/csv (coordinate columns + intensity per wavenumber)
- **Spectrum CSV**: one file per coordinate after splitting
- **Label CSV**: `filename, peak_flag(0/1)`
- **Detection CSV**: filename, prediction, confidence probability

## Notes

- Update `config.ini` paths for your local or NAS data locations.
