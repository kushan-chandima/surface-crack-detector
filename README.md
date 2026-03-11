# 🔍 Surface Crack Detector

A deep learning-based surface crack detection system using **CNN with Transfer Learning (MobileNetV2)**. Upload any surface image and instantly detect whether it contains a crack — with **~99% accuracy**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 🏗️ **Two model architectures** — MobileNetV2 (recommended) & Custom CNN
- 📊 **Full training pipeline** — data augmentation, callbacks, auto-save best model
- 📈 **Comprehensive evaluation** — confusion matrix, ROC curve, classification report
- 🌡️ **Grad-CAM visualization** — see exactly where the model detects cracks
- 🌐 **Streamlit web app** — drag-and-drop image upload for live predictions
- 📦 **GitHub-ready** — clean structure, tests, documentation

---

## 🧠 Approach & Methodology

### Why Transfer Learning?

Training a CNN from scratch requires **massive datasets** and **compute power**. Instead, we leverage **MobileNetV2** — a model pre-trained on 1.4M ImageNet images — and **fine-tune** it for crack detection. This gives us:

- ✅ **Higher accuracy** with less training data
- ✅ **Faster convergence** (fewer epochs needed)
- ✅ **Smaller model size** (~8 MB vs hundreds of MB)

### Pipeline Overview

```
┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌──────────────┐
│  Raw Images  │───▶│  Preprocessing   │───▶│  Model Training   │───▶│  Evaluation  │
│  (40K imgs)  │    │  + Augmentation   │    │  (MobileNetV2)    │    │  + Grad-CAM  │
└──────────────┘    └──────────────────┘    └───────────────────┘    └──────────────┘
```

| Stage | Details |
|---|---|
| **Data Split** | 70% train / 15% validation / 15% test |
| **Augmentation** | Rotation, horizontal/vertical flip, zoom, brightness shift |
| **Architecture** | MobileNetV2 (frozen) → GlobalAvgPool → Dropout → Dense(128) → Dropout → Dense(1, sigmoid) |
| **Callbacks** | EarlyStopping (patience=5), ReduceLROnPlateau, ModelCheckpoint |
| **Evaluation** | Confusion matrix, ROC-AUC, classification report, Grad-CAM heatmaps |

### Why Not a Custom CNN?

A custom CNN is also included for comparison. While it achieves ~97% accuracy, MobileNetV2 transfer learning reaches **~99% accuracy** with fewer trainable parameters and better generalization.

---

## 📋 Requirements

| Dependency | Version | Purpose |
|---|---|---|
| **Python** | 3.9+ | Runtime |
| TensorFlow | 2.15.0 | Deep learning framework |
| NumPy | 1.26.4 | Numerical computing |
| Pandas | 2.3.3 | Data manipulation |
| Matplotlib | 3.10.8 | Plotting & visualization |
| Seaborn | 0.13.2 | Statistical visualization |
| Scikit-learn | 1.7.2 | Metrics & preprocessing |
| Streamlit | 1.55.0 | Web app interface |
| Pillow | 12.1.1 | Image processing |
| OpenDatasets | 0.1.22 | Kaggle dataset download |
| Pytest | 9.0.2 | Unit testing |

> All versions are pinned in `requirements.txt` for full reproducibility.

---

## 🖼️ Demo & Screenshots

After training, you'll get outputs like:

- **Training Curves** — Loss and accuracy plots saved in `models/`
- **Confusion Matrix** — Visual breakdown of correct vs incorrect predictions
- **ROC Curve** — AUC score showing model discrimination ability
- **Grad-CAM Heatmaps** — Visual explanation of *where* the model detects cracks

```
Example Predictions:
┌─────────────────────────────────────────────────────────┐
│  Input Image    →    Prediction: Crack (99.7%)          │
│  [surface.jpg]  →    Grad-CAM: highlights crack region  │
└─────────────────────────────────────────────────────────┘
```

> 💡 Run `streamlit run app/app.py` to try it interactively with your own images!

---

## 📁 Project Structure

```
surface-crack-detector/
├── README.md                    # You are here
├── requirements.txt             # Pinned dependencies
├── setup.py                     # Package setup
├── LICENSE                      # MIT License
│
├── data/                        # Dataset (auto-downloaded)
├── models/                      # Saved trained models
│
├── src/
│   ├── config.py                # Configuration constants
│   ├── dataset.py               # Data loading & preprocessing
│   ├── model.py                 # Model architectures
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation & visualizations
│   └── predict.py               # Prediction & Grad-CAM
│
├── scripts/
│   ├── download_data.py         # Download dataset from Kaggle
│   └── run_training.py          # End-to-end training
│
├── app/
│   └── app.py                   # Streamlit web interface
│
└── tests/
    └── test_model.py            # Unit tests
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/surface-crack-detector.git
cd surface-crack-detector
```

### 2. Create & Activate Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
# source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 **GPU Support:** If you have an NVIDIA GPU, replace `tensorflow==2.15.0` with `tensorflow[and-cuda]==2.15.0` in `requirements.txt` for faster training.

### 4. Download Dataset

The project uses the [Surface Crack Detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) dataset from Kaggle (40,000 images).

**Option A — Automatic download:**
```bash
python scripts/download_data.py
```
You'll be prompted for your Kaggle credentials (get from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token).

**Option B — Manual download:**
1. Download from [Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)
2. Extract to `data/surface-crack-detection/`

---

## 🏋️ Training

### Full Training Pipeline

```bash
python scripts/run_training.py
```

### With Custom Parameters

```bash
# Use MobileNetV2 (recommended)
python scripts/run_training.py --model mobilenetv2 --epochs 20

# Use Custom CNN
python scripts/run_training.py --model custom_cnn --epochs 30

# Skip dataset download (if already downloaded)
python scripts/run_training.py --skip-download
```

Training will:
1. Download & organize the dataset (if needed)
2. Train the model with data augmentation
3. Save the best model to `models/best_model.keras`
4. Generate evaluation plots (confusion matrix, ROC curve, etc.)

---

## 🔍 Prediction

### Command Line

```bash
# Basic prediction
python -m src.predict --image path/to/image.jpg

# With Grad-CAM visualization
python -m src.predict --image path/to/image.jpg --gradcam

# Save visualization to specific path
python -m src.predict --image path/to/image.jpg --gradcam --output result.png
```

### Python API

```python
from src.predict import predict_image, predict_with_gradcam

# Simple prediction
result = predict_image("path/to/image.jpg")
print(f"Crack: {result['has_crack']}, Confidence: {result['confidence']:.2%}")

# With Grad-CAM
result = predict_with_gradcam("path/to/image.jpg", save_path="result.png")
```

---

## 🌐 Web App

Launch the interactive Streamlit web interface:

```bash
streamlit run app/app.py
```

Features:
- 📤 Drag-and-drop image upload
- 🔍 Real-time crack detection
- 🌡️ Grad-CAM heatmap visualization
- ⚙️ Adjustable confidence threshold

---

## 📊 Model Performance

| Model | Accuracy | AUC | Parameters |
|---|---|---|---|
| **MobileNetV2** (recommended) | **~99%** | **~0.99** | ~2.3M (trainable: ~200K) |
| Custom CNN | ~97% | ~0.97 | ~300K |

---

## 🧪 Testing

```bash
python -m pytest tests/test_model.py -v
```

---

## 🛠️ Configuration

All settings are in `src/config.py`:

| Setting | Default | Description |
|---|---|---|
| `MODEL_TYPE` | `"mobilenetv2"` | Model architecture |
| `IMG_SIZE` | `(224, 224)` | Input image size |
| `BATCH_SIZE` | `32` | Training batch size |
| `EPOCHS` | `30` | Max training epochs |
| `LEARNING_RATE` | `1e-4` | Initial learning rate |
| `DROPOUT_RATE` | `0.3` | Dropout for regularization |

---

## ❓ Troubleshooting

<details>
<summary><b>ModuleNotFoundError: No module named 'tensorflow'</b></summary>

Make sure your virtual environment is activated and dependencies installed:
```bash
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```
</details>

<details>
<summary><b>Kaggle API: Could not find kaggle.json</b></summary>

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Place the downloaded `kaggle.json` in:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

Or use **Option B** (manual download) from the Quick Start section.
</details>

<details>
<summary><b>CUDA / GPU not detected</b></summary>

1. Ensure compatible NVIDIA drivers are installed
2. Install GPU-enabled TensorFlow:
   ```bash
   pip install tensorflow[and-cuda]==2.15.0
   ```
3. Verify: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

> Training works fine on CPU — it's just slower (~2-3x compared to GPU).
</details>

<details>
<summary><b>Out of Memory (OOM) during training</b></summary>

Reduce batch size in `src/config.py`:
```python
BATCH_SIZE = 16  # Default is 32
```
</details>

<details>
<summary><b>Streamlit web app shows "Model not found"</b></summary>

Train the model first by running:
```bash
python scripts/run_training.py
```
The trained model will be saved to `models/best_model.keras`.
</details>

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** this repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** your changes: `git commit -m "Add your feature"`
4. **Push** to the branch: `git push origin feature/your-feature`
5. **Open** a Pull Request

Please make sure to:
- Follow the existing code style
- Add tests for new features
- Update documentation as needed

---

## 📚 References

- **Dataset:** [Surface Crack Detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) (Kaggle)
- **Reference Notebook:** [Concrete Crack Image Detection](https://www.kaggle.com/code/gcdatkin/concrete-crack-image-detection/notebook)
- **MobileNetV2:** [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- **Grad-CAM:** [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
