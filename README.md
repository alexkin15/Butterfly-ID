# 🦋 Butterfly Species Identifier (Hong Kong)

This project focuses on **butterfly species classification** using **machine learning (PyTorch + ResNet50)**.  
It fills a **gap in the Hong Kong market**, where no existing app specifically helps users identify local butterfly species.

The project combines:
- **Deep learning (PyTorch, ResNet50)**
- **Computer vision (Torchvision)**
- **Interactive web deployment (Gradio)**

---

## 📂 Project Structure

- `train.py` &rarr; Training loop (ResNet50, checkpoint, validation)  
- `app.py` &rarr; Gradio web app for butterfly species identification  
- `species_info.json` &rarr; Metadata (Chinese/English names, order, family, genus, wingspan, etc.)  
- `best_model_state.pt` &rarr; Trained model weights (used by `app.py`)  
- `checkpoint.pth` &rarr; Checkpoint for resuming training

---

## 🚀 Workflow

1. 📂 **Dataset**  
2. 🔍 **Auto Class Detection**  
3. 🧠 **ResNet50 Model**  
4. 📊 **Training Loop**  
    - 💾 **Checkpoint Save every 15min** &rarr; `checkpoint.pth`
    - ✅ **Validation**  
        - 🌟 **Save Best Model if Val improves** &rarr; `best_model_state.pt`
5. 🚀 **Gradio Web App**

---

- **`checkpoint.pth`** &rarr; Resume training from the last checkpoint.
- **`best_model_state.pt`** &rarr; Used in the web app (`app.py`).

---

## ⚡ Features

- **Auto Class Detection:** Detects all butterfly species in your dataset.
- **Transfer Learning:** Trains a ResNet50 model.
- **Autosaves:** Saves checkpoints every 15 minutes to prevent data loss.
- **Best Model Saving:** Automatically saves the best model when validation accuracy improves.
- **Gradio Web App:** Deploys as an interactive web app for easy testing.

---

## 📦 Installation

**Clone the repository:**
```bash
git clone https://github.com/alexkin15/butterfly-id.git
cd butterfly-id

```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

**Run the training script:**
```bash
python train.py
```

---

## 🌐 Web App

**After training, launch the web app:**
```bash
python app.py
```

Upload a butterfly photo → get species prediction + info (Chinese + English).

---

## 📘 Future Work

- Expand dataset to include all 260+ butterfly species in Hong Kong
- Improve model accuracy with data augmentation
- Add a mobile app version for field use

---
## 🚀 Flow Chart
```mermaid
flowchart TD
    A[📂 Dataset] --> B[🔍 Auto Class Detection]
    B --> C[🧠 ResNet50 Model]
    C --> D[📊 Training Loop]
    D --> E[✅ Validation]
    D --> F[💾 Checkpoint Save every 15min]
    E --> G[🌟 Save Best Model if Val improves]
    F --> H[📂 checkpoint.pth]
    G --> I[📂 best_model_state.pt]
    H --> I
    I --> J[🚀 Gradio Web App]
