# =============================
# Butterfly ID - Gradio Web App with Species List
# =============================

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from pathlib import Path
import gradio as gr

# -----------------
# Config
# -----------------
class Config:
    # Define main directories for project
    root_dir = "/content/drive/MyDrive/butterfly-id"   # Root folder in Google Drive
    data_dir = root_dir + "/data/images"               # Image dataset folder
    out_dir  = root_dir + "/output"                    # Model outputs/checkpoints
    app_dir  = root_dir + "/app"                       # App-related files (labels, species info, etc.)

cfg = Config()

# Select device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------
# Load labels
# -----------------
# labels.txt should contain one class name per line (species name)
# We map index -> class name for predictions
LABELS_PATH = Path(cfg.app_dir) / "labels.txt"
with open(LABELS_PATH, "r") as f:
    idx_to_class = {i: line.strip() for i, line in enumerate(f)}

# -----------------
# Load species info (optional)
# -----------------
# species_info.json can contain additional info about each butterfly species
# Example: {"Monarch": "Common in North America", "Swallowtail": "Large colorful wings"}
SPECIES_INFO_PATH = Path(cfg.app_dir) / "species_info.json"
if SPECIES_INFO_PATH.exists():
    with open(SPECIES_INFO_PATH, "r") as f:
        species_info = json.load(f)
else:
    species_info = {}

# -----------------
# Load trained model
# -----------------
# Load the ResNet50 backbone (must match training architecture)
BEST_PATH = Path(cfg.out_dir) / "best_model_state.pt"

model = models.resnet50(weights=None)  # No pretrained weights here, load your own
model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))  # Adjust FC layer to match species count
model.load_state_dict(torch.load(BEST_PATH, map_location=device))  # Load trained weights
model = model.to(device).eval()  # Move model to device & set eval mode

# -----------------
# Prediction function
# -----------------
def predict(image):
    """
    Run butterfly species prediction on an uploaded image.
    Returns top-5 predictions with probabilities and species info.
    """
    try:
        # Convert input image to RGB
        img = image.convert("RGB")

        # Apply same transforms as training (resize + tensor conversion)
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        x = tf(img).unsqueeze(0).to(device)  # Add batch dimension

        # Forward pass
        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)  # Convert to probabilities

            # Pick top-5 predictions (or fewer if dataset <5 classes)
            num_classes = outputs.size(1)
            topk = min(5, num_classes)
            top_prob, top_idx = torch.topk(probs, topk)

        # Format prediction results
        results = []
        for prob, idx in zip(top_prob[0], top_idx[0]):
            sp = idx_to_class[idx.item()]  # Convert index to species name
            info = species_info.get(sp, "No info yet.")  # Add description if available
            results.append(f"{sp} ({prob.item()*100:.2f}%)\n{info}")

        return "\n\n".join(results)

    except Exception as e:
        return f"[ERROR] Prediction failed:\n {str(e)}"

# -----------------
# Species list function
# -----------------
def list_species():
    """
    Return full list of all trained species (with optional extra info).
    Useful for checking which classes are available in the trained model.
    """
    lines = []
    for idx in range(len(idx_to_class)):
        sp = idx_to_class[idx]
        info = species_info.get(sp, "No info yet.")
        lines.append(f"🦋 {sp}\n{info}")
    return "\n\n".join(lines)

# -----------------
# Gradio Tabs
# -----------------
# Build a 2-tab Gradio interface:
#   - Tab 1: Upload image -> get prediction
#   - Tab 2: View all species in training set
with gr.Blocks() as demo:
    # First tab: Prediction
    with gr.Tab("Identify"):
        gr.Markdown("### 🦋 Butterfly Species Identifier\nUpload a butterfly image to identify.")
        img_input = gr.Image(type="pil")  # Upload widget
        output_text = gr.Textbox(label="Prediction Result")  # Display output
        img_input.change(fn=predict, inputs=img_input, outputs=output_text)

    # Second tab: Species list
    with gr.Tab("Species List"):
        gr.Markdown("### ✅ Trained Species")
        btn = gr.Button("Refresh List")
        species_output = gr.Textbox(label="All trained species", lines=20)
        btn.click(fn=list_species, inputs=None, outputs=species_output)

# Launch the Gradio app (with public sharing enabled)
demo.launch(share=True)
