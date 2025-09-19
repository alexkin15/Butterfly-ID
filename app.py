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
# Config (local paths)
# -----------------
from pathlib import Path

class Config:
    root_dir = Path(__file__).resolve().parent  # project root
    data_dir = root_dir / "data" / "images"
    out_dir  = root_dir / "output"
    app_dir  = root_dir / "app"

cfg = Config()


device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------
# Load labels
# -----------------
LABELS_PATH = Path(cfg.app_dir) / "labels.txt"
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    idx_to_class = {i: line.strip() for i, line in enumerate(f)}

# -----------------
# Load species info (optional)
# -----------------
SPECIES_INFO_PATH = Path(cfg.app_dir) / "species_info.json"
if SPECIES_INFO_PATH.exists():
    with open(SPECIES_INFO_PATH, "r", encoding="utf-8") as f:
        species_info = json.load(f)
else:
    species_info = {}


# -----------------
# Load trained model
# -----------------
BEST_PATH = Path(cfg.out_dir) / "best_model_state.pt"

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model = model.to(device).eval()

# -----------------
# Prediction function
# -----------------
def format_info(info):
    """Format species_info dict into multiple lines."""
    if isinstance(info, dict):
        return "\n".join([f"- {k}: {v}" for k, v in info.items()])
    return info

def predict(image):
    try:
        img = image.convert("RGB")
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        x = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)

            num_classes = outputs.size(1)
            topk = min(5, num_classes)
            top_prob, top_idx = torch.topk(probs, topk)

        results = []
        for prob, idx in zip(top_prob[0], top_idx[0]):
            sp = idx_to_class[idx.item()]
            info = species_info.get(sp, "No info yet.")
            formatted_info = format_info(info)
            results.append(f"### {sp} ({prob.item()*100:.2f}%)\n{formatted_info}")

        return "\n\n".join(results)

    except Exception as e:
        return f"[ERROR] Prediction failed:\n {str(e)}"


# -----------------
# Species list function
# -----------------
def list_species():
    lines = []
    for idx in range(len(idx_to_class)):
        sp = idx_to_class[idx]
        info = species_info.get(sp, "No info yet.")
        lines.append(f"🦋 {sp}\n{info}")
    return "\n\n".join(lines)

# -----------------
# Gradio Tabs
# -----------------
with gr.Blocks() as demo:
    with gr.Tab("Identify"):
        gr.Markdown("### 🦋 Butterfly Species Identifier\nUpload a butterfly image to identify.")
        img_input = gr.Image(type="pil")
        output_text = gr.Markdown(label="Prediction Result")  # changed to Markdown
        img_input.change(fn=predict, inputs=img_input, outputs=output_text)

    with gr.Tab("Species List"):
        gr.Markdown("### ✅ Trained Species")
        btn = gr.Button("Refresh List")
        species_output = gr.Markdown()  # changed to Markdown
        btn.click(fn=list_species, inputs=None, outputs=species_output)

demo.launch(share=True)
