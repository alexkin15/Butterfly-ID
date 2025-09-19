import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# ========================
# Config
# ========================
MODEL_PATH = "output/best_model_state.pt"
IMG_SIZE = 224

# ========================
# Load trained model
# ========================
@st.cache_resource  # cache to avoid reloading every time
def load_model():
    model = models.resnet50(weights=None)  # same arch as training
    num_classes = 260  # update based on your dataset
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ========================
# Preprocessing
# ========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ========================
# Web App UI
# ========================
st.title("🦋 Butterfly Species Classifier")
st.write("Upload an image of a butterfly, and the model will predict its species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_t = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)

    st.success(f"Predicted Species ID: {pred.item()}")
    st.write("⚠️ Map this ID back to actual species name using train_ds.classes")
