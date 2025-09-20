# 🦋 How to Use the Butterfly-ID Gradio App

This guide explains how to run the **Butterfly-ID** app locally after cloning the repository.

---

## 1. Prerequisites

- Install **Python 3.9+** (Python 3.10 or 3.11 recommended).
- (Optional but recommended) Create a virtual environment to keep dependencies isolated:


```
python -m venv jdaEnv
source jdaEnv/bin/activate   # Linux/Mac
jdaEnv\Scripts\activate      # Windows PowerShell
Install the required packages:
```
```
pip install -r requirements.txt
```
---

**2. Download the Model**
Follow the "Model Setup Guide.md" to download and place the trained model
best_model_state.pt into the output/ folder.

---

**3. Run the App**
From inside the project folder, launch the Gradio app:
```
python app.py
```
You will see a local URL in the terminal (e.g., http://127.0.0.1:7860).
Open this link in your browser.

---

**4. Features**

🔍 Identify Tab
Upload a butterfly photo (JPG/PNG).

The app will return the top-5 predicted species, along with:

Confidence percentage

Chinese/English names

Family, genus, order

Wingspan

Notes (if available)

📋 Species List Tab
View all trained butterfly species.

Each entry includes:

Species name

Basic biological info

---

**5. Example Workflow**

Start the app with 
```python app.py.```

1. Go to the Identify Tab.

2. Upload a butterfly photo.

3. Read the prediction results and species details.

4. Switch to the Species List Tab to see all species included in training.

---

**6. Notes**
The app runs locally — no internet connection required after setup.

If you want to share the app temporarily with others, you can launch it with:

```
python app.py --share
```
This will generate a public Gradio link (valid for a limited time).

✅ You’re now ready to explore Hong Kong’s butterflies using AI! 🦋
