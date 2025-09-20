# 🦋 Model Setup Guide

The trained model file **`best_model_state.pt`** is required to run the Butterfly-ID app.  
Since it is too large to store on GitHub, please download it from Google Drive and place it manually.

---

## 1. Download the Model
Click the link below to download the trained model:

👉 [Download best_model_state.pt](https://drive.google.com/file/d/1sHueJcP7yT95_5s0XdEb1JCk_ZkXE4U0/view?usp=drive_link)

---

## 2. Create the `output/` Folder (if not exists)
Inside the cloned project folder, check if the `output/` directory exists:
```
butterfly-id/
│── app/
│── data/
│── output/ <-- place model file here
│── app.py
│── train.py
│── requirements.txt
```


If `output/` does not exist, create it manually.

---

## 3. Move the Model File
Place the downloaded **`best_model_state.pt`** file into the `output/` folder:
```
butterfly-id/output/best_model_state.pt
```

---

## 4. Verify
Your folder should look like this:
```
butterfly-id/
│── app/
│── data/
│── output/
│ └── best_model_state.pt
│── app.py
│── train.py
│── requirements.txt
```


---

## 5. Run the App
Now you can run the app normally:

If everything is set up correctly, the web app will launch and use the trained model to make predictions.


---
