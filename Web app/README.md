# 🌿 LeafScan — Plant Disease Detection App

A simple two-part app: a **Python/Flask backend** that runs your trained model,
and a **standalone HTML frontend** you open in any browser.

---

## 📁 File structure

```
leaf_disease_app/
├── app.py            ← Flask backend (loads your model)
├── index.html        ← Frontend (open in browser)
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## 🚀 Setup (3 steps)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your model next to app.py

```bash
# Copy your trained model into this folder:
cp /path/to/best_model.keras ./best_model.keras
```

> You can also set a custom path via env variable:
> `export MODEL_PATH=/path/to/best_model.keras`

### 3. Start the backend

```bash
python app.py
```

You should see:
```
✅ Model loaded from best_model.keras
 * Running on http://127.0.0.1:5000
```

---

## 🖥 Open the frontend

Just double-click **`index.html`** in your file manager (or drag it into Chrome/Firefox).

- The default API URL is `http://localhost:5000/predict`
- You can change it in the top bar if you deploy the backend elsewhere

---

## 🌱 How it works

1. You upload a leaf photo (drag & drop or click)
2. The frontend sends the image to Flask via HTTP POST
3. Flask resizes the image to 128×128, normalises it, runs your CNN
4. The top-5 predicted classes + confidence scores are returned and displayed

---

## 📋 Supported plants & diseases

The model was trained on the **PlantVillage** dataset — 38 classes covering:
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato,
Raspberry, Soybean, Squash, Strawberry, Tomato (healthy + diseased variants).

---

## 🛠 Troubleshooting

| Problem | Fix |
|---|---|
| CORS error in browser | Make sure `flask-cors` is installed and `app.py` is running |
| Model not found | Put `best_model.keras` next to `app.py` |
| Wrong predictions | Check that your training used `sorted(os.listdir(...))` for class order |
| Port 5000 taken | Change `app.run(port=5001)` in `app.py` and update the URL in the browser |
