

TEST IT: **https://riskfraudie.streamlit.app/**


# 🧠 Deepfake Detection App

An interactive web application that detects whether a facial image is **real or AI-generated (deepfake)** using a deep learning model.

---

## 🚀 Live Demo

👉 https://riskfraudie.streamlit.app

---

## 📌 Project Overview

This project uses a trained deep learning model to classify images as **real** or **fake (deepfake)**.

The application is built with Streamlit and allows users to upload an image and receive an instant prediction.

⚠️ **Important:**
For accurate results, the image must be a **close-up of a human face**.

---

## 🧠 How It Works

* A pre-trained **ResNet18 model** is fine-tuned on deepfake detection data
* The model analyzes facial features and patterns typical of AI-generated images
* The app processes the uploaded image and outputs a prediction:

  * ✅ Real
  * ❌ Fake

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **Streamlit**
* **NumPy / PIL**

---

## 📂 Project Structure

```
app.py              # Streamlit app interface
model.py            # Model architecture
predict.py          # Prediction logic
config.py           # Configuration settings
best_resnet18_improved.pt   # Trained model weights
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/verimarket-deepfake.git
cd verimarket-deepfake
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Use Cases

* Detect AI-generated profile pictures
* Identify manipulated media
* Educational tool for understanding deepfake technology

---

## ⚠️ Limitations

* Works best on **clear, close-up facial images**
* Performance may decrease on:

  * Group photos
  * Low-quality images
  * Non-human subjects

---

## 📈 Future Improvements

* Improve model accuracy with larger datasets
* Add confidence scores and visual explanations
* Support video-based deepfake detection

---

## 👤 Author

Jim Vincent
🔗 https://www.linkedin.com/in/jimvincent1202/

---
