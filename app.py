import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import hashlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="VeriMarket AI Deepfake Engine", layout="wide")

st.title("🛡️ VeriMarket AI Deepfake Detection Engine")
st.markdown("Deepfake model trained on real vs fake image datasets.")

# -----------------------------
# Load Deepfake Model
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "dima806/deepfake_vs_real_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

processor, model = load_model()

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)[0]

    fake_prob = probs[1].item()
    real_prob = probs[0].item()

    return round(fake_prob, 3), round(real_prob, 3)

# -----------------------------
# Blockchain Hash
# -----------------------------
def generate_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()

# -----------------------------
# Risk Badge
# -----------------------------
def risk_badge(score):
    if score > 0.75:
        return "🔴 HIGH RISK"
    elif score > 0.45:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("Upload Image for Deepfake Analysis", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    fake_prob, real_prob = predict(image)
    img_hash = generate_hash(image)

    with col2:
        st.subheader("AI Deepfake Classification")
        st.metric("Fake Probability", fake_prob)
        st.metric("Real Probability", real_prob)
        st.markdown(f"### {risk_badge(fake_prob)}")

        st.subheader("Blockchain Evidence Hash")
        st.code(img_hash)

    # Probability Chart
    st.subheader("Probability Distribution")
    fig, ax = plt.subplots()
    ax.bar(["Real", "Fake"], [real_prob, fake_prob])
    ax.set_ylim(0,1)
    st.pyplot(fig)

st.caption("Model trained on deepfake datasets. Production version integrates oracle validation & DAO governance.")
