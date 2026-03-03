import streamlit as st
from predict import DeepfakePredictor
from PIL import Image

st.set_page_config(page_title="VeriMarket Deepfake Engine", layout="centered")

st.title("🛡️ VeriMarket Deepfake Detection Engine")

@st.cache_resource
def load_predictor():
    return DeepfakePredictor(threshold=0.35)

predictor = load_predictor()

uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    result = predictor.predict(image)

    st.image(image, caption="Uploaded Image", width=350)

    st.metric("Prediction", result["label"].upper())
    st.metric("Confidence", f"{result['confidence']*100:.1f}%")

    st.progress(result["prob_fake"], text=f"P(fake): {result['prob_fake']*100:.1f}%")