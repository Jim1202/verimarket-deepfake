import streamlit as st
from PIL import Image
import numpy as np

# -----------------------------
# Image Risk Analysis
# -----------------------------
def analyze_image(image):

    img_array = np.array(image)

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    # Pixel variance (over-smoothing detection)
    variance = np.var(gray)

    # Edge intensity proxy
    gradient = np.abs(np.diff(gray, axis=0)).mean()

    # Noise estimate
    noise_level = np.std(gray)

    # Risk scoring heuristic
    score = 0

    if variance < 500:
        score += 0.3

    if gradient < 5:
        score += 0.3

    if noise_level < 10:
        score += 0.2

    # Normalize
    final_score = min(score, 1)

    return round(final_score, 3), variance, gradient, noise_level


def risk_label(score):
    if score > 0.6:
        return "🔴 HIGH MANIPULATION RISK"
    elif score > 0.3:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="VeriMarket – Image Deepfake Detector")

st.title("🖼 VeriMarket – Image Deepfake Detector")
st.markdown("MVP Structural Image Manipulation Detection Engine")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    score, variance, gradient, noise = analyze_image(image)

    st.subheader("📊 Deepfake Risk Assessment")
    st.metric("Manipulation Risk Score", score)
    st.write(risk_label(score))

    st.markdown("### 🔬 Structural Indicators")
    st.write(f"Pixel Variance: {round(variance,2)}")
    st.write(f"Edge Gradient Intensity: {round(gradient,2)}")
    st.write(f"Noise Estimate: {round(noise,2)}")

st.markdown("---")
st.caption("⚠️ MVP heuristic detector. Production version integrates CNN-based forensic AI.")
