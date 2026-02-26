import streamlit as st
from PIL import Image
import numpy as np
from scipy import fftpack
import hashlib
import matplotlib.pyplot as plt

# -----------------------------
# Image Analysis Engine
# -----------------------------
def analyze_image(image):
    img_array = np.array(image)

    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    # Basic metrics
    variance = np.var(gray)
    gradient = np.abs(np.diff(gray, axis=0)).mean()
    noise = np.std(gray)

    # Frequency analysis
    fft = fftpack.fft2(gray)
    fft_shift = fftpack.fftshift(fft)
    magnitude = np.abs(fft_shift)
    high_freq_energy = np.mean(magnitude[30:-30, 30:-30])

    # Saturation anomaly
    if len(img_array.shape) == 3:
        saturation = np.std(img_array[:, :, 0]) + np.std(img_array[:, :, 1]) + np.std(img_array[:, :, 2])
    else:
        saturation = 0

    # Local contrast map
    local_contrast = np.mean(np.abs(np.diff(gray)))

    # Texture uniformity
    texture_uniformity = np.std(np.diff(gray, axis=1))

    # -------------------------
    # Enhanced Risk Scoring
    # -------------------------
    score = 0

    # Too smooth (AI smoothing)
    if variance < 500:
        score += 0.15

    # Unrealistic edge consistency
    if gradient < 3:
        score += 0.15

    # Suspicious frequency suppression
    if high_freq_energy < 20:
        score += 0.15

    # Artificial saturation patterns
    if saturation > 200:
        score += 0.15

    # Excessive local contrast (compositing)
    if local_contrast > 40:
        score += 0.2

    # Texture anomalies
    if texture_uniformity < 5:
        score += 0.2

    final_score = round(min(score, 1), 3)

    return {
        "score": final_score,
        "variance": round(variance, 2),
        "gradient": round(gradient, 2),
        "noise": round(noise, 2),
        "frequency": round(high_freq_energy, 2),
        "saturation": round(saturation, 2),
        "local_contrast": round(local_contrast, 2),
        "texture_uniformity": round(texture_uniformity, 2)
    }

# -----------------------------
# Risk Label
# -----------------------------
def risk_label(score):
    if score > 0.7:
        return "🔴 HIGH RISK"
    elif score > 0.4:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

# -----------------------------
# Blockchain Hash
# -----------------------------
def generate_hash(image):
    img_bytes = image.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()

# -----------------------------
# Risk Gauge
# -----------------------------
def risk_gauge(score):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.barh(0, score, color="red" if score > 0.7 else "orange" if score > 0.4 else "green")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_title("Manipulation Probability")
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="VeriMarket AI Forensics", layout="wide")

st.title("🛡️ VeriMarket – AI Image Forensics Engine")
st.markdown("Advanced structural & frequency-based manipulation detection (MVP).")

uploaded_file = st.file_uploader("Upload an image for forensic analysis", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = analyze_image(image)
    image_hash = generate_hash(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Risk Assessment")
        st.metric("Manipulation Risk Score", results["score"])
        st.write(risk_label(results["score"]))
        st.pyplot(risk_gauge(results["score"]))

    with col2:
        st.subheader("🔐 Blockchain Evidence Hash")
        st.code(image_hash)

    st.markdown("### 🔬 Forensic Indicators")
    st.write(f"Pixel Variance: {results['variance']}")
    st.write(f"Edge Gradient Intensity: {results['gradient']}")
    st.write(f"Noise Estimate: {results['noise']}")
    st.write(f"High-Frequency Energy: {results['frequency']}")
    st.write(f"Color Channel Inconsistency: {results['color_inconsistency']}")

    st.markdown("---")
    st.info("In production, this evidence hash would be anchored on-chain for immutable audit trails.")

st.markdown("---")
st.caption("⚠️ MVP structural forensic model. Production version integrates CNN-based deepfake AI + Oracle validation.")
