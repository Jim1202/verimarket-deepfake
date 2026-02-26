import streamlit as st
from PIL import Image
import numpy as np
from scipy import fftpack
import hashlib
import matplotlib.pyplot as plt

# -----------------------------
# Image Forensics Engine
# -----------------------------
def analyze_image(image):
    img_array = np.array(image)

    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    variance = np.var(gray)
    gradient = np.abs(np.diff(gray, axis=0)).mean()
    noise = np.std(gray)

    # Frequency domain
    fft = fftpack.fft2(gray)
    fft_shift = fftpack.fftshift(fft)
    magnitude = np.abs(fft_shift)
    high_freq_energy = np.mean(magnitude[30:-30, 30:-30])

    # Saturation
    if len(img_array.shape) == 3:
        saturation = (
            np.std(img_array[:, :, 0]) +
            np.std(img_array[:, :, 1]) +
            np.std(img_array[:, :, 2])
        )
    else:
        saturation = 0

    # Local contrast
    local_contrast = np.mean(np.abs(np.diff(gray)))

    # Texture uniformity
    texture_uniformity = np.std(np.diff(gray, axis=1))

    # Risk scoring
    score = 0
    if variance < 500: score += 0.15
    if gradient < 3: score += 0.15
    if high_freq_energy < 20: score += 0.15
    if saturation > 200: score += 0.15
    if local_contrast > 40: score += 0.2
    if texture_uniformity < 5: score += 0.2

    final_score = round(min(score, 1), 3)

    return {
        "score": final_score,
        "variance": round(variance, 2),
        "gradient": round(gradient, 2),
        "noise": round(noise, 2),
        "frequency": round(high_freq_energy, 2),
        "saturation": round(saturation, 2),
        "local_contrast": round(local_contrast, 2),
        "texture_uniformity": round(texture_uniformity, 2),
        "gray": gray
    }


# -----------------------------
# Visual Anomaly Heatmap
# -----------------------------
def generate_heatmap(gray):
    # Compute gradient map
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))

    # Pad to original size
    gx = np.pad(gx, ((0,0),(0,1)), mode='constant')
    gy = np.pad(gy, ((0,1),(0,0)), mode='constant')

    anomaly_map = gx + gy

    # Normalize
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    return anomaly_map


# -----------------------------
# Blockchain Hash
# -----------------------------
def generate_hash(image):
    img_bytes = image.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()


# -----------------------------
# Risk Label
# -----------------------------
def risk_label(score):
    if score > 0.7:
        return "🔴 HIGH MANIPULATION RISK"
    elif score > 0.4:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"


# -----------------------------
# Risk Gauge
# -----------------------------
def risk_gauge(score):
    fig, ax = plt.subplots(figsize=(4, 2))
    color = "red" if score > 0.7 else "orange" if score > 0.4 else "green"
    ax.barh(0, score, color=color)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_title("Manipulation Probability")
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="VeriMarket – AI Image Forensics", layout="wide")

st.title("🛡️ VeriMarket – AI Image Forensics Engine")
st.markdown("Structural + Frequency + Visual Anomaly Detection (MVP).")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = analyze_image(image)
    heatmap = generate_heatmap(results["gray"])
    image_hash = generate_hash(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Deepfake Risk Assessment")
        st.metric("Manipulation Risk Score", results["score"])
        st.write(risk_label(results["score"]))
        st.pyplot(risk_gauge(results["score"]))

    with col2:
        st.subheader("🔐 Blockchain Evidence Anchor")
        st.code(image_hash)

    st.markdown("### 🔥 Visual Anomaly Heatmap")

    fig2, ax2 = plt.subplots()
    ax2.imshow(heatmap, cmap='hot')
    ax2.axis('off')
    st.pyplot(fig2)

    st.markdown("### 🔬 Forensic Indicators")
    st.write(f"Pixel Variance: {results['variance']}")
    st.write(f"Edge Gradient Intensity: {results['gradient']}")
    st.write(f"High-Frequency Energy: {results['frequency']}")
    st.write(f"Color Saturation Level: {results['saturation']}")
    st.write(f"Local Contrast: {results['local_contrast']}")
    st.write(f"Texture Uniformity: {results['texture_uniformity']}")

    st.info("In production, anomaly clusters are validated via CNN-based detection and recorded on-chain.")

st.markdown("---")
st.caption("⚠️ MVP forensic engine. Production integrates GAN fingerprint detection + Oracle validation + DAO consensus.")
