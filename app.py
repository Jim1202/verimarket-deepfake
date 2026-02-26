import streamlit as st
from PIL import Image
import numpy as np
from scipy import fftpack

# -----------------------------
# Advanced Image Analysis
# -----------------------------

def analyze_image(image):
    img_array = np.array(image)

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    # ---------------------
    # 1. Pixel Variance
    # ---------------------
    variance = np.var(gray)

    # ---------------------
    # 2. Edge Intensity
    # ---------------------
    gradient = np.abs(np.diff(gray, axis=0)).mean()

    # ---------------------
    # 3. Noise Estimate
    # ---------------------
    noise = np.std(gray)

    # ---------------------
    # 4. Frequency Analysis (AI smoothing detection)
    # ---------------------
    fft = fftpack.fft2(gray)
    fft_shift = fftpack.fftshift(fft)
    magnitude = np.abs(fft_shift)
    high_freq_energy = np.mean(magnitude[10:-10, 10:-10])

    # ---------------------
    # 5. Color Channel Inconsistency
    # ---------------------
    if len(img_array.shape) == 3:
        r_var = np.var(img_array[:, :, 0])
        g_var = np.var(img_array[:, :, 1])
        b_var = np.var(img_array[:, :, 2])
        color_inconsistency = abs(r_var - g_var) + abs(g_var - b_var)
    else:
        color_inconsistency = 0

    # ---------------------
    # Risk Scoring Logic
    # ---------------------
    score = 0

    if variance < 400:
        score += 0.2

    if gradient < 4:
        score += 0.2

    if noise < 15:
        score += 0.2

    if high_freq_energy < 20:
        score += 0.2

    if color_inconsistency < 500:
        score += 0.2

    final_score = round(min(score, 1), 3)

    return {
        "score": final_score,
        "variance": round(variance, 2),
        "gradient": round(gradient, 2),
        "noise": round(noise, 2),
        "frequency_energy": round(high_freq_energy, 2),
        "color_inconsistency": round(color_inconsistency, 2)
    }


def risk_label(score):
    if score > 0.7:
        return "🔴 HIGH MANIPULATION RISK"
    elif score > 0.4:
        return "🟠 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="VeriMarket – Image Deepfake Detector", layout="centered")

st.title("🖼 VeriMarket – Enhanced Image Deepfake Detector")
st.markdown("Advanced structural & frequency-based AI image forensics (MVP).")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = analyze_image(image)

    st.subheader("📊 Deepfake Risk Assessment")
    st.metric("Manipulation Risk Score", results["score"])
    st.write(risk_label(results["score"]))

    st.markdown("### 🔬 Forensic Indicators")
    st.write(f"Pixel Variance: {results['variance']}")
    st.write(f"Edge Gradient Intensity: {results['gradient']}")
    st.write(f"Noise Estimate: {results['noise']}")
    st.write(f"High-Frequency Energy: {results['frequency_energy']}")
    st.write(f"Color Channel Inconsistency: {results['color_inconsistency']}")

st.markdown("---")
st.caption(
    "⚠️ MVP forensic model. Production system integrates CNN-based deepfake detection and blockchain evidence anchoring."
)
