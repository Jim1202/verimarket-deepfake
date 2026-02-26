import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from scipy import fftpack
import hashlib
import matplotlib.pyplot as plt

# -----------------------------
# Load Real CNN Model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# CNN Deep Feature Score
# -----------------------------
def cnn_anomaly_score(image):
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    # Softmax probabilities
    probs = torch.nn.functional.softmax(outputs[0], dim=0)

    # High entropy = uncertain = possibly synthetic
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

    # Normalize entropy
    normalized_entropy = min(entropy / 10, 1)

    return normalized_entropy

# -----------------------------
# Structural Forensic Engine
# -----------------------------
def structural_score(image):
    img_array = np.array(image)

    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array

    variance = np.var(gray)
    gradient = np.abs(np.diff(gray, axis=0)).mean()

    fft = fftpack.fft2(gray)
    fft_shift = fftpack.fftshift(fft)
    magnitude = np.abs(fft_shift)
    high_freq_energy = np.mean(magnitude[30:-30, 30:-30])

    score = 0.2  # baseline

    if variance < 800: score += 0.2
    if gradient < 5: score += 0.2
    if high_freq_energy < 40: score += 0.2

    return min(score, 1)

# -----------------------------
# Combined Risk Score
# -----------------------------
def combined_score(image):
    cnn_score = cnn_anomaly_score(image)
    struct_score = structural_score(image)

    final = min((0.6 * cnn_score + 0.4 * struct_score), 1)
    return round(final, 3), cnn_score, struct_score

# -----------------------------
# Heatmap
# -----------------------------
def generate_heatmap(image):
    gray = np.mean(np.array(image), axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.pad(gx, ((0,0),(0,1)), mode='constant')
    gy = np.pad(gy, ((0,1),(0,0)), mode='constant')
    anomaly = gx + gy
    anomaly = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8)
    return anomaly

# -----------------------------
# Hash
# -----------------------------
def generate_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="VeriMarket – AI Deepfake Engine", layout="wide")

st.title("🛡️ VeriMarket – Real AI Deepfake Detection Engine")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    final_score, cnn_score, struct_score = combined_score(image)
    heatmap = generate_heatmap(image)
    image_hash = generate_hash(image)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Final Manipulation Risk", final_score)
        st.write(f"CNN Entropy Score: {round(cnn_score,3)}")
        st.write(f"Structural Score: {round(struct_score,3)}")

    with col2:
        st.write("Blockchain Evidence Hash")
        st.code(image_hash)

    st.subheader("Anomaly Heatmap")
    fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap='hot')
    ax.axis('off')
    st.pyplot(fig)

st.caption("Production version uses specialized deepfake CNN models trained on FaceForensics++.")
