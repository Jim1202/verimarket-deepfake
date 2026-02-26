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
# DARK THEME STYLING
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.block-container {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #FFFFFF;
}
.stMetric {
    background-color: #1C1F26;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Real CNN Backbone
# -----------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# Remove classifier head to extract embeddings
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# CNN Embedding Anomaly
# -----------------------------
def cnn_embedding_score(image):
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(img_tensor)

    features = features.flatten()
    embedding_variance = torch.var(features).item()

    # Lower variance often seen in synthetic smooth images
    score = 0
    if embedding_variance < 0.02:
        score += 0.4
    elif embedding_variance < 0.05:
        score += 0.2

    return score, embedding_variance

# -----------------------------
# Structural Forensic Layer
# -----------------------------
def structural_score(image):
    img_array = np.array(image)
    gray = np.mean(img_array, axis=2)

    variance = np.var(gray)
    gradient = np.abs(np.diff(gray, axis=0)).mean()

    fft = fftpack.fft2(gray)
    fft_shift = fftpack.fftshift(fft)
    magnitude = np.abs(fft_shift)
    high_freq = np.mean(magnitude[30:-30, 30:-30])

    score = 0.2

    if variance < 800: score += 0.2
    if gradient < 5: score += 0.2
    if high_freq < 40: score += 0.2

    return score

# -----------------------------
# Combined Score
# -----------------------------
def final_score(image):
    cnn_score, embedding_var = cnn_embedding_score(image)
    struct_score = structural_score(image)

    combined = min((0.6 * cnn_score + 0.4 * struct_score), 1)

    return round(combined, 3), cnn_score, struct_score, embedding_var

# -----------------------------
# Heatmap
# -----------------------------
def generate_heatmap(image):
    gray = np.mean(np.array(image), axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.pad(gx, ((0,0),(0,1)))
    gy = np.pad(gy, ((0,1),(0,0)))
    anomaly = gx + gy
    anomaly = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8)
    return anomaly

# -----------------------------
# Blockchain Hash
# -----------------------------
def generate_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()

# -----------------------------
# UI
# -----------------------------
st.title("🛡 VeriMarket Deepfake Intelligence Engine")
st.markdown("Real CNN + Forensic Hybrid Detection System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    score, cnn_s, struct_s, embed_var = final_score(image)
    heatmap = generate_heatmap(image)
    img_hash = generate_hash(image)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Manipulation Risk", score)
        st.write(f"CNN Feature Score: {round(cnn_s,3)}")
        st.write(f"Structural Score: {round(struct_s,3)}")
        st.write(f"Embedding Variance: {round(embed_var,6)}")

    with col2:
        st.write("Blockchain Evidence Hash")
        st.code(img_hash)

    st.subheader("Anomaly Visualization")
    fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap="inferno")
    ax.axis("off")
    st.pyplot(fig)

st.caption("Hybrid CNN-Forensic MVP. Production integrates GAN fingerprint training and FaceForensics++ fine-tuning.")
