# -----------------------------
# Model Path
# -----------------------------

MODEL_SAVE_PATH = "best_resnet18_improved.pt"

# -----------------------------
# Device
# -----------------------------

DEVICE = "cpu"

# -----------------------------
# Label Mapping
# -----------------------------

LABEL_MAP = {
    "real": 0,
    "fake": 1
}

LABEL_MAP_INV = {
    0: "real",
    1: "fake"
}

# -----------------------------
# Threshold
# -----------------------------

DEFAULT_THRESHOLD = 0.35