# -----------------------------
# Deployment Configuration
# -----------------------------

MODEL_SAVE_PATH = "best_resnet18_improved.pt"

DEVICE = "cpu"

LABEL_MAP = {
    "real": 0,
    "fake": 1
}

LABEL_MAP_INV = {
    0: "real",
    1: "fake"
}

DEFAULT_THRESHOLD = 0.35