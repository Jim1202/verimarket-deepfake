# predict.py
# ─────────────────────────────────────────────────────────────
# Single-image and batch inference — this is the main module
# your Streamlit app will import.
#
# Usage (standalone):
#   python predict.py --image path/to/face.jpg
#
# Usage (from Streamlit):
#   from predict import DeepfakePredictor
#   predictor = DeepfakePredictor()
#   result    = predictor.predict(pil_image)
# ─────────────────────────────────────────────────────────────
import torchvision.transforms as transforms

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
import argparse
import numpy as np
import torch
from PIL import Image

from config import MODEL_SAVE_PATH, DEVICE, LABEL_MAP_INV
from model   import load_trained_model


class DeepfakePredictor:
    """
    Stateful predictor — loads the model once and exposes a simple
    predict() method for single PIL images.

    Designed to be instantiated once and cached (e.g. with
    @st.cache_resource in Streamlit) to avoid reloading weights
    on every request.

    Args:
        model_path : path to the .pt checkpoint (default from config.py)
        threshold  : decision threshold for the fake class.
                     - Pass a float directly (e.g. 0.35) if you have already
                       run evaluate.py and noted the optimal threshold.
                     - Leave as None to use the default 0.5.
        device     : torch.device (default from config.py)
    """

    def __init__(self,
                 model_path: str = MODEL_SAVE_PATH,
                 threshold: float = None,
                 device=DEVICE):

        self.device    = device
        self.threshold = threshold if threshold is not None else 0.5
        self.model     = load_trained_model(model_path, device)
        print(f"Predictor ready | threshold: {self.threshold:.3f}")

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single PIL image.

        Args:
            image: PIL.Image (any mode; converted to RGB internally)

        Returns:
            {
              "label"     : "real" or "fake",
              "confidence": float (0–1),  # confidence in the predicted label
              "prob_fake" : float (0–1),  # raw P(fake) score
              "prob_real" : float (0–1),  # raw P(real) score
            }
        """
        image  = image.convert("RGB")
        tensor = inference_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs   = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

        prob_real = float(probs[0])
        prob_fake = float(probs[1])
        label     = "fake" if prob_fake >= self.threshold else "real"
        confidence = prob_fake if label == "fake" else prob_real

        return {
            "label"     : label,
            "confidence": round(confidence, 4),
            "prob_fake" : round(prob_fake,  4),
            "prob_real" : round(prob_real,  4),
        }

    def predict_batch(self, images: list) -> list:
        """
        Run inference on a list of PIL images.
        Returns a list of result dicts in the same order as input.
        """
        return [self.predict(img) for img in images]

    def set_threshold(self, threshold: float):
        """Update the decision threshold without reloading the model."""
        self.threshold = threshold
        print(f"Threshold updated to {threshold:.3f}")


# ── Standalone CLI ────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detector — single image inference")
    parser.add_argument("--image",     type=str, required=True,  help="Path to input image")
    parser.add_argument("--model",     type=str, default=MODEL_SAVE_PATH, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5,  help="Decision threshold for fake class")
    return parser.parse_args()


def main():
    args      = parse_args()
    image     = Image.open(args.image)
    predictor = DeepfakePredictor(model_path=args.model, threshold=args.threshold)
    result    = predictor.predict(image)

    print("\n── Prediction ──────────────────────")
    print(f"  Label      : {result['label'].upper()}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"  P(fake)    : {result['prob_fake']*100:.1f}%")
    print(f"  P(real)    : {result['prob_real']*100:.1f}%")
    print(f"  Threshold  : {args.threshold}")
    print("────────────────────────────────────")


if __name__ == "__main__":
    main()
