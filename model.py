import torch
import torch.nn as nn
import torchvision.models as models
from config import MODEL_SAVE_PATH, DEVICE

def load_trained_model(model_path=MODEL_SAVE_PATH, device=DEVICE):
    model = models.resnet18(weights=None)

    # 🔥 This line was missing
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    return model