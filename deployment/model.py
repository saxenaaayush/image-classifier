import torch
import torch.nn as nn
from torchvision import models

def build_model(backbone="efficientnet_b4", pretrained=False):
    model = getattr(models, backbone)(pretrained=pretrained)
    if backbone.startswith("efficientnet"):
        in_feats = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feats, 1)
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model

def load_model(weights_path, device):
    model = build_model("efficientnet_b4", pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
