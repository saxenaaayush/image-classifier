from fastapi import FastAPI, File, UploadFile
import torch
from .model import load_model
from .utils import preprocess_image

app = FastAPI()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model_effnet.pth"
model = load_model(MODEL_PATH, DEVICE)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_tensor = preprocess_image(file).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).item()
        label = int(prob > 0.5)
    return {
        "prediction": "Cataract" if label else "Normal",
        "probability": round(prob, 4)
    }
