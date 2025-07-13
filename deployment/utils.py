from PIL import Image
import io
from torchvision.transforms import functional as TF
from .transforms import val_transforms

def preprocess_image(upload_file):
    image_bytes = upload_file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return val_transforms(image).unsqueeze(0)
