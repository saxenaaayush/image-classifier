# Cataract Detection Using EfficientNet + FastAPI

A deep learning-based binary classifier that detects cataracts from retinal images with high clinical precision. Built using PyTorch and deployed with FastAPI for seamless image inference.

---

## Project Structure

image-classifier/
├── assets/ # Trained model weights (.pth)
├── data/ # (Optional) Local data (from Kaggle)
├── deployment/
│ ├── main.py # FastAPI app entry point
│ ├── model.py # Model architecture and loader
│ ├── utils.py # Image preprocessing functions
├── notebooks/
│ ├── 01_exploration.ipynb # Dataset exploration and CLAHE/Gamma tests
│ ├── 02_train_final.ipynb # Final EfficientNet training + evaluation
├── requirements.txt # Project dependencies
├── README.md # 

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image-classifier.git
cd image-classifier

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt


Dataset: Kaggle: Cataract Image Dataset

kaggle datasets download -d nandanp6/cataract-image-dataset
unzip cataract-image-dataset.zip -d data/


data/
└── raw/
    └── processed_images/
        ├── train/
        │   ├── cataract/
        │   └── normal/
        └── test/
            ├── cataract/
            └── normal/

DATA_ROOT = "data/raw/processed_images/"


### Running the API
1. Launch FastAPI server
bash
Copy
Edit
uvicorn deployment.main:app --reload
2. Test the API
Go to: http://127.0.0.1:8000/docs
Upload an image via Swagger UI.

Example Output
json
{
  "prediction": "Cataract",
  "probability": 0.9741
}
