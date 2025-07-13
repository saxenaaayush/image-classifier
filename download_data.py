import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def download_and_extract():
    dataset_name = "nandanp6/cataract-image-dataset"
    output_dir = "data/raw"
    zip_file = os.path.join(output_dir, "cataract-image-dataset.zip")
    extract_dir = os.path.join(output_dir, "cataract-image-dataset")

    os.makedirs(output_dir, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()

    # print(f"Downloading {dataset_name}...")
    api.dataset_download_files(dataset_name, path=output_dir, unzip=False)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_file)
    # print(f"Extracted to: {extract_dir}")

if __name__ == "__main__":
    download_and_extract()
