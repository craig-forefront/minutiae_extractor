"""Downloads the pre-trained weights of the four networks from Hugging Face.

Run it from the repository root, with the dependencies installed:

    python scripts/install_model.py

It takes no arguments and writes into the directory models/ (about 1.6 GB):
CoarseNet.h5, FineNet.h5, ClassifyNet_6_classes.h5 and CoreNet.weights. The source is
the model repository tourmaline05/MinutiaeNet on huggingface.co. A second run
downloads only what is missing.
"""
from huggingface_hub import snapshot_download
from pathlib import Path

def download_minutiaenet(destination="models"):
    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Download every file of the model repository
    print("Downloading MinutiaeNet model from Hugging Face...")
    local_dir = snapshot_download(repo_id="tourmaline05/MinutiaeNet", local_dir=dest_path)

    print(f"Model downloaded successfully to: {local_dir}")

if __name__ == "__main__":
    download_minutiaenet()
