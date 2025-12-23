"""Simple local tester for the running prediction server.

Sends each image from ../test_images/ as a multipart/form-data POST to /predict
and prints the responses.

Usage:
    pipenv run python scripts/test_local.py

Make sure the server is running (default: http://127.0.0.1:8000). If
`requests` is not installed in your Pipenv, install it with:
    pipenv install requests
"""

import os
import sys
from pathlib import Path

SERVER_URL = os.environ.get("PREDICT_SERVER_URL", "http://127.0.0.1:8000/predict")

# Resolve repo root and test images directory relative to this file
repo_root = Path(__file__).resolve().parents[1]
images_dir = repo_root / "test_images"

if not images_dir.is_dir():
    print(f"Test images directory not found: {images_dir}")
    sys.exit(1)

images = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
if not images:
    print(f"No images found in {images_dir}")
    sys.exit(1)

try:
    import requests
except Exception:
    print("The 'requests' package is required to run this test. Install it with: pipenv install requests")
    sys.exit(1)


def test_all_images():
    print(f"Sending {len(images)} images to {SERVER_URL}")

    for img_path in images:
        print(f"\n---\nUploading: {img_path.name}")
        try:
            with open(img_path, "rb") as f:
                files = {"file": (img_path.name, f, "image/jpeg")}
                resp = requests.post(SERVER_URL, files=files, timeout=10)

            print(f"HTTP {resp.status_code}")
            try:
                print(resp.json())
            except Exception:
                print(resp.text)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")


if __name__ == "__main__":
    test_all_images()
