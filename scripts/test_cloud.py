"""Cloud service tester for image URL prediction.

Takes an image URL as input, downloads it, and sends it to the cloud prediction service.

Usage:
    python test_cloud.py <image_url>
"""

import sys
import requests

# AWS Load balancer url
CLOUD_URL = "http://vehicle-type-identifier-880645094.ap-south-1.elb.amazonaws.com/predict"


def test_image_url(image_url):
    """Download image from URL and send to cloud service."""
    print(f"Downloading image from: {image_url}")
    
    try:
        # Download image from URL
        img_response = requests.get(image_url, timeout=10)
        img_response.raise_for_status()
        
        # Send image to cloud service
        files = {"file": ("image.jpg", img_response.content, "image/jpeg")}
        resp = requests.post(CLOUD_URL, files=files, timeout=30)
        
        print(f"HTTP {resp.status_code}")
        try:
            print(resp.json())
        except Exception:
            print(resp.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_cloud.py <image_url>")
        sys.exit(1)
    
    image_url = sys.argv[1]
    test_image_url(image_url)
