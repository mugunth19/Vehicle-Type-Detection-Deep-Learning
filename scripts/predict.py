import onnxruntime as ort
import numpy as np
import os
from PIL import Image

# Server dependencies
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO

# Resolve repository root from this script's directory and use model from `model_file/`
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_filename = "vehicle_identifier_mobilenet_v2.onnx"
onnx_model_path = os.path.join(repo_root, "model_file", model_filename)

if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}. Place the model in the model_file/ directory at the repo root: {repo_root}")

# Load the ONNX model
sess = ort.InferenceSession(onnx_model_path)

# Get input and output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Define preprocessing parameters (must match training transforms - specifically validation transforms)
input_size = 224
_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Lightweight PIL + NumPy preprocessing to avoid heavy torchvision dependency
def preprocess_pil(img, input_size=input_size):
    # Resize with bilinear interpolation
    img = img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # H, W, C in [0,1]
    # Normalize per channel
    arr = (arr - _mean) / _std
    # Convert HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    return arr

# Define class labels (must match the order used during training)
class_labels = ['hatchback', 'motorcycle', 'pickup', 'sedan', 'suv']

# Helper function to run prediction on a PIL Image
def predict_image(pil_img):
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    # Preprocess using the lightweight function
    img_arr = preprocess_pil(pil_img)
    input_batch = np.expand_dims(img_arr, axis=0).astype(np.float32)

    outputs = sess.run([output_name], {input_name: input_batch})
    logits = outputs[0]
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp / np.sum(exp, axis=1, keepdims=True)

    predicted_class_idx = int(np.argmax(probabilities, axis=1)[0])
    predicted_class_label = class_labels[predicted_class_idx]
    confidence = float(probabilities[0, predicted_class_idx])

    return {
        "label": predicted_class_label,
        "confidence": confidence,
        "probabilities": probabilities[0].tolist()
    }

# FastAPI app to serve predictions
app = FastAPI(title="Vehicle Type Predictor")

@app.get("/health")
async def health():
    """Health check endpoint for load balancers and orchestration platforms."""
    return JSONResponse(content={"status": "ok", "model": model_filename})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):


    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = predict_image(img)
    return JSONResponse(content=result)

if __name__ == "__main__":
    # Run the server (for development/testing). For production, use `uvicorn scripts.predict:app` instead.
    uvicorn.run(app, host="0.0.0.0", port=8000)
