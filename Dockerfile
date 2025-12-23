# Use a Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock and install runtime dependencies via pipenv
COPY Pipfile Pipfile.lock ./

# Install minimal system libs and Python deps, then use pipenv to install locked deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxrender1 libxext6 build-essential \
    && pip install --upgrade pip pipenv \
    && pipenv install --deploy --system --ignore-pipfile \
    && rm -rf /var/lib/apt/lists/*

# Copy only the server script and the ONNX model (keep the image small)
COPY scripts/predict.py /app/scripts/predict.py
COPY model_file/vehicle_identifier_mobilenet_v2.onnx /app/model_file/vehicle_identifier_mobilenet_v2.onnx
# Copy external data file required by the ONNX model (if present)
COPY model_file/vehicle_identifier_mobilenet_v2.onnx.data /app/model_file/vehicle_identifier_mobilenet_v2.onnx.data

# Create a non-root user and switch to it
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# Expose the default port
EXPOSE 8000


# Default command to run the server
# Use Uvicorn to run the FastAPI app defined at scripts.predict:app
CMD ["uvicorn", "scripts.predict:app", "--host", "0.0.0.0", "--port", "8000"]
