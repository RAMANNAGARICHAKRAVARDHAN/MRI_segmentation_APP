from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import numpy as np
import io
import tensorflow as tf  # or torch if using PyTorch

app = FastAPI()

# Load the trained model (replace with your model path)
model = tf.keras.models.load_model('best_model.h5')


def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))  # Resize to match model input size
    image = np.array(image)
    image = image / 255.0  # Normalize image
    return np.expand_dims(image, axis=0)  # Add batch dimension


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file and preprocess it
    image = Image.open(io.BytesIO(await file.read()))
    processed_image = preprocess_image(image)

    # Get model prediction
    prediction = model.predict(processed_image)

    # Post-process prediction to obtain binary mask
    mask = (prediction > 0.5).astype(np.uint8)

    return {"segmentation_mask": mask.tolist()}  # Convert to list for JSON serialization


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
