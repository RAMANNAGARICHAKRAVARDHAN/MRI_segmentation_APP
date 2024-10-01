from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import numpy as np
import io
import tensorflow as tf  

app = FastAPI()

model = tf.keras.models.load_model('best_model.h5')


def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))  
    image = np.array(image)
    image = image / 255.0  
    return np.expand_dims(image, axis=0)  


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    mask = (prediction > 0.5).astype(np.uint8)

    return {"segmentation_mask": mask.tolist()}  


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
