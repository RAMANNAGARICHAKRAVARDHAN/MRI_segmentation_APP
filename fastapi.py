from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf

unetpp_model = tf.keras.models.load_model('unetpp_model.h5', compile=False)
att_unet_model = tf.keras.models.load_model('att_unet_model.h5', compile=False)

app = FastAPI()

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    if len(image.shape) == 2:  
        image = np.expand_dims(image, axis=-1)
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = 'unetpp'):
    image = Image.open(file.file).convert('L')  
    processed_image = preprocess_image(image)

    if model_name == 'unetpp':
        prediction = unetpp_model.predict(processed_image)
    else:
        prediction = att_unet_model.predict(processed_image)

    pred_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
    result_image = Image.fromarray(pred_mask)

    return {"filename": file.filename, "prediction": result_image.tobytes()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
