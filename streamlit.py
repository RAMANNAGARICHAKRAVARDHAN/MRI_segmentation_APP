import streamlit as st
import requests
from PIL import Image
import io

# Title and instructions
st.title("Brain MRI Metastasis Segmentation")
st.write("Upload an MRI image, and select a model to get the segmentation result.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type="tif")

# Model selection
model_option = st.selectbox("Select Model", ("U-Net++", "Attention U-Net"))

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Convert Streamlit model option to FastAPI parameter
    model_param = "unetpp" if model_option == "U-Net++" else "att_unet"

    if st.button("Run Segmentation"):
        # Send the image to FastAPI for prediction
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"http://localhost:8000/predict?model_name={model_param}", files=files)

        # Handle the response from FastAPI
        if response.status_code == 200:
            pred_mask = Image.open(io.BytesIO(response.content))
            st.image(pred_mask, caption="Segmentation Result", use_column_width=True)
        else:
            st.write("Error in prediction. Please try again.")
