import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

data_dir = "Data"


def load_images_and_masks(data_dir):
    images = []
    masks = []

    patients = os.listdir(data_dir)
    print(f"Total patient directories: {len(patients)}")

    for patient in patients:
        patient_dir = os.path.join(data_dir, patient)

        if os.path.isdir(patient_dir):
            files = os.listdir(patient_dir)
            print(f"Processing patient: {patient}, Files: {len(files)}")

            for file_name in files:
                if file_name.endswith('.tif') and not file_name.endswith('_mask.tif'):
                    image_path = os.path.join(patient_dir, file_name)
                    mask_path = os.path.join(patient_dir, file_name.replace('.tif', '_mask.tif'))

                    if os.path.exists(image_path) and os.path.exists(mask_path):
                        print(f"Found image-mask pair: {file_name}, {file_name.replace('.tif', '_mask.tif')}")

                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        if image is None or mask is None:
                            print(f"Warning: Unable to load image or mask for {file_name}")
                            continue
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        image = clahe.apply(image)

                        image = image / 255.0
                        mask = mask / 255.0

                        image = cv2.resize(image, (256, 256))
                        mask = cv2.resize(mask, (256, 256))

                        images.append(image.reshape(256, 256, 1))
                        masks.append(mask.reshape(256, 256, 1))
                    else:
                        print(f"Missing pair: {file_name} or {file_name.replace('.tif', '_mask.tif')}")

    print(f"Total images loaded: {len(images)}")
    print(f"Total masks loaded: {len(masks)}")

    return np.array(images), np.array(masks)


images, masks = load_images_and_masks(data_dir)

if images.size == 0 or masks.size == 0:
    raise ValueError("No images or masks loaded. Please check the dataset directory and file structure.")

x_train, x_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

print(f"Training set size: {x_train.shape[0]}")
print(f"Testing set size: {x_test.shape[0]}")
