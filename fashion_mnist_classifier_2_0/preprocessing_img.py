# This script has functionality for image
# processing depending on the type of model.

import numpy as np
from PIL import Image


def preprocessing_img(img, model_name: str) -> np.array:
    # A function that resizes the type and normalizes
    # the photo depending on the selected type of model.
    if model_name == "Standard CNN":
        img = np.array(Image.open(img).resize((28, 28)).convert("L"), dtype="float32")
        img = np.expand_dims(img, -1)
    elif model_name == "VGG16":
        img = np.array(Image.open(img).resize((64, 64)).convert("RGB"), dtype="float32")
    img = np.expand_dims(img, 0)
    return img / 255.0
