# Functionality for load models.

from tensorflow.keras import models


def load_model(model_name: str):
    # Function for load model.
    if model_name == "Standard CNN":
        file_name = "fashion_mnist_classifier_2_0/data/models/cnn_fashion.h5"
    elif model_name == "VGG16":
        file_name = (
            "fashion_mnist_classifier_2_0/data/models/vgg16_fashion_fine_tune.h5"
        )
    model = models.load_model(file_name)
    return model
