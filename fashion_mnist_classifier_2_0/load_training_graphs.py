# Script for loading a training graphs.

from PIL import Image


def load_train_graphs(model_name: str):
    # A function to download a training schedule depending on the type of model.
    if model_name == "Standard CNN":
        image = Image.open(
            "fashion_mnist_classifier_2_0/data/training_graphs/cnn_training_graphs.png"
        )
    elif model_name == "VGG16":
        image = Image.open(
            "fashion_mnist_classifier_2_0/data/training_graphs/vgg16_training_graphs.png"
        )
    return image.resize((1000, 400))
