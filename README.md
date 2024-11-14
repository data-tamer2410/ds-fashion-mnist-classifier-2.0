Fashion MNIST Classifier 2.0

This project builds on a previous effort, where a simpler fully connected classifier was created for the Fashion MNIST dataset. Here, we compare the performance of custom CNN, VGG16 feature extraction, and VGG16 fine-tuning methods for improved image classification accuracy.

## Description
This project involves creating and evaluating neural networks to classify images in the Fashion MNIST dataset, which contains grayscale images of various clothing items (e.g., shirts, trousers, shoes). The primary goal was to compare the performance of different network architectures and training techniques in image classification tasks.

Three neural network architectures were implemented:
1. **Custom Convolutional Neural Network (CNN)**: This model was built from scratch using convolutional, pooling, and dense layers. It demonstrated a high level of accuracy in detecting patterns within the images.
2. **VGG16 Feature Extraction**: Leveraging transfer learning, we used VGG16’s pre-trained convolutional layers as a feature extractor and trained custom dense layers to classify the images. This approach provided a reliable baseline and showed how pre-trained models could help in feature extraction for image classification.
3. **VGG16 Fine-Tuning**: To further enhance accuracy, we fine-tuned the top layers of VGG16 while freezing the rest, allowing the model to adapt better to the dataset's specific characteristics.

Additionally, results of a dense (fully connected) classifier were loaded for comparison, highlighting the effectiveness of convolutional architectures in image recognition tasks.
