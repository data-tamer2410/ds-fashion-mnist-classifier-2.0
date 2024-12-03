# Script for loading description text and instruction text

with open(
    "fashion_mnist_classifier_2_0/data/texts/description.txt", "r", encoding="utf-8"
) as f:
    description = f.read()
with open(
    "fashion_mnist_classifier_2_0/data/texts/instruction.txt", "r", encoding="utf-8"
) as f:
    instruction = f.read()
