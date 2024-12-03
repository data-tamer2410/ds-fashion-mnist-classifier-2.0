# A script to convert model predictions into a convenient format.

import pandas as pd
import numpy as np

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def processing_proba(proba: list) -> tuple[pd.DataFrame, str]:
    df = pd.DataFrame({"Probability": proba[0]}, index=class_names)
    df["Probability"] = df["Probability"].apply(lambda x: f"{x:.2%}")
    predicted_thing = class_names[np.argmax(proba)]
    return df, predicted_thing
