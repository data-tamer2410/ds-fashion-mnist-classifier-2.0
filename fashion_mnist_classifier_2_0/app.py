# Main script.

import streamlit as st
import time

from fashion_mnist_classifier_2_0.load_models import load_model
from fashion_mnist_classifier_2_0.preprocessing_img import preprocessing_img
from fashion_mnist_classifier_2_0.load_training_graphs import load_train_graphs
from fashion_mnist_classifier_2_0.preprocessing_result_model import processing_proba
from fashion_mnist_classifier_2_0.load_data import description, instruction


def view_description():
    # Function for view animation description.
    for el in description.split(" "):
        yield el + " "
        time.sleep(0.07)


def main():
    # Sidebar
    st.sidebar.title(":gray[Settings]")
    st.sidebar.divider()

    model_name = st.sidebar.radio("**Select a model type**", ["Standard CNN", "VGG16"])
    st.sidebar.divider()

    orig_img = st.sidebar.file_uploader(
        "**Load image for predict**", type=["jpg", "jpeg", "png"]
    )
    if orig_img is not None:
        st.success("The image has been successfully uploaded")
        st.sidebar.image(orig_img, use_container_width=True)

    # Main
    st.title(":gray[Fashion MNIST Classifier 2.0]")
    st.divider()

    place_for_description = st.empty()
    if "not_start" in st.session_state:
        place_for_description.markdown(description)

    with st.expander("Instruction"):
        st.markdown(instruction)

    st.header("Model training schedules")
    st.image(load_train_graphs(model_name))

    _, col2, _ = st.columns([3, 1, 3])
    if col2.button("Predict"):
        if orig_img is None:
            st.error("First you need to upload a image for prediction")
        else:
            model = load_model(model_name)
            img = preprocessing_img(orig_img, model_name)
            proba = model.predict(img, verbose=0)
            df_proba, prediction = processing_proba(proba)
            col1, col2 = st.columns([2, 1])
            col1.table(df_proba)
            col2.markdown(f"### Prediction thing: :green[**{prediction}**]")

    if "not_start" not in st.session_state:
        st.session_state["not_start"] = True
        place_for_description.write_stream(view_description)


if __name__ == "__main__":
    main()
