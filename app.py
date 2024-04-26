# Saikiran Talluri 1002052828 Sai Priyanka Sanku 1002068022 Swarna Ravula 1002033344
import streamlit as st
from PIL import Image
from predict import Predict

make_pred = Predict()

def main():
    st.title("Image Captioning Streamlit App")

    uploaded_file = st.file_uploader("Choose an image to caption", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        prediction = make_pred.predict(image)
        display_results(uploaded_file, prediction)

def display_results(uploaded_file, prediction):
    st.subheader("Prediction:")
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Generated Caption: {prediction}")

if __name__ == "__main__":
    main()
