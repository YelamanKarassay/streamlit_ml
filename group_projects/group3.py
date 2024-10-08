import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageOps
import cv2
import time
import base64
from io import BytesIO

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Prepare labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build a more robust neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# Streamlit app
def main():
    st.title("Neural Network Digit Recognition")
    st.write("This app recognizes handwritten digits using a more advanced neural network trained on the MNIST dataset.")

    # Example image download link
    def get_image_download_link(img, filename, text):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
        return href

    # Provide an example image to download
    example_image = Image.open('examples/exmple_group_3.png').convert('L')
    resized_example_image = example_image.resize((100, 100))
    st.image(resized_example_image, caption='Example Image for Testing', width=150)
    st.markdown(get_image_download_link(resized_example_image, "example_digit.png", "Download Example Digit Image"), unsafe_allow_html=True)

    # User input for uploading an image
    st.write("### Upload an image of a digit (any size, but preferably clear):")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        with st.spinner('Processing the image...'):
            time.sleep(1)  # Simulate processing time
            image = Image.open(uploaded_file).convert('L')
            image = ImageOps.invert(image)  # Invert the colors to match MNIST format (white digit on black background)
            img = np.array(image)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict the digit
            prediction = model.predict(img)
            predicted_digit = np.argmax(prediction)

            # Display the prediction
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write(f"### Predicted Digit: {predicted_digit}")
            st.write("### Prediction Probabilities:")
            st.bar_chart(prediction[0])

if __name__ == "__main__":
    main()