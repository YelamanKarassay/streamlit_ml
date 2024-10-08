import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load pre-trained MobileNetV2 model for feature extraction
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom layers for emotion classification
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit app
def main():
    st.title("Facial Emotion Detection")
    st.write("This app detects emotions in an uploaded image using a pre-trained MobileNetV2 model.")

    # Provide example images for users without an image
    st.write("### Example Images:")
    example_images = {
        'Angry': 'examples/example1_gr5.jpeg',
        'Happy': 'examples/example2_gr5.jpg',
    }

    for label, path in example_images.items():
        st.image(path, caption=f'Example: {label}', use_column_width=True)

    # User input for uploading an image
    st.write("### Upload an image of a face:")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert('RGB')
        image = ImageOps.fit(image, (224, 224), method=Image.Resampling.LANCZOS)
        img = img_to_array(image)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Predict the emotion
        prediction = model.predict(img)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        # Display the prediction
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f"### Predicted Emotion: {predicted_emotion}")
        st.write("### Prediction Probabilities:")
        prediction_dict = {emotion_labels[i]: prediction[0][i] for i in range(len(emotion_labels))}
        st.bar_chart(prediction_dict)

if __name__ == "__main__":
    main()