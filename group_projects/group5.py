import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance

# Streamlit app
def main():
    st.title("Image to Colorful Watercolor Art")
    st.write("This app converts your image into a colorful watercolor-style artwork.")

    # User input for uploading an image
    st.write("### Upload an image:")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file).convert('RGB')

        # Enhance colors
        enhancer = ImageEnhance.Color(image)
        image_enhanced = enhancer.enhance(2.0)  # Increase color saturation

        # Apply watercolor effect using edge enhancement and blurring
        watercolor = image_enhanced.filter(ImageFilter.EDGE_ENHANCE_MORE)
        watercolor = watercolor.filter(ImageFilter.SMOOTH_MORE)
        watercolor = watercolor.filter(ImageFilter.SMOOTH_MORE)

        # Display the original and watercolor images
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.image(watercolor, caption='Colorful Watercolor Art', use_column_width=True)

if __name__ == "__main__":
    main()