# Iris Flower Classification - Streamlit App

This is a Streamlit application that classifies Iris flowers into one of three species: Setosa, Versicolor, or Virginica. The app uses a machine learning model (Random Forest Classifier) trained on the popular Iris dataset. Users can input flower features such as sepal length, sepal width, petal length, and petal width, and the app will predict the species and provide the prediction probability. Additionally, it displays an image of the predicted Iris flower species.

## Features

- **Interactive User Input**: Use sliders to input flower features (sepal length, sepal width, petal length, petal width).
- **Machine Learning Prediction**: Classifies the Iris flower species using a trained Random Forest model.
- **Prediction Probability**: Displays the probabilities for each possible species.
- **Visual Representation**: Displays an image of the predicted Iris species.

## Dataset

The app uses the well-known **Iris dataset** from the scikit-learn library. This dataset contains 150 samples of Iris flowers with the following features:
- Sepal length
- Sepal width
- Petal length
- Petal width

The dataset has three classes of Iris flowers:
- Setosa
- Versicolor
- Virginica

## How to Use

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/iris-flower-classification.git
   cd iris-flower-classification
   ```

2. **Install the Dependencies**:
   Make sure to have Python installed. Install all dependencies with the following command:
   ```
   pip install -r requirements.txt
   ```

3. **Run the App**:
   Start the Streamlit server to view the application:
   ```
   streamlit run main.py
   ```

4. **Interact with the App**:
   - Use the sliders on the main page to set the flower features.
   - The app will display the predicted species and show an image of the respective flower.

## File Structure

- `main.py`: The main script that runs the Streamlit application.
- `groupX.py`: Individual Python scripts representing different group projects.
- `examples/`: Contains images of Iris flowers used by the app for visual representation.
- `requirements.txt`: Lists all the necessary Python packages to run the application.

## Requirements

- Python 3.7 or higher
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Deployment

The app can be deployed using **Streamlit Cloud**. Follow these steps:
- Create a GitHub repository and upload all the project files.
- Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your GitHub repository.
- Deploy the app by selecting the `main.py` file as the entry point.

## Example Images

The app uses the following images for visual representation based on the predicted species:
- `iris_setosa.jpg`
- `iris_versicolor.jpg`
- `iris_virginica.jpg`

These images must be located in the `examples/` folder to be used properly.

## Screenshots

![App Screenshot](examples/iris_app_screenshot.png)

## License

This project is licensed under the MIT License. Feel free to use and modify it.

## Acknowledgments

- **Scikit-Learn**: For providing the Iris dataset.
- **Streamlit**: For making it easy to create interactive data apps.
- **UCI Machine Learning Repository**: For originally hosting the Iris dataset.

## Contact

For any questions or feedback, please contact [yelamankarassay@ln.hk].

