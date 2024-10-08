import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target_names[data.target]

# Split data into training and testing sets
X = df[data.feature_names]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Streamlit app
def main():
    st.title('Iris Flower Classification')
    st.write("This app predicts the Iris flower species based on its features.")

    # Display Iris flower image
    st.image('../examples/iris.jpg', caption='Iris Flower', use_column_width=True)

    # Input features on the main page
    st.header('Input Features')
    sepal_length = st.slider('Sepal Length (cm)', float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
    sepal_width = st.slider('Sepal Width (cm)', float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
    petal_length = st.slider('Petal Length (cm)', float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
    petal_width = st.slider('Petal Width (cm)', float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))

    # Model prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = clf.predict(input_data)
    prediction_proba = clf.predict_proba(input_data)

    # Display results
    st.subheader('Prediction Results')
    st.write('Predicted Species:', prediction[0])
    st.write('Prediction Probability:')
    proba_df = pd.DataFrame(prediction_proba, columns=data.target_names)
    st.dataframe(proba_df.style.format('{:.2f}'))

    # Display specific flower image based on prediction
    st.subheader('Predicted Iris Flower')
    if prediction[0] == 'setosa':
        st.image('../examples/iris_setosa.png', caption='Iris Setosa', use_column_width=True)
    elif prediction[0] == 'versicolor':
        st.image('../examples/iris_versicolor.png', caption='Iris Versicolor', use_column_width=True)
    elif prediction[0] == 'virginica':
        st.image('../examples/iris_verginica.png', caption='Iris Virginica', use_column_width=True)

if __name__ == "__main__":
    main()
