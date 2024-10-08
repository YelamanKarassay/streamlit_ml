import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Example dataset
def load_example_data():
    data = {
        'Size_in_sqft': [1500, 2000, 1800, 2200, 1200, 2500, 1700, 2100, 1300, 1900, 1600, 2300, 1400, 2400, 1750, 1950, 1550, 2250, 1350, 1850],
        'Number_of_rooms': [3, 4, 3, 4, 2, 5, 3, 4, 2, 3, 3, 4, 2, 5, 3, 4, 3, 4, 2, 3],
        'Age_of_house': [10, 5, 15, 7, 20, 3, 12, 8, 18, 6, 10, 4, 25, 2, 14, 9, 11, 3, 22, 7],
        'Distance_to_city_center': [5, 10, 8, 3, 12, 7, 9, 4, 11, 6, 10, 2, 15, 5, 7, 6, 8, 4, 13, 9],
        'House_Price': [300000, 400000, 350000, 450000, 200000, 500000, 320000, 420000, 220000, 380000, 310000, 470000, 180000, 480000, 330000, 390000, 305000, 460000, 190000, 345000]
    }
    return pd.DataFrame(data)

# Streamlit app
def main():
    st.title("House Price Prediction")
    st.write("This app predicts the price of a house based on features such as size, number of rooms, and location.")

    # File upload or use example data
    uploaded_file = st.file_uploader("Upload a CSV file with house data", type="csv")
    use_example_data = st.checkbox("Use example data")

    if uploaded_file is not None:
        # Load data from uploaded file
        df = pd.read_csv(uploaded_file)
    elif use_example_data:
        # Load example data
        df = load_example_data()
        st.write("### Example Data Preview")
        st.write(df.head())
    else:
        st.write("Please upload a CSV file or select to use example data.")
        return

    # Data visualization
    st.write("### Data Visualization")
    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("Feature Distributions")
    for column in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax, color='blue')
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)

    # Feature selection
    st.write("### Select Features for Prediction")
    selected_features = st.multiselect("Select features to use for prediction", df.columns.tolist(), default=df.columns.tolist()[:-1])
    target = st.selectbox("Select the target variable", df.columns.tolist(), index=len(df.columns.tolist()) - 1)

    if selected_features and target:
        X = df[selected_features]
        y = df[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Model evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Evaluation")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")

        # Visualization of predictions
        st.write("### Predictions vs Actual Values")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Predicted vs Actual House Prices")
        st.pyplot(fig)

        # User input for new prediction
        st.write("### Predict House Price")
        input_data = []
        for feature in selected_features:
            value = st.number_input(f"Enter value for {feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
            input_data.append(value)

        if st.button("Predict Price"):
            input_data = np.array(input_data).reshape(1, -1)
            predicted_price = model.predict(input_data)[0]
            st.write(f"### Predicted House Price: ${predicted_price:,.2f}")

if __name__ == "__main__":
    main()