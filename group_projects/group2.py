import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Sample Data
def load_sample_data():
    data = {
        'CustomerID': range(1, 21),
        'Annual_Income_(k$)': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        'Spending_Score': [39, 81, 6, 77, 40, 76, 94, 72, 29, 88, 15, 39, 75, 34, 78, 83, 16, 54, 71, 79],
        'Age': [19, 21, 20, 23, 31, 22, 35, 23, 64, 30, 67, 40, 37, 29, 48, 50, 27, 33, 38, 45]
    }
    return pd.DataFrame(data)

# Streamlit app
def main():
    st.title('Customer Segmentation using K-Means Clustering')
    st.write("This app clusters customer data into different segments based on spending habits.")

    # File upload or use example data
    st.write("### Upload a CSV file with customer data or use example data")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    use_example = st.checkbox("Use example data")

    if uploaded_file is not None:
        # Load data from uploaded file
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(df.head())
    elif use_example:
        # Load example data
        df = load_sample_data()
        st.write("### Example Data Preview")
        st.write(df.head())
    else:
        st.write("Please upload a CSV file or select to use example data.")
        return

    # Feature selection
    st.write("### Select Features for Clustering")
    selected_features = st.multiselect("Select features to use for clustering", df.columns.tolist(), default=df.columns.tolist())

    if selected_features:
        X = df[selected_features]

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Number of clusters selection
        st.write("### Select Number of Clusters")
        n_clusters = st.slider("Number of clusters", 2, 10, 3)

        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df['Cluster'] = cluster_labels

        # Display results
        st.write("### Clustered Data")
        st.write(df.head())

        # Visualization
        st.write("### Clusters Visualization")
        if len(selected_features) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=df['Cluster'], palette='viridis', ax=ax, s=100)
            ax.set_xlabel(selected_features[0])
            ax.set_ylabel(selected_features[1])
            ax.set_title("Customer Segments")
            st.pyplot(fig)
        else:
            st.write("Please select at least two features to visualize clusters.")

if __name__ == "__main__":
    main()