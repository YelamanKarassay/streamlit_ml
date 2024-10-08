import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Example CSV data
EXAMPLE_CSV = """Transaction,Item
1,Bread
1,Milk
1,Eggs
1,Butter
2,Bread
2,Diapers
2,Beer
2,Chips
3,Milk
3,Diapers
3,Beer
3,Cola
4,Bread
4,Milk
4,Diapers
4,Beer
4,Chips
5,Bread
5,Milk
5,Eggs
5,Butter
6,Diapers
6,Beer
6,Cola
7,Bread
7,Milk
7,Eggs
8,Bread
8,Milk
8,Diapers
8,Butter
9,Bread
9,Milk
9,Eggs
9,Beer
9,Chips
10,Milk
10,Diapers
10,Beer
10,Cola"""

# Streamlit App
def main():
    st.title("Market Basket Analysis App")
    st.write("Upload a dataset and analyze product relationships using association rule mining.")

    # Checkbox for using example data
    use_example_data = st.checkbox("Use example dataset")

    # File Upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
    elif use_example_data:
        st.write("Using example dataset.")
        from io import StringIO
        df = pd.read_csv(StringIO(EXAMPLE_CSV))
    else:
        st.warning("Please upload a CSV file or select the example dataset.")
        return

    # Display Dataset
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # Transform the dataset into the appropriate format
    st.write("### Data Transformation:")
    transaction_df = df.groupby(['Transaction', 'Item'])['Item'].count().unstack().fillna(0)
    transaction_df = transaction_df.applymap(lambda x: 1 if x > 0 else 0)
    st.write(transaction_df.head())

    # Apply Apriori Algorithm
    min_support = st.slider("Select Minimum Support: ", min_value=0.01, max_value=1.0, value=0.3, step=0.01)
    frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)

    st.write("### Frequent Itemsets:")
    st.dataframe(frequent_itemsets)

    # Visualize Frequent Itemsets
    st.write("### Frequent Itemsets Bar Chart:")
    if not frequent_itemsets.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=frequent_itemsets['support'], y=frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x))))
        plt.xlabel('Support')
        plt.ylabel('Itemsets')
        plt.title('Frequent Itemsets')
        st.pyplot(plt)

    # Generate Association Rules
    metric = st.selectbox("Select Metric for Association Rules: ", ["lift", "confidence", "support"])
    min_threshold = st.slider("Select Minimum Threshold for the Metric: ", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    st.write("### Association Rules:")
    st.dataframe(rules)

    # Visualize Association Rules
    st.write("### Association Rules Scatter Plot:")
    if not rules.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=rules[metric], y=rules['lift'], hue=rules['confidence'], size=rules['support'], sizes=(20, 200), alpha=0.6)
        plt.xlabel(metric.capitalize())
        plt.ylabel('Lift')
        plt.title('Association Rules Visualization')
        st.pyplot(plt)

if __name__ == "__main__":
    main()