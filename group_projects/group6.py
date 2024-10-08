import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment >= 0.25:
        return 'Positive', sentiment
    elif sentiment <= -0.25:
        return 'Negative', sentiment
    else:
        return 'Neutral', sentiment

# Streamlit App
def main():
    st.title("Sentiment Analysis App")
    st.write("Enter text to analyze its sentiment (positive, negative, neutral).")

    user_input = st.text_area("Enter your text here:")

    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment, polarity = analyze_sentiment(user_input)
            if sentiment == 'Negative':
                st.error(f"The sentiment of the text is: {sentiment}")
            elif sentiment == 'Positive':
                st.success(f"The sentiment of the text is: {sentiment}")
            else:
                st.info(f"The sentiment of the text is: {sentiment}")

            # Add a visual representation of sentiment polarity
            st.write("\n**Sentiment Polarity:**")
            fig, ax = plt.subplots()
            ax.barh(["Sentiment"], [polarity], color="green" if polarity > 0 else "red" if polarity < 0 else "gray")
            ax.set_xlim([-1, 1])
            ax.set_xlabel("Polarity")
            st.pyplot(fig)
        else:
            st.warning("Please enter some text to analyze.")

    # Example Texts Section
    st.write("\n**Example Texts to Try:**")
    example_texts = [
        "I am so happy with the service I received today!",
        "This is the worst product I have ever bought.",
        "The weather is okay, not too bad but not great either.",
        "I love spending time with my family on weekends.",
        "The movie was quite boring and I wouldn't recommend it.",
        "I'm feeling pretty neutral about the new update."
    ]
    for i, example in enumerate(example_texts, start=1):
        st.write(f"{i}. {example}")

if __name__ == "__main__":
    main()