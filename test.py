import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Load data
column_names = ['id', 'Konteks', 'Response', 'Komentar']
df = pd.read_csv("twitter_training.csv", header=None, names=column_names)

# Sidebar (Navbar)
selected_page = st.sidebar.radio("Navigation", ["Home", "Dataframe", "Sentiment Distribution", "Word Clouds"])

# Main content based on selected page
if selected_page == "Home":
    # Display your main content here
    st.write("Kelompok 1")
elif selected_page == "Dataframe":
    # Display dataframe
    st.title('Dataframe')
    st.dataframe(df)
elif selected_page == "Sentiment Distribution":
    # Sentiment Distribution Bar Chart
    sentiment_counts = df['Response'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')
    plt.show()
    st.pyplot(fig)
elif selected_page == "Word Clouds":
    # Drop Null Values
    df = df.dropna(axis=0)

    # Word Clouds
    positive_texts = ' '.join(df[df['Response'] == 'Positive']['Komentar'])
    wordcloud_Positive = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(positive_texts)

    negative_texts = ' '.join(df[df['Response'] == 'Negative']['Komentar'])
    wordcloud_Negative = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(negative_texts)

    neutral_texts = ' '.join(df[df['Response'] == 'Neutral']['Komentar'])
    wordcloud_Neutral = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(neutral_texts)

    irrelevant_texts = ' '.join(df[df['Response'] == 'Irrelevant']['Komentar'])
    wordcloud_Irrelevant = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(irrelevant_texts)

    # Display the Word Clouds
    st.title('Word Cloud - Sentimen Positif')
    st.image(wordcloud_Positive.to_image(), use_column_width=True)
    st.title('Word Cloud - Sentimen Negative')
    st.image(wordcloud_Negative.to_image(), use_column_width=True)
    st.title('Word Cloud - Sentimen Neutral')
    st.image(wordcloud_Neutral.to_image(), use_column_width=True)
    st.title('Word Cloud - Sentimen Irrelevant')
    st.image(wordcloud_Irrelevant.to_image(), use_column_width=True)
