import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

st.title("Distribusi Sentimen!")
column_names = ['id', 'Konteks', 'Response', 'Komentar']

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None, names=column_names)

    # Sentiment Distribution Bar Chart
    sentiment_counts = df['Response'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')

    # Display dataframe
    st.title('Dataframe')
    st.dataframe(df)

    # Display barchart
    st.title('Distribusi Sentimen')
    st.pyplot(fig)
    
    # Drop Sentimen
    st.title('Drop Semua Data Null')

    # Word Cloud for Positive Sentiments
    df = df.dropna(axis=0)
    positive_texts = ' '.join(df[df['Response'] == 'Positive']['Komentar'])
    wordcloud_Positive = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(positive_texts)

    # Word Cloud for Negative Sentiments
    negative_texts = ' '.join(df[df['Response'] == 'Negative']['Komentar'])
    wordcloud_Negative = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(negative_texts)

    # Word Cloud for Neutral Sentiments
    neutral_texts = ' '.join(df[df['Response'] == 'Neutral']['Komentar'])
    wordcloud_Neutral = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(neutral_texts)

    # Word Cloud for Irrelevant Sentiments
    irrelevant_texts = ' '.join(df[df['Response'] == 'Irrelevant']['Komentar'])
    wordcloud_Irrelevant = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(irrelevant_texts)
    
    # Display the Word Cloud using Streamlit
    st.title('Word Cloud - Sentimen Positif')
    st.image(wordcloud_Positive.to_image(), use_column_width=True)
    st.title('Word Cloud - Sentimen Negative')
    st.image(wordcloud_Negative.to_image(), use_column_width=True)
    st.title('Word Cloud - Sentimen Neutral')
    st.image(wordcloud_Neutral.to_image(), use_column_width=True)
    st.title('Word Cloud - Sentimen Irrelevant')
    st.image(wordcloud_Irrelevant.to_image(), use_column_width=True)
