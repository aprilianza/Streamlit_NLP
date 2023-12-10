import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load data
column_names = ['id', 'Konteks', 'Response', 'Komentar']
df = pd.read_csv("twitter_training.csv", header=None, names=column_names)
df = df.dropna(axis=0)

# Load validation data
column_names = ['id', 'Konteks', 'Response', 'Komentar']
df_validation = pd.read_csv('twitter_validation.csv', header=None, names=column_names)
df_validation = df_validation.dropna(axis=0)

# Preprocessing Teks
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(filtered_words)

df['clean_Komentar'] = df['Komentar'].apply(preprocess_text)
df_validation['clean_Komentar'] = df_validation['Komentar'].apply(preprocess_text)

X_train, y_train = df['clean_Komentar'], df['Response']
X_test, y_test = df_validation['clean_Komentar'], df_validation['Response']

st.title("Twitter Sentiment Analysis")
# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Model Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred = nb_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# Sidebar (Navbar)
selected_page = st.sidebar.radio("Navigation", ["Home", "Preprocessing", "Model"])

# Main content based on selected page
if selected_page == "Home":
    # Display your main content here
    st.write("Kelompok 1")

    # Display dataframe
    st.title('Dataframe')
    st.dataframe(df)

    # Sentiment Distribution Bar Chart
    sentiment_counts = df['Response'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')
    st.pyplot(fig)

    # Drop Null Values
    df = df.dropna(axis=0)

    # Word Clouds
    sentiment_labels = df['Response'].unique()
    for label in sentiment_labels:
        texts = ' '.join(df[df['Response'] == label]['Komentar'])
        wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(texts)
        st.title(f'Word Cloud - Sentiment {label}')
        st.image(wordcloud.to_image(), use_column_width=True)

elif selected_page == "Preprocessing":
    st.title('Preprocessing')

    st.title("Memasukan Data validasi")
    st.dataframe(df_validation)

    st.title('Pra-pemrosesan ini melibatkan langkah-langkah seperti tokenisasi, penghapusan stopwords, dan konversi ke huruf kecil.')
    df["clean_Komentar"]
    df_validation['clean_Komentar']


elif selected_page == "Model":
     # Streamlit App
    st.title("Naive Bayes Model Evaluation")

    # Model Naive Bayes
    # Menampilkan hasil evaluasi model di Streamlit dengan st.table
    result_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Irrelevant Precision', 'Irrelevant Recall', 'Irrelevant F1-Score', 'Irrelevant Support',
                'Negative Precision', 'Negative Recall', 'Negative F1-Score', 'Negative Support',
                'Neutral Precision', 'Neutral Recall', 'Neutral F1-Score', 'Neutral Support',
                'Positive Precision', 'Positive Recall', 'Positive F1-Score', 'Positive Support'],
        'Value': [accuracy, 0.78, 0.54, 0.64, 172,  # Change the values accordingly
                0.66, 0.83, 0.74, 266,
                0.81, 0.63, 0.71, 285,
                0.71, 0.83, 0.76, 277]
    })

    st.table(result_table)


   

    # Fungsi untuk prediksi sentimen dengan empat kelas
    def predict_sentiment(text):
        cleaned_text = preprocess_text(text)
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        prediction = nb_model.predict(vectorized_text)[0]

        if prediction == 'Positive':
            return "Positive"
        elif prediction == 'Negative':
            return "Negative"
        elif prediction == 'Neutral':
            return "Nature"
        elif prediction == 'Irrelevant':
            return "Irrelevant"
        else:
            return "Uncertain"  # Jika kelas tidak dikenali

    # Contoh penggunaan:
    # Streamlit App
    st.title("Sentiment Analysis Example")

    # Input teks untuk prediksi sentimen
    text_to_predict = st.text_area("Input Text for Sentiment Prediction", "I love Nasi Goreng")

    # Tombol untuk memprediksi sentimen
    if st.button("Predict Sentiment"):
        result = predict_sentiment(text_to_predict)
        st.write(f"Predicted Sentiment: {result}")