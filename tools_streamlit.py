import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# --------------- SETUP -------------------
st.set_page_config(page_title="AI Tweets Sentiment Dashboard", layout="wide")

# نزليها يدوي مرة وخلاص
# nltk.download('vader_lexicon')

# --------------- FUNCTIONS -------------------
@st.cache_data
def load_data():
    df = pd.read_csv('D:/VScode_projs/Streamlit/ai_tweets.csv', encoding='ISO-8859-1')
    if '_id' in df.columns:
        df.drop(columns=['_id'], inplace=True)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    return df

@st.cache_data
def prepare_sentiment(df):
    vader = SentimentIntensityAnalyzer()
    
    def analyze_vader(txt):
        if pd.isnull(txt):
            return 0
        return vader.polarity_scores(str(txt))['compound']

    df['vader_score'] = df['Tweet'].apply(analyze_vader)
    df['sentiment'] = df['vader_score'].apply(lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral'))
    return df

@st.cache_data
def get_nlp_cv_tweets(df):
    nlp_keywords = ['NLP', 'Natural Language Processing', 'Text', 'Language model']
    cv_keywords = ['Computer Vision', 'Image', 'YOLO', 'ResNet', 'Object detection']
    nlp_tweets = df[df['Tweet'].str.contains('|'.join(nlp_keywords), case=False, na=False)].copy()
    cv_tweets = df[df['Tweet'].str.contains('|'.join(cv_keywords), case=False, na=False)].copy()
    return nlp_tweets, cv_tweets

# --------------- LOAD DATA -------------------
df = load_data()
df = prepare_sentiment(df)
nlp_tweets, cv_tweets = get_nlp_cv_tweets(df)

filtered_df = df[df['created_at'] >= '2025-02-15']

daily_tweets = filtered_df.groupby(pd.Grouper(key='created_at', freq='D')).agg({
    'Tweet': 'count',
    'vader_score': 'mean'
})

weekly_tweets = filtered_df.groupby(pd.Grouper(key='created_at', freq='W-MON')).agg({
    'Tweet': 'count',
    'vader_score': 'mean'
})

# --------------- STREAMLIT APP ---------------
st.title("AI Tweets Sentiment Analysis Dashboard")

st.sidebar.title("Choose what to display:")

# Sidebar Buttons
options = st.sidebar.multiselect("Select plots to display:", [
    "Sentiment Distribution",
    "Sentiment Over Time",
    "NLP vs CV Comparison",
    "Key Events Analysis",
    "WordCloud",
    "Tweets Volume",
    "Sentiment Heatmap",
    "Boxplot NLP vs CV"
])

# --- Dynamic rendering based on options ---
if options:

    if "Sentiment Distribution" in options:
        st.header("Sentiment Distribution")
        fig, ax = plt.subplots()
        sentiment_counts = df['sentiment'].value_counts()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        st.pyplot(fig)

    if "Sentiment Over Time" in options:
        st.header("Sentiment Over Time")
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.plot(daily_tweets.index, daily_tweets['vader_score'], label='Daily Sentiment', color='#264653')
        plt.title("Daily Sentiment")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(20, 10))
        plt.plot(weekly_tweets.index, weekly_tweets['vader_score'], label='Weekly Sentiment', color='#2a9d8f')
        plt.title("Weekly Sentiment")
        st.pyplot(fig)

    if "NLP vs CV Comparison" in options:
        st.header("NLP vs Computer Vision Sentiment Comparison")
        compare_df = pd.DataFrame({
            'Category': ['NLP', 'CV'],
            'Average_Sentiment': [nlp_tweets['vader_score'].mean(), cv_tweets['vader_score'].mean()]
        })
        fig, ax = plt.subplots()
        sns.barplot(x='Category', y='Average_Sentiment', hue='Category', data=compare_df, palette='viridis', legend=False)
        st.pyplot(fig)

    if "Key Events Analysis" in options:
        st.header("Sentiment During Key Events")
        events = {
            'ChatGPT Homework surge': {'Start_Date': '2025-02-12', 'End_Date': '2025-02-18', "keyword": ['ChatGPT', 'homework']},
            'AI Crypto Hype': {'Start_Date': '2025-03-12', 'End_Date': '2025-03-16', 'keyword': ["invest", "crypto", "AI"]}
        }
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.plot(daily_tweets.index, daily_tweets['vader_score'], color='gray')
        for idx, event in events.items():
            plt.axvspan(pd.to_datetime(event['Start_Date']), pd.to_datetime(event['End_Date']), alpha=0.3, label=idx)
        plt.legend()
        st.pyplot(fig)

    if "WordCloud" in options:
        st.header("Most Frequent AI Keywords (WordCloud)")
        all_text = ' '.join(df['Tweet'].dropna())
        wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(all_text)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    if "Tweets Volume" in options:
        st.header("Tweets Volume Over Time")
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.fill_between(daily_tweets.index, daily_tweets['Tweet'], color="#90be6d", alpha=0.7)
        plt.title("Tweets Volume (Daily)")
        st.pyplot(fig)

    if "Sentiment Heatmap" in options:
        st.header("Sentiment by AI Keyword (Heatmap)")
        keywords = ['NLP', 'Computer Vision', 'ChatGPT', 'YOLO', 'Crypto', 'Language model']
        heatmap_data = []
        for keyword in keywords:
            keyword_tweets = df[df['Tweet'].str.contains(keyword, case=False, na=False)]
            avg_score = keyword_tweets['vader_score'].mean()
            heatmap_data.append(avg_score)
        heatmap_df = pd.DataFrame([heatmap_data], columns=keywords)
        fig, ax = plt.subplots()
        sns.heatmap(heatmap_df, cmap="coolwarm", annot=True)
        st.pyplot(fig)

    if "Boxplot NLP vs CV" in options:
        st.header("Boxplot of Sentiment Scores for NLP vs CV")
        nlp_tweets['Category'] = 'NLP'
        cv_tweets['Category'] = 'CV'
        combined = pd.concat([nlp_tweets, cv_tweets])
        fig, ax = plt.subplots()
        sns.boxplot(x='Category', y='vader_score', hue='Category', data=combined, palette='Set2', legend=False)
        st.pyplot(fig)

else:
    st.info("Please select one or more plots from the sidebar to display")
