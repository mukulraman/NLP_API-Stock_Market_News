import pandas as pd
from textblob import TextBlob
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = TextBlob(text).sentiment.polarity
    return entities, sentiment

# Function to process news data
def process_news(news_df):
    # Apply the extraction function to each headline
    news_df['entities'] = news_df['headline'].apply(lambda x: extract_entities(x)[0])
    news_df['headline_sentiment'] = news_df['headline'].apply(lambda x: extract_entities(x)[1])
    news_df['datetime'] = pd.to_datetime(news_df['datetime']).dt.date
    return news_df
