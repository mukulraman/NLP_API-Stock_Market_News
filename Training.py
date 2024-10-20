import yfinance as yf
import pandas as pd
import joblib

from FeatureExtraction_Merge import(
    process_news
)

from ModelBuilding import(
    train_test_split_features,
    fit_and_evaluate
)

def training_data(news_data, end_date):

    apple_stock = yf.Ticker("AAPL")
    price_df = apple_stock.history(start=news_data['datetime'][0], end=end_date)

    price_df.reset_index(inplace=True)
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date

    news_df = pd.DataFrame(news_data)
    news_df = process_news(news_df)

    daily_sentiment = news_df.groupby('datetime')['headline_sentiment'].mean().reset_index()
    merged_df = pd.merge(price_df, daily_sentiment, left_on='Date', right_on='datetime', how='left').drop(columns=['datetime'])
    merged_df.fillna(0, inplace=True)

    X_train, X_test, y_train, y_test=train_test_split_features(merged_df)

    model=fit_and_evaluate(X_train, X_test, y_train, y_test)

    joblib.dump(model,"Model_Stock.pkl")

    return news_df,merged_df,model,price_df