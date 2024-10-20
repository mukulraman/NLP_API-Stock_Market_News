import matplotlib.pyplot as plt
import joblib
import pandas as pd
from Training import training_data

start_date = "2024-01-01"
end_date = "2024-01-31"

news_data = {
        'datetime': pd.date_range(start=start_date, periods=5),
        'headline': [
         "Apple launches new iPhone Fifteen, stocks soar",
         "Apple reports quarterly earnings, beats expectations",
         "Apple faces supply chain issues, stocks dip",
         "Apple announces new product line, stocks steady",
         "Apple under investigation for antitrust violations"
    ]
}

news_df,merged_df,model,price_df=training_data(news_data, end_date)
model=joblib.load("models\Model_Stock.pkl")

merged_df['price_return'] = merged_df['Close'].pct_change()
correlation_headline = merged_df['price_return'].corr(merged_df['headline_sentiment'])
print(f"Correlation between headline sentiment and stock price return: {correlation_headline}")

plt.figure(figsize=(10,6))
plt.plot(merged_df['Date'], merged_df['price_return'], label="Stock Price Return", marker='o')
plt.plot(merged_df['Date'], merged_df['headline_sentiment'], label="Headline Sentiment", marker='x')
plt.title('Stock Price Return vs Sentiment Analysis')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

print("\nNamed Entity Recognition on Headlines:")
for i, row in news_df.iterrows():
    print(f"Headline: {row['headline']}")
    print(f"Entities: {row['entities']}")
