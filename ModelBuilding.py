from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_test_split_features(merged_df):
    merged_df['target'] = merged_df['Close'].shift(-1)
    merged_df.dropna(inplace=True)
    X = merged_df[['Open', 'High', 'Low', 'Close', 'Volume', 'headline_sentiment']]
    y = merged_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def fit_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE of the model: {rmse}")
    return model
