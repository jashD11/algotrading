# financial_forecast_comparison.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

def preprocess_data(ticker='^GSPC', start_date='2020-01-01'):
    """Fetch and prepare Yahoo Finance data"""
    import yfinance as yf
    data = yf.download(ticker, start=start_date)
    features = data[['Open', 'High', 'Low', 'Volume']].values
    target = data['Close'].values
    return features, target

def train_linear_regression(X_train, y_train):
    """Train and evaluate linear model"""
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def train_lstm(X_train, y_train, window_size=30):
    """Build and train LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, X_train.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    return model, history

def evaluate_models(models, X_test, y_test):
    """Compare model performance"""
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        results[name] = {
            'RMSE': np.sqrt(mse),
            'Sharpe': calculate_sharpe_ratio(preds, y_test)
        }
    return results

def calculate_sharpe_ratio(preds, actuals, risk_free_rate=0.0):
    """Calculate financial performance metric"""
    returns = np.diff(preds.flatten()) / preds.flatten()[:-1]
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

if __name__ == "__main__":
    # Data preparation
    features, target = preprocess_data()
    
    # Model training
    lr_model = train_linear_regression(features[:-30], target[:-30])
    lstm_model, history = train_lstm(features.reshape(-1, 1, features.shape[1])[:-30], target[:-30])
    
    # Evaluation
    metrics = evaluate_models({
        'Linear Regression': lr_model,
        'LSTM': lstm_model
    }, features[-30:], target[-30:])
    
    print(f"Performance Metrics:\n{metrics}")
