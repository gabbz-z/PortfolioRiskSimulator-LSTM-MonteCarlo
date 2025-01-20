# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 2: Load and Prepare Data
file_path = "merged_portfolio_data.csv"  
data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

# Step 2.1: Data Cleaning
print("Checking for missing or invalid values in the dataset...")
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
data.dropna(inplace=True)  # Drop rows with NaN values
print(f"Data cleaned. Remaining missing values (if any): {data.isnull().sum().sum()}")

# Step 2.2: Normalize data using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 3: Prepare data for LSTM
lookback = 60  # Number of past days to use for prediction
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i])  # Past 60 days
    y.append(scaled_data[i])  # Next day's price/return

X, y = np.array(X), np.array(y)

# Split data into training and testing sets
split = int(0.8 * len(X))  # 80% training, 20% testing
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Step 4: Build and Train LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(y_train.shape[1])  # Output layer for all assets
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Step 5: Predict Future Returns or Prices
predictions = model.predict(X_test)

# Reverse the scaling for better interpretability
predictions_unscaled = scaler.inverse_transform(predictions)
y_test_unscaled = scaler.inverse_transform(y_test)

# Evaluate the model
mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
print(f"Mean Squared Error: {mse:.5f}")

# Step 6: Monte Carlo Simulation with LSTM Predictions
num_simulations = 1000
trading_days = 252  # Approximate number of trading days in a year

# Ensure predictions_unscaled has no infinite or NaN values
print("Cleaning predictions...")
predictions_unscaled = np.nan_to_num(predictions_unscaled, nan=0.0, posinf=0.0, neginf=0.0)

simulated_portfolios = []

for _ in range(num_simulations):
    # Randomly sample daily returns from predictions
    simulated_daily_returns = np.random.choice(
        predictions_unscaled.flatten(), size=trading_days, replace=True
    )
    
    # Filter and clip invalid values in simulated daily returns
    simulated_daily_returns = np.clip(simulated_daily_returns, -100, 100)  # Clip extreme outliers
    
    # Calculate cumulative return
    cumulative_return = np.prod(1 + simulated_daily_returns / 100) - 1  # Convert percentage to ratio
    simulated_portfolios.append(cumulative_return)

# Remove invalid portfolio results (if any)
simulated_portfolios = [r for r in simulated_portfolios if np.isfinite(r)]

# Plot Monte Carlo simulation results
plt.figure(figsize=(10, 8))
plt.hist(simulated_portfolios, bins=50, alpha=0.75, edgecolor='black')
plt.title("Monte Carlo Simulation of Portfolio Returns (LSTM Predictions)")
plt.xlabel("Portfolio Annual Return")
plt.ylabel("Frequency")
plt.show()

# Step 7: Portfolio Metrics
# Calculate Expected Return
expected_return = np.mean(simulated_portfolios)
print(f"Expected Annual Return: {expected_return:.2%}")

# Calculate Annual Volatility
volatility = np.std(simulated_portfolios)
print(f"Annual Volatility: {volatility:.2%}")

# Calculate Sharpe Ratio
risk_free_rate = 0.02  # Assume 2% risk-free rate
sharpe_ratio = (expected_return - risk_free_rate) / volatility
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
