# PortfolioRiskSimulator-LSTM-MonteCarlo

This project demonstrates a Portfolio Risk Simulation using machine learning and statistical techniques. It predicts future asset returns with an LSTM model and evaluates portfolio performance through Monte Carlo simulations.The system predicts future asset returns using historical data and simulates thousands of portfolio outcomes to assess risk exposure and optimize strategies for risk-adjusted returns.


## 🧠 Methodology
1. Data Preprocessing:
    * Cleaned and normalized historical asset price data.
    * Used a rolling lookback window of 60 days to create input features for predictions.
2. LSTM Model for Predictions:
    * Built a deep learning model to predict the next day’s return based on historical patterns.
    * Trained on a dataset of multiple assets to capture trends and volatilities.
3. Monte Carlo Simulation:
    * Generated 1000 random portfolio return scenarios using the LSTM predictions.
    * Simulated returns over 252 trading days (1 year).
4. Portfolio Metrics and Visualization:
    * Calculated metrics such as Expected Return, Volatility, and Sharpe Ratio.
    * Visualized the simulated portfolio performance with histograms and cumulative return graphs.

## 📈 Results
Using an example dataset of assets like Apple (AAPL), Tesla (TSLA), and others, the program provided the following:
1. Monte Carlo Simulation Histogram
* Shows the distribution of annual portfolio returns under simulated scenarios.
2. Portfolio Metrics
* Expected Annual Return: +25.12%
* Annual Volatility: 14.45%
* Sharpe Ratio: 1.60
3. Cumulative Portfolio Returns
Example visualization of cumulative portfolio performance over a year

![Figure_1mc](https://github.com/user-attachments/assets/cd713e71-6817-4149-b096-dded45b80250)



## 🛠️ How It Works
1. Data Cleaning
Historical asset prices are cleaned, missing or invalid values are handled, and data is scaled using MinMaxScaler for consistent input to the LSTM model.
2. Prediction with LSTM
A rolling window of 60 days is used to predict the next day’s returns. The LSTM model learns from historical trends to make predictions.
3. Monte Carlo Simulation
The LSTM predictions are used to generate random daily returns for 1000 simulated portfolio outcomes, analyzing annual performance under various conditions.


📌 Example Dataset
The example dataset includes historical prices of assets like:
* AAPL (Apple)
* TSLA (Tesla)
* SPY (S&P 500 ETF)
* GLD (Gold ETF)



