import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load data from CSV
file_path = 'ASIANPAINT.csv'  # Replace with your CSV file path
stock_data = pd.read_csv(file_path)

# Step 2: Feature engineering - adding moving averages
stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()

# Drop rows with missing values caused by rolling calculations
stock_data = stock_data.dropna()

# Step 3: Prepare data for the model
# Use the 'Close' price and moving averages as features, and predict the 'Close' price
X = stock_data[['Close', 'SMA_10', 'SMA_50']]  # Features
y = stock_data['Close'].shift(-1)  # Target: next day's closing price

# Drop the last row since its target is NaN (no next day available for it)
X = X[:-1]
y = y[:-1]

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Train the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = rf_regressor.predict(X_test)

# Step 7: Evaluate the model (using mean squared error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 8: Predict the stock price for the next day using the latest data
last_data = X.iloc[-1].values.reshape(1, -1)
predicted_price = rf_regressor.predict(last_data)
print(f'Predicted stock price for the next day: {predicted_price[0]}')
