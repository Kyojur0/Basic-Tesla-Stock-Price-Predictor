import yfinance as yf
import numpy as np
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def write_onto_file(time_stamp, data, predicted):
    with open('db.txt', 'a') as f:
        f.write(f"{time_stamp},{data['Open'][0]},{data['Close'][0]},{data['High'][0]},{data['Low'][0]},{data['Volume'][0]}, {predicted}\n")

# Define the stock ticker symbol
symbol = "TSLA"

# Create a ticker object for Tesla
tsla = yf.Ticker(symbol)

# Load the saved model
model_path = 'TeslaPredictor.h5'
model = load_model(model_path)

# Set the number of timesteps to consider
n_steps = 100

# Set up the scaler to normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))

# Continuously predict the next day's closing price
while True:
    # Get the latest stock data

    data = yf.download("TSLA", period="100m", interval="1m")
    time_stamp = data.index[0]
    timestamp_string = time_stamp.strftime('%H:%M:%S')
    close_price = data['Close'][0]
    # data Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')

    # Load the previous data from the last n_steps days
    history = tsla.history(period="1d", interval="1m", start=None, end=None)
    df1 = history['Close'].values.reshape(-1, 1)
    df1 = scaler.fit_transform(df1)

    # Take the last n_steps days of data for prediction
    temp_input = df1[-n_steps:].reshape((1, -1))

    # Normalize the latest close price
    close_price_scaled = scaler.transform([[close_price]])

    # Add the latest data to temp_input
    temp_input = np.append(temp_input, close_price_scaled)

    if len(temp_input) > n_steps:
        # Remove the oldest data
        temp_input = temp_input[1:]

    # Reshape temp_input for input to the model
    x_input = temp_input.reshape((1, n_steps, 1))

    # Make the prediction and rescale to the original range
    y_hat = model.predict(x_input)[0][0]
    y_hat = scaler.inverse_transform([[y_hat]])[0][0]

    # Print the prediction
    print(f"Predicted close price for {symbol} tomorrow: {y_hat}")

    write_onto_file(timestamp_string, data, y_hat)
    # Wait for a minute before making the next prediction

