from processData import processData
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras import optimizers

def buildRNN(num_layers, num_neurons, input_shape, lr):
    model = Sequential()
    model.add(LSTM(units=num_neurons,return_sequences=True,input_shape= input_shape))
    model.add(Dropout(0.2))
    
    i = 2
    while i <= num_layers:
            
        if i == num_layers:
            model.add(LSTM(units=50))
        else:
            model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.2))
        i+=1

    model.add(Dense(units=1))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),loss='mean_squared_error')
    return model



stocks = []
print("Enter the ticker symbols of interest (enter spacebar to exit)")
s = "  "
while s != " ":
    s = input("Ticker: ")
    if s == " " and len(stocks) == 0:
        print("Please enter a ticker symbol")
        s = "  "
        continue
    elif s == " ":
        break
    if s.upper() not in stocks:
        stocks.append(s.upper())

predicted_prices = []

for x in stocks:
    prices, general, target, cluster, pc = processData(x, 10)
    variables = ["Close", "PC0", "PC1"]
    training_set = pd.concat([prices, general, pc], axis = 1).loc[:,variables].dropna().values
    train_y = [row[0] for row in training_set]
    train_y = np.array(train_y).reshape(-1,1)

    x_sc = MinMaxScaler(feature_range=(0,1))
    y_sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = x_sc.fit_transform(training_set)
    train_y = y_sc.fit_transform(train_y)
    X_train = []
    y_train = []
    for i in range(60, int(len(training_set_scaled)*0.75)):
        X_train.append(training_set_scaled[i-60:i, :])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(variables)))

    model = buildRNN(4,100, (X_train.shape[1], len(variables)), 0.005)

    model.fit(X_train,y_train,epochs=25,batch_size=32)

    X_test = []
    y_test = []
    for i in range(int(len(training_set_scaled)*0.75), len(training_set_scaled)):
        X_test.append(training_set_scaled[i-60:i, :])
        y_test.append(training_set_scaled[i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(variables)))

    real_stock_price = prices.iloc[-len(X_test):, 3:4].values

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = y_sc.inverse_transform(predicted_stock_price)

    today = pd.concat([prices["Close"], pc.loc[:, ["PC0", "PC1"]]], axis = 1).iloc[-60:].values
    today_scaled = x_sc.transform(today)

    predicted_price = model.predict(np.array([today_scaled]))
    predicted_price = y_sc.inverse_transform(predicted_price)

    predicted_prices.append(predicted_price[0])
    
    for i in range(len(stocks)):
    text = "Predicted price for " + stocks[i] + ": $" + str(predicted_prices[i][0])
    print(text)