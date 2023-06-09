!pip install cpi
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from processData import processData
import numpy as np

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
    

n = int(input("How many trading days in the future would you like to predict? "))

prices, general, target, cluster, pc = processData(stocks[0], n)

# Create predictor variables
df = prices
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low
  
# Store all predictor variables in a variable X
X = df[['Open-Close', 'High-Low']]

# Target variables
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

split_percentage = 0.8
split = int(split_percentage*len(df))
  
# Train data set
X_train = X[:split]
y_train = y[:split]
  
# Test data set
X_test = X[split:]
y_test = y[split:]

# Define the hyperparameters and their ranges for grid search
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

# Create an SVM model
svm_model = SVC()

# Perform grid search on SVM model with cross-validation
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and their corresponding evaluation score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the SVM model with best hyperparameters
svm_model = SVC(**best_params)
svm_model.fit(X_train, y_train)

# Test the SVM model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
