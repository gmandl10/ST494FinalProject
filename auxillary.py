import pandas as pd
import numpy as np
from scipy.stats import norm
import cpi
from dateutil.relativedelta import relativedelta

def roc(price, n):
    """
    Calculates the Rate of Change (ROC) of a price over n periods
    """
    return (price.diff(n)/price.shift(n))*100

def calculateFuturePrice(df, n):
    """
    Calculates the equity price n days in the future
    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        n (int) - number of days in future
    Outputs: 
        fp (Series) - the future prices of the equity 
    """
    fp = df["Close"].shift(-n)
    return fp

def calculateFuturePriceChange(df, n):
    """
    Calculates the equity price change n days in the future
    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        n (int) - number of days in future
    Outputs: 
        fp (Series) - the future price changes of the equity 
    """
    fpc = (df["Close"].shift(-n) - df["Close"])/df["Close"]
    return fpc

def calculateFuturePriceClass(df,n):
    """
    Calculates if price rises n days in the future
    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        n (int) - number of days in future
    Outputs: 
        fp (Series) - the future price behaviour of the equity 
    """
    fp_class = pd.cut(df["Close"].diff(-n), bins = [-100, -1, 100], labels = [0,1])
    return fp_class


def percentile(column):
    mu = column.mean()
    sigma = column.std()
    percentile_col = norm(loc = mu, scale = sigma).cdf(column)
    return percentile_col

def rocApproximation(series, n = 10):
     roc = (1/2)*series.diff(1)/series.shift(1)
     for i in range(2,n+1):
          roc = roc + (1/(i+1))*series.diff(i)/series.shift(i)
     return roc

def adjustPrices(df):
    """
    Adjust the value of historical prices to present day dollars to account for inflation
    """
    data = df.copy()
    adj_close = []
    adj_open = []
    adj_high = []
    adj_low = []

    for i in df.index:
        if cpi.LATEST_MONTH + relativedelta(months = 1) <= i.date():
            adj_close.append(df.loc[i, "Close"])
            adj_open.append(df.loc[i, "Open"])
            adj_high.append(df.loc[i, "High"])
            adj_low.append(df.loc[i, "Low"])
        else:
            adj_close.append(cpi.inflate(df.loc[i, "Close"], i))
            adj_open.append(cpi.inflate(df.loc[i, "Open"], i))
            adj_high.append(cpi.inflate(df.loc[i, "High"], i))
            adj_low.append(cpi.inflate(df.loc[i, "Low"], i))
    
    data["Close"] = adj_close
    data["Low"] = adj_low
    data["High"] = adj_high
    data["Open"] = adj_open

    return data