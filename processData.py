from auxillary import *
from indicators import createFeatures
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, lognorm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


NORM_COLS = ["CCI", "Disparity", "KST", "MACD", "OBV", "RSI", "RVI", "Stochastic_Oscillator_ROC", "RVI_ROC", "KST_ROC", "MACD_ROC", "OBV_ROC", "RSI_ROC"]
LOG_NORM_COLS = ["ADX", "ADX_ROC", "Aroon_Up_ROC", "Aroon_Down_ROC"]

def processData(ticker, n):
    prices = yf.Ticker(ticker).history("max")

    prices = adjustPrices(prices)
    
    cont_vars, cat_vars = createFeatures(prices)
    
    target_price = calculateFuturePrice(prices, n)
    target_price_change = calculateFuturePriceChange(prices, n)
    target_price_class = calculateFuturePriceClass(prices, n)
    target_df = pd.DataFrame([target_price, target_price_change, target_price_class]).T
    target_df.columns = ["Future_Price", "Future_Price_Change", "Future_Price_Class"]

    scaler = StandardScaler()
    scaled_cont_vars = scaler.fit_transform(cont_vars)
    scaled_cont_vars = pd.DataFrame(scaled_cont_vars, columns= cont_vars.columns, index = cont_vars.index)

    for i in scaled_cont_vars.columns:
        col_name = i + "_ROC"
        scaled_cont_vars[col_name] = rocApproximation(scaled_cont_vars[i])
    
    for i in LOG_NORM_COLS:
        p = lognorm(loc = scaled_cont_vars[i].mean(), s = scaled_cont_vars[i].std()).cdf(scaled_cont_vars[i])
        col_name = i + "_Percentile"
        scaled_cont_vars[col_name] = p

    for i in NORM_COLS:
        p = norm(loc = scaled_cont_vars[i].mean(), scale = scaled_cont_vars[i].std()).cdf(scaled_cont_vars[i])
        col_name = i + "_Percentile"
        scaled_cont_vars[col_name] = p

    regular_df = pd.concat([scaled_cont_vars, cat_vars], axis = 1)

    pca = PCA(n_components=len(regular_df.columns)//2)

    pc_df = pd.DataFrame(pca.fit_transform(regular_df.dropna()), index = regular_df.dropna().index)

    clusterdf = pd.DataFrame(KMeans(n_clusters=6, n_init="auto").fit_transform(regular_df.dropna()), index = regular_df.dropna().index)

    return prices, regular_df, target_df, clusterdf, pc_df