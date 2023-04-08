from auxillary import *
from indicators import createFeatures
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, lognorm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


NORM_COLS = ["CCI", "Disparity", "KST", "MACD", "OBV", "RSI", "RVI", "Stochastic_Oscillator_ROC", "RVI_ROC", "KST_ROC", "MACD_ROC", "OBV_ROC", "RSI_ROC"]
LOG_NORM_COLS = ["ADX", "ADX_ROC", "Aroon_Up_ROC", "Aroon_Down_ROC"]

def processData(tickers, n):
    n = 0
    for i in tickers:
        ticker = yf.Ticker(i).history("max")

        df = adjustPrices(ticker)
        
        cont_vars, cat_vars = createFeatures(df)
        cont_vars = cont_vars.drop(["Stock Splits", "Dividends"], axis=1)
        
        target_price = calculateFuturePrice(ticker, n)
        target_price_change = calculateFuturePriceChange(ticker, n)
        target_price_class = calculateFuturePriceClass(ticker, n)
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

        pc_df["Ticker"] = pd.Series([len(pc_df)*[i]])
        clusterdf["Ticker"] = pd.Series([len(clusterdf)*[i]])
        regular_df["Ticker"] = pd.Series([len(regular_df)*[i]])
        target_df["Ticker"] = pd.Series([len(target_df)*[i]])

        if n == 0:
            master_regular = regular_df
            master_target = target_df
            master_cluster = clusterdf
            master_pc = pc_df
        else:
            master_regular = pd.concat([master_regular,regular_df], axis = 0)
            master_target = pd.concat([master_target, target_df], axis = 0)
            master_cluster = pd.concat([master_cluster, clusterdf], axis = 0)
            master_pc = pd.concat([master_pc, pc_df], axis = 0)

        n += 1

    return master_regular, master_target, master_cluster, master_pc