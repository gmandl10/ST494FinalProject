from auxillary import roc


def calculateADX (df, period = 14):
    """
    Calculates the Average Directional Index (ADX) values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
    Outputs:
        adx (Series) - a column containing adx for the corresponding date
        pdi (Series) - a column containing postive directional index for the corresponding entry
        ndi (Series) - a column containing negative directional index for the corresponding entry
    """
    plus_dm = df["High"].diff()
    minus_dm = df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    t1 = pd.DataFrame(df["High"] - df["Low"])
    t2 = pd.DataFrame(abs(df["High"] - df["Close"].shift(1)))
    t3 = pd.DataFrame(abs(df["Low"] - df["Close"].shift(1)))

    tr = pd.concat([t1, t2, t3], axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(period).mean()

    pdi = 100 * (plus_dm.ewm(alpha = 1/period).mean() / atr)
    ndi = abs(100 * (minus_dm.ewm(alpha = 1/period).mean() / atr))
    dx = (abs(pdi - ndi) / abs(pdi + ndi)) * 100
    adx1 = ((dx.shift(1) * (period - 1)) + dx) / period
    adx = adx1.ewm(alpha = 1/period).mean()

    adx_indicator = []
    adx_indicator.append("Neutral")
    i=1

    while i < len(adx):
        adx1 = adx[i-1]
        adx2 = adx[i]

        if adx1 < 25 and adx2 > 25 and pdi[i] > ndi[i]:
            adx_indicator.append("Buy")
        elif adx1 < 25 and adx2 > 25 and ndi[i] > pdi[i]:
            adx_indicator.append("Sell")
        else:
            adx_indicator.append("Neutral")
        i+=1
        
    return adx, adx_indicator

def calculateAroon(df, period = 25):
    """
    Calculates the Aroon Indicator values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 25
        
    Outputs:
        aroonup (Series) - a column containing Aroon Up values for the corresponding entry
        aroondown (Series) - a column containing Aroon Down values for the corresponding entry
    """   
    high = df["High"]
    low = df["Low"]

    aroonup = 100 * high.rolling(period + 1).apply(lambda x: x.argmax()) / period
    aroondown = 100 * low.rolling(period + 1).apply(lambda x: x.argmin()) / period

    aroon_crossover = []
    aroon_crossover.append("Neutral")

    i= 1
    while i < len(aroonup):
        aroonup1 = aroonup[i-1]
        aroonup2 = aroonup[i]
        aroondown1 = aroondown[i-1]
        aroondown2 = aroondown[i]
        if aroonup1 < aroondown1 and aroonup2 > aroondown2:
            aroon_crossover.append("Long")
        elif aroonup1 > aroondown1 and aroonup2 < aroondown2:
            aroon_crossover.append("Short")
        else:
            aroon_crossover.append("Neutral")
        
        i += 1

    aroon_indicator = []

    for i in range(len(aroonup)):
        up = aroonup[i]
        down = aroondown[i]

        if up > 70 and down < 30:
            aroon_indicator.append("Long")
        elif down > 70 and up < 30: 
            aroon_indicator.append("Short")
        elif down < 50 and up < 50:
            aroon_indicator.append("PriceConsolidating")
        else:
            aroon_indicator.append("Neutral")
    
    return aroonup, aroondown, aroon_crossover, aroon_indicator

def calculateCCI(df, period = 20):
    """
    Calculates the Commodity Channel Index values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 20
        
    Outputs:
        cci (Series) - a column containing cci values for the corresponding entry

    """   
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    typical = (high + low + close)/3

    MA = typical.rolling(period).mean()

    mean_deviation = abs(typical - MA)

    cci = (typical -MA)/0.015*mean_deviation

    CCI_indicator = []
    oversold = -150
    overbought = 150

    CCI_indicator.append("Neutral")

    i = 1
    while i < len(cci):
        CCI1 = cci[i-1]
        CCI2 = cci[i]
        if CCI1 > oversold and CCI2 < oversold:
            CCI_indicator.append("Buy")
        elif CCI1 < overbought and CCI2 > overbought:
            CCI_indicator.append("Sell")
        else:
            CCI_indicator.append("Neutral")
        i+=1

    return cci, CCI_indicator


def calculateDisparity(df, period = 14):
    """
    Calculates the Disparity Index values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
        
    Outputs:
        disparity (Series) - a column containing disparity values for the corresponding entry
    """
    close = df["Close"]

    SMA = close.rolling(period).mean()
    disparity = (close - SMA)/(SMA*100)

    disparity_indicator = []

    for _ in range(5):
        disparity_indicator.append("Neutral")

    i = 5
    while i < len(disparity):
        di1 = disparity[i-5]
        di2 = disparity[i-4]
        di3 = disparity[i-3]
        di4 = disparity[i-2]
        di5 = disparity[i-1]
        ditoday = disparity[i]

        if di1 < 0 and di2 < 0 and di3 < 0 and di4 < 0 and di5 < 0 and ditoday > 0:
            disparity_indicator.append("Buy")
        elif di1 > 0 and di2 > 0 and di3 > 0 and di4 > 0 and di5 > 0 and ditoday < 0:
            disparity_indicator.append("Sell")
        else:
            disparity_indicator.append("Neutral")
        
        i+=1

    return disparity, disparity_indicator



def calculateKST(df, signal = 9):
    """
    Calculates the Know Sure Thing (KST) values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        signal (int) - period to calculate the signal of indidcator on, standard value is 9
        
    Outputs:
        kst (Series) - a column containing KST for the corresponding entry
        kstsignal (Series) - a column containing KST signal for the corresponding entry
    """

    close = df["Close"]

    r1 = roc(close, 10).rolling(10).mean()
    r2 = roc(close, 15).rolling(10).mean()
    r3 = roc(close, 20).rolling(10).mean()
    r4 = roc(close, 30).rolling(15).mean()
    kst = r1 + 2*r2 + 3*r3 + 4*r4

    kstsignal = kst.rolling(signal).mean()

    kst_crossover = []
    kst_crossover.append("Neutral")

    i = 1
    while i < len(kst):
        kst1 = kst[i-1]
        kst2 = kst[i]
        signal1 = kstsignal[i-1]
        signal2 = kstsignal[i]

        if kst1 < signal1 and kst2 > signal2:
            kst_crossover.append("Buy")
        elif kst1 > signal1 and kst2 < signal2:
            kst_crossover.append("Sell")
        else:
            kst_crossover.append("Neutral")
        i+=1

    return kst, kst_crossover

def calculateMACD (df, long = 26, short = 12, lSignal = 9):
    """
    Calculates the moving average convergence divergence values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        long (int) - the length of longer EMA (general metric = 26)
        short (int) - the length of the shofter EMA (general metric = 12)
        signal (int) - the timeframe to compute the signal for the MACD (general metric = 9)
    Outputs:
        macd (Series) - a column containing macd for the corresponding entry
        signal (Series) - a column containing the signal for the corresponding entry
    """
    shortma = df["Close"].ewm(span = short, adjust = False).mean()
    longma = df["Close"].ewm(span = long, adjust = False).mean()

    macd = shortma - longma
    signal = macd.ewm(span = lSignal, adjust = False).mean()

    MACDcrossover = []

    for i in range(len(macd)):
        macd1 = macd[i-1]
        signal1 = signal[i-1]
        macd2 = macd[i]
        signal2 = signal[i]
        if macd1 < signal1 and macd2 > signal2:
            MACDcrossover.append("Buy")
        elif macd1 > signal1 and macd2 < signal2:
            MACDcrossover.append("Sell")
        else:
            MACDcrossover.append("Neutral")

    return macd, MACDcrossover

def calculateOBV(df):
    volume = df["Volume"]
    close = df["Close"]
    obv = []
    obv.append(volume[0])
    
    for i in range(0,len(volume)-1):
        if close[i+1] > close[i]:
            obv.append(obv[i] + volume[i+1])
        elif close[i+1] == close[i]:
            obv.append(obv[i])
        else:
            obv.append(obv[i] - volume[i+1])

    obv = pd.Series(obv, index = volume.index)
    return obv

def calculateRSI(df, period = 14):
    """
    Calculates the Relative Strength Index (RSI) values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
    Outputs:
        rsi (Series) - a column containing rsi for the corresponding entry
    """
    close = df["Close"]  
    diff = close.diff()
    gain = []
    loss = []
    for i in range(len(diff)):
        if diff[i] < 0:
            gain.append(0)
            loss.append(diff[i])
        else:
            gain.append(diff[i])
            loss.append(0)
    gain = pd.Series(gain)
    loss = pd.Series(loss)

    gainEMA = gain.ewm(span = period - 1, adjust = False).mean()
    lossEMA = abs(loss.ewm(span = period - 1, adjust = False).mean())
    rs = gainEMA/lossEMA
    rsi = 100 - (100 / (1 + rs))
    rsi_ = []
    for i in rsi:
        rsi_.append(i)
    rsisignal = []
    for i in range(len(rsi)):
        if rsi[i] < 30:
            rsisignal.append("Buy")
        elif rsi[i] > 70:
            rsisignal.append("Sell")
        else:
            rsisignal.append("Neutral")
    return rsi_, rsisignal

def calculateRVI(df, period = 10):
    """
    Calculates the Relative Vigor Index (RVI) values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 10
        
    Outputs:
        rvi (Series) - a column containing RVI for the corresponding entry
        rvisignal (Series) - a column containing RVI signal for the corresponding entry
    """

    high = df["High"]
    low = df["Low"]
    _open = df["Open"]
    close = df["Close"]

    a = close - _open
    b = close.shift(1) - _open.shift(1)
    c = close.shift(2) - _open.shift(2)
    d = close.shift(3) - _open.shift(3)

    numerator = 1/6*(a+2*b+2*c+d)

    e = high - low
    f = high.shift(1) - low.shift(1)
    g = high.shift(2) - low.shift(2)
    h = high.shift(3) - low.shift(3)

    denominator = 1/6*(e + 2*f + 2*g +h)

    rvi = numerator.rolling(10).mean()/denominator.rolling(10).mean()

    i = rvi.shift(1)
    j = rvi.shift(2)
    k = rvi.shift(3)

    rvisignal = 1/6*(rvi+2*i+2*j+k)

    rvi_crossover = []
    rvi_crossover.append("Neutral")

    i = 1 
    while i < len(rvi):
        rvi1 = rvi[i-1]
        rvi2 = rvi[i]
        signal1 = rvisignal[i-1]
        signal2 = rvisignal[i]
        if rvi1 < signal1 and rvi2 > signal2:
            rvi_crossover.append("Buy")
        elif rvi1 > signal1 and rvi2 < signal2:
            rvi_crossover.append("Sell")
        else:
            rvi_crossover.append("Neutral")
        i+=1
    rvi_divergence = []
    rvi_divergence.append("Neutral")

    i = 1 
    while i < len(rvi):
        rvi1 = rvi[i-1]
        rvi2 = rvi[i]
        price1 = close[i-1]
        price2 = close[i]
        if price2 > price1 and rvi2 < rvi1:
            rvi_divergence.append("Sell")
        elif price1 > price2 and rvi1 < rvi2:
            rvi_divergence.append("Buy")
        else:
            rvi_divergence.append("Neutral")
        i+=1
    return rvi, rvi_crossover, rvi_divergence

def calculateStochasticOscillator (df, period = 14, signal = 3):
    """
    Calculates the Stochastic Oscillator values for the dataframe

    Inputs:
        df (DataFrame) - dataframe that contains the data of the equity of interest
        period (int) - period to calculate the indidcator on, standard value is 14
        signal (int) - the timeframe to compute the signal for the Stochastic Oscillator,
         standard value is 3
        
    Outputs:
        so (Series) - a column containing Stochastic Oscillator for the corresponding entry
        sosignal (Series) - a column containing Stochastic Oscillator signal for the 
            corresponding entry
    """
    highest_high = df["High"].rolling(period).max()
    lowest_low = df["Low"].rolling(period).min()
    
    so = 100*((df["Close"] - lowest_low)/ (highest_high - lowest_low))
    sosignal = so.rolling(signal).mean()

    stochastic_indicator = []
    for i in range(len(so)):
        s = so[i]
        ma = sosignal[i]
        if s < 20 and ma < 20 and s < ma:
            stochastic_indicator.append("Buy")
        elif s > 80 and ma > 80 and s > ma:
            stochastic_indicator.append("Sell")
        else:
            stochastic_indicator.append("Neutral")

    return so, stochastic_indicator

def createFeatures(data):
    df = data.copy()

    adx, adx_indicator = calculateADX(df)
    df["ADX"] = adx
    df["ADX_Indicator"] = adx_indicator

    aroonup, aroondown, arooncrossover, aroonindicator = calculateAroon(df)
    df["Aroon_Up"] = aroonup
    df["Aroon_Down"] = aroondown
    df["Aroon_Crossover"] =arooncrossover
    df["Aroon_Indicator"] = aroonindicator

    cci, cci_indicator = calculateCCI(df)
    df["CCI"] = cci
    df["CCI_Indicator"] = cci_indicator

    disparity, disparityindicator = calculateDisparity(df)
    df["Disparity"] = disparity
    df["Disparity_Indicator"] = disparityindicator

    kst, kst_crossover = calculateKST(df)
    df["KST"] = kst
    df["KST_Crossover"] = kst_crossover

    macd, macdcrossover = calculateMACD(df)
    df["MACD"]= macd
    df["MACD_Crossover"] = macdcrossover

    df["OBV"] = calculateOBV(df)

    rsi, rsi_indicator = calculateRSI(df)
    df["RSI"]= rsi
    df["RSI_Indicator"] = rsi_indicator

    rvi, rvi_crossover, rvi_divergence = calculateRVI(df)
    df["RVI"] = rvi
    df["RVI_Crossover"] = rvi_crossover
    df["RVI_Divergence"] = rvi_divergence

    so, so_signal = calculateStochasticOscillator(df)
    df["Stochastic_Oscillator"] = so
    df["Stochastic_Oscillator_Indicator"] = so_signal

    return df