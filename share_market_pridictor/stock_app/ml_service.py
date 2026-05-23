import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Global thread-safe in-memory cache to solve latency and CPU GIL blocking
PREDICTION_CACHE = {}


POPULAR_INDIAN_STOCKS = {
    "RELIANCE.NS": "Reliance Industries Ltd.",
    "TCS.NS": "Tata Consultancy Services Ltd.",
    "HDFCBANK.NS": "HDFC Bank Ltd.",
    "INFY.NS": "Infosys Ltd.",
    "ICICIBANK.NS": "ICICI Bank Ltd.",
    "TATAMOTORS.NS": "Tata Motors Ltd.",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel Ltd.",
    "ITC.NS": "ITC Ltd.",
    "^NSEI": "NIFTY 50 (NSE Index)",
    "^BSESN": "SENSEX (BSE Index)",
    "^NSEBANK": "BANK NIFTY (NSE Index)",
    "WIPRO.NS": "Wipro Ltd.",
    "LT.NS": "Larsen & Toubro Ltd.",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd.",
    "ADANIENT.NS": "Adani Enterprises Ltd."
}

STOCK_NAME_MAPPINGS = {
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "ICICI": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "SBI": "SBIN.NS",
    "SBIN": "SBIN.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "BHARTI": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "WIPRO": "WIPRO.NS",
    "L&T": "LT.NS",
    "LARSEN": "LT.NS",
    "LARSEN & TOUBRO": "LT.NS",
    "HINDUSTAN UNILEVER": "HINDUNILVR.NS",
    "HUL": "HINDUNILVR.NS",
    "UNILEVER": "HINDUNILVR.NS",
    "ADANI": "ADANIENT.NS",
    "ADANI ENTERPRISES": "ADANIENT.NS",
    "NIFTY": "^NSEI",
    "NIFTY 50": "^NSEI",
    "NIFTY50": "^NSEI",
    "NSE": "^NSEI",
    "SENSEX": "^BSESN",
    "BSE": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
    "BANK NIFTY": "^NSEBANK",
    "NIFTYBANK": "^NSEBANK",
    "NIFTY BANK": "^NSEBANK",
    "TATA STEEL": "TATASTEEL.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "AXIS": "AXISBANK.NS",
    "AXIS BANK": "AXISBANK.NS",
    "MARUTI": "MARUTI.NS",
    "SUZUKI": "MARUTI.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "KOTAK": "KOTAKBANK.NS",
    "KOTAK BANK": "KOTAKBANK.NS",
    "JIO": "JIOFIN.NS",
    "JIO FINANCIAL": "JIOFIN.NS",
    "ONGC": "ONGC.NS",
    "POWER GRID": "POWERGRID.NS",
    "NTPC": "NTPC.NS"
}

def clean_stock_ticker(ticker):
    """
    Cleans stock tickers to match Indian market standards (NSE by default).
    Handles spaces, abbreviations, and maps common stock names to correct tickers.
    """
    if not ticker:
        return "RELIANCE.NS"
        
    query = ticker.strip().upper()
    
    # Direct dictionary mapping check
    if query in STOCK_NAME_MAPPINGS:
        return STOCK_NAME_MAPPINGS[query]
        
    # Check with spaces removed (e.g. "TATA MOTORS" -> "TATAMOTORS")
    query_no_space = query.replace(" ", "").replace("-", "")
    if query_no_space in STOCK_NAME_MAPPINGS:
        return STOCK_NAME_MAPPINGS[query_no_space]
        
    for name_key, target_ticker in STOCK_NAME_MAPPINGS.items():
        if query_no_space == name_key.replace(" ", ""):
            return target_ticker
            
    # Suffix matching
    if query in ["NIFTY", "NIFTY50", "NIFTY 50", "^NSEI", "NSE"]:
        return "^NSEI"
    if query in ["SENSEX", "^BSESN", "BSE"]:
        return "^BSESN"
    if query in ["BANKNIFTY", "BANK NIFTY", "NIFTYBANK", "NIFTY BANK", "^NSEBANK"]:
        return "^NSEBANK"
        
    # Standardize NSE tickers (append .NS if it doesn't have suffix and isn't index)
    if not (query.endswith(".NS") or query.endswith(".BO") or query.startswith("^")):
        query = f"{query_no_space}.NS"
    else:
        query = query_no_space
        
    return query

def get_stock_name(ticker):
    return POPULAR_INDIAN_STOCKS.get(ticker, ticker.replace(".NS", "").replace(".BO", "") + " (NSE/BSE Equity)")

def fetch_and_predict(ticker_symbol):
    ticker = clean_stock_ticker(ticker_symbol)
    stock_name = get_stock_name(ticker)
    
    # 1. In-Memory Cache Check for instant serving (< 2ms)
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    cache_key = (ticker, today_str)
    if cache_key in PREDICTION_CACHE:
        print(f"CACHE HIT: Serving cached predictions instantly for {ticker} on {today_str}")
        return PREDICTION_CACHE[cache_key]
        
    # Fetch 2.5 years of historical daily data to have enough for MA200 and robust training
    end_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=int(2.5*365))).strftime('%Y-%m-%d')
    
    print(f"Downloading live data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty or len(df) < 220:
        raise ValueError(f"Ticker symbol '{ticker_symbol}' was not found or lacks sufficient trading history (minimum 220 trading days required).")
        
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Standardize index name
    df.index.name = 'Date'
    
    # Ensure standard price columns are present and clean up any NaNs
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            
    # Handle Volume specifically (indices often lack Volume reporting or have NaNs)
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0.0)
    else:
        df['Volume'] = 0.0
    
    # 1. Technical Indicators Engineering
    # Moving Averages
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # RSI (Relative Strength Index - 14 Days)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    # NEW: 4 Advanced Technical Indicators to maximize accuracy
    # A. CCI (Commodity Channel Index - 20 periods)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad_tp = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (tp - sma_tp) / (0.015 * mad_tp + 1e-9)

    # B. ATR (Average True Range - 14 periods)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # C. ROC (Rate of Change - 10 periods)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100

    # D. OBV (On-Balance Volume)
    direction = np.where(df['Close'].diff() > 0, 1, np.where(df['Close'].diff() < 0, -1, 0))
    df['OBV'] = (direction * df['Volume']).cumsum()
    
    # NEW features for high accuracy swing trend predictions
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_3d'] = df['Close'].pct_change(3)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_Volatility'] = df['Return_1d'].rolling(window=10).std()
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['MACD_Lag1'] = df['MACD'].shift(1)

    # 1b. Stationary / Relative Feature Normalization (Fixes price-level leakage & BEARISH BIAS)
    df['Open_Pct'] = (df['Open'] - df['Close']) / df['Close']
    df['High_Pct'] = (df['High'] - df['Close']) / df['Close']
    df['Low_Pct'] = (df['Low'] - df['Close']) / df['Close']
    df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-9)
    df['MA10_Pct'] = (df['MA10'] - df['Close']) / df['Close']
    df['MA50_Pct'] = (df['MA50'] - df['Close']) / df['Close']
    df['MA200_Pct'] = (df['MA200'] - df['Close']) / df['Close']
    df['BB_Upper_Pct'] = (df['BB_Upper'] - df['Close']) / df['Close']
    df['BB_Lower_Pct'] = (df['BB_Lower'] - df['Close']) / df['Close']
    df['MACD_Pct'] = df['MACD'] / df['Close']
    df['MACD_Signal_Pct'] = df['MACD_Signal'] / df['Close']
    df['MACD_Lag1_Pct'] = df['MACD_Lag1'] / df['Close']
    df['ATR_Pct'] = df['ATR'] / df['Close']
    df['OBV_Pct'] = (df['OBV'] - df['OBV'].rolling(window=20).mean()) / (df['OBV'].rolling(window=20).std() + 1e-9)

    # Create target: simple next-day closing price movement
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Keep a copy of full data with indicators before dropping NaN (for charting latest records)
    full_df = df.copy()
    
    # Drop rows with NaN values for ML training
    train_df = df.dropna().copy()
    
    if len(train_df) < 100:
        raise ValueError("Insufficient trading data points after applying technical indicators.")
        
    # Feature columns (incorporating CCI, ATR, ROC, OBV and newly engineered high-accuracy features)
    feature_cols = [
        'Open_Pct', 'High_Pct', 'Low_Pct', 'Volume_Ratio', 
        'MA10_Pct', 'MA50_Pct', 'MA200_Pct', 'RSI', 
        'MACD_Pct', 'MACD_Signal_Pct', 'BB_Upper_Pct', 'BB_Lower_Pct',
        'CCI', 'ATR_Pct', 'ROC', 'OBV_Pct',
        'Return_1d', 'Return_3d', 'Return_5d', 
        'Return_Volatility', 'RSI_Lag1', 'MACD_Lag1_Pct'
    ]
    
    X = train_df[feature_cols]
    y = train_df['Target']
    
    # Chronological Split (15% test split for time-series out-of-sample backtesting)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    # Fit scaling preprocessing on training set
    scaler_eval = StandardScaler()
    X_train_scaled = scaler_eval.fit_transform(X_train)
    X_test_scaled = scaler_eval.transform(X_test)
    
    # Elite Voting Ensemble Model for evaluation (shallower tree parameters + LogisticRegression)
    rf_eval = RandomForestClassifier(n_estimators=70, max_depth=6, min_samples_leaf=5, random_state=42)
    hgb_eval = HistGradientBoostingClassifier(max_iter=40, max_depth=3, learning_rate=0.01, l2_regularization=5.0, random_state=42)
    lr_eval = LogisticRegression(C=0.05, penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
    ensemble_eval = VotingClassifier(
        estimators=[('rf', rf_eval), ('hgb', hgb_eval), ('lr', lr_eval)],
        voting='soft'
    )
    ensemble_eval.fit(X_train_scaled, y_train)
    
    # Predict on test set using dynamic OMRO blending for each out-of-sample row
    y_pred = []
    for idx in range(len(X_test)):
        row_feat = pd.DataFrame([X_test.iloc[idx]])
        row_scaled = scaler_eval.transform(row_feat)
        ml_p_up = ensemble_eval.predict_proba(row_scaled)[0][1]
        
        # Look up corresponding raw row in train_df
        df_row = train_df.iloc[len(X_train) + idx]
        r_rsi = float(df_row['RSI'])
        r_cci = float(df_row['CCI'])
        r_close = float(df_row['Close'])
        r_bbl = float(df_row['BB_Lower'])
        r_bbu = float(df_row['BB_Upper'])
        r_bbm = float(df_row['BB_Mid'])
        
        # Calculate reversion scores
        r_rsi_s = np.clip((35 - r_rsi)/15 if r_rsi < 35 else ((65 - r_rsi)/15 if r_rsi > 65 else 0.0), -1.0, 1.0)
        r_cci_s = np.clip((-100 - r_cci)/100 if r_cci < -100 else ((100 - r_cci)/100 if r_cci > 100 else 0.0), -1.0, 1.0)
        r_bb_s = np.clip((r_bbm - r_close) / (((r_bbu - r_bbl) + 1e-9) / 2), -1.0, 1.0)
        r_osc_sig = (r_rsi_s + r_cci_s + r_bb_s) / 3.0
        
        if abs(r_osc_sig) > 0.1:
            row_rev_prob = 0.5 + (0.5 * r_osc_sig)
            row_final_p_up = (0.70 * ml_p_up) + (0.30 * row_rev_prob)
        else:
            row_final_p_up = ml_p_up
            
        y_pred.append(1 if row_final_p_up >= 0.5 else 0)
    y_pred = np.array(y_pred)
    
    # Backtest metrics calculation
    test_acc = np.mean(y_pred == y_test)
    
    # Precision & Recall (Bullish target=1)
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    predicted_positives = np.sum(y_pred == 1)
    actual_positives = np.sum(y_test == 1)
    
    precision = (true_positives / predicted_positives) if predicted_positives > 0 else 0.0
    recall = (true_positives / actual_positives) if actual_positives > 0 else 0.0
    
    # Retrain on 100% of available data for the final next-day forecasting prediction
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    rf_final = RandomForestClassifier(n_estimators=70, max_depth=6, min_samples_leaf=5, random_state=42)
    hgb_final = HistGradientBoostingClassifier(max_iter=40, max_depth=3, learning_rate=0.01, l2_regularization=5.0, random_state=42)
    lr_final = LogisticRegression(C=0.05, penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
    
    ensemble_final = VotingClassifier(
        estimators=[('rf', rf_final), ('hgb', hgb_final), ('lr', lr_final)],
        voting='soft'
    )
    ensemble_final.fit(X_scaled, y)
    
    # Calculate In-Sample Training Accuracy with vectorized OMRO blending
    train_rsi = train_df['RSI'].values
    train_cci = train_df['CCI'].values
    train_close = train_df['Close'].values
    train_bbl = train_df['BB_Lower'].values
    train_bbu = train_df['BB_Upper'].values
    train_bbm = train_df['BB_Mid'].values
    
    train_rsi_s = np.clip(np.where(train_rsi < 35, (35 - train_rsi)/15, np.where(train_rsi > 65, (65 - train_rsi)/15, 0.0)), -1.0, 1.0)
    train_cci_s = np.clip(np.where(train_cci < -100, (-100 - train_cci)/100, np.where(train_cci > 100, (100 - train_cci)/100, 0.0)), -1.0, 1.0)
    train_bb_s = np.clip((train_bbm - train_close) / (((train_bbu - train_bbl) + 1e-9) / 2), -1.0, 1.0)
    train_osc_sig = (train_rsi_s + train_cci_s + train_bb_s) / 3.0
    
    train_ml_p_up = ensemble_final.predict_proba(X_scaled)[:, 1]
    train_rev_prob = 0.5 + (0.5 * train_osc_sig)
    
    train_final_p_up = np.where(np.abs(train_osc_sig) > 0.1, 
                                (0.70 * train_ml_p_up) + (0.30 * train_rev_prob), 
                                train_ml_p_up)
    
    y_train_pred = (train_final_p_up >= 0.5).astype(int)
    train_acc = np.mean(y_train_pred == y)
    
    # Calculate Feature Importances of the final trained model
    importances = ensemble_final.named_estimators_['rf'].feature_importances_
    feature_importance_dict = [
        {"feature": feat, "importance": round(float(imp) * 100, 2)}
        for feat, imp in zip(feature_cols, importances)
    ]
    feature_importance_dict = sorted(feature_importance_dict, key=lambda x: x["importance"], reverse=True)
    
    # Predict next day's movement (using the absolute latest row of data)
    latest_row = full_df.iloc[-1]
    latest_df = pd.DataFrame([full_df[feature_cols].iloc[-1]])
    
    # If there are NaNs in latest features, backfill and forward fill
    if latest_df.isnull().any().any():
        full_df.ffill(inplace=True)
        full_df.bfill(inplace=True)
        latest_df = pd.DataFrame([full_df[feature_cols].iloc[-1]])
        latest_row = full_df.iloc[-1]
 
    latest_scaled = scaler_final.transform(latest_df)
    ml_prob_up = ensemble_final.predict_proba(latest_scaled)[0][1]
    
    # OMRO for the latest row
    rsi_val = float(latest_row['RSI'])
    cci_val = float(latest_row['CCI'])
    close_price = float(latest_row['Close'])
    bb_lower = float(latest_row['BB_Lower'])
    bb_upper = float(latest_row['BB_Upper'])
    bb_mid = float(latest_row['BB_Mid'])
    
    rsi_score = np.clip((35 - rsi_val) / 15 if rsi_val < 35 else ((65 - rsi_val) / 15 if rsi_val > 65 else 0.0), -1.0, 1.0)
    cci_score = np.clip((-100 - cci_val) / 100 if cci_val < -100 else ((100 - cci_val) / 100 if cci_val > 100 else 0.0), -1.0, 1.0)
    bb_score = np.clip((bb_mid - close_price) / (((bb_upper - bb_lower) + 1e-9) / 2), -1.0, 1.0)
    
    oscillator_signal = (rsi_score + cci_score + bb_score) / 3.0
    
    if abs(oscillator_signal) > 0.1:
        reversion_prob = 0.5 + (0.5 * oscillator_signal)
        final_prob_up = (0.70 * ml_prob_up) + (0.30 * reversion_prob)
    else:
        final_prob_up = ml_prob_up
        
    prediction = 1 if final_prob_up >= 0.5 else 0
    confidence = round(float(final_prob_up if prediction == 1 else (1.0 - final_prob_up)) * 100, 1)
    
    # 2. Dynamic Technical Analysis Breakdown (Expert Explainer)
    tech_signals = []
    
    # RSI Signal
    rsi_val = float(latest_row['RSI'])
    if rsi_val < 30:
        rsi_sig = "Strongly Bullish"
        rsi_desc = f"RSI is at {rsi_val:.1f}, indicating stock is in oversold territory. A reversal is highly likely."
        rsi_status = "bullish"
    elif rsi_val > 70:
        rsi_sig = "Strongly Bearish"
        rsi_desc = f"RSI is at {rsi_val:.1f}, indicating stock is overbought. Selling pressure could build up."
        rsi_status = "bearish"
    else:
        rsi_sig = "Neutral"
        rsi_desc = f"RSI is at {rsi_val:.1f}, which is in the standard trading zone (30-70)."
        rsi_status = "neutral"
    tech_signals.append({"indicator": "Relative Strength Index (RSI)", "value": f"{rsi_val:.1f}", "signal": rsi_sig, "desc": rsi_desc, "status": rsi_status})
    
    # MACD Signal
    macd_val = float(latest_row['MACD'])
    signal_val = float(latest_row['MACD_Signal'])
    if macd_val > signal_val:
        macd_sig = "Bullish"
        macd_desc = f"MACD line ({macd_val:.3f}) is above the Signal line ({signal_val:.3f}), forming a bullish crossover."
        macd_status = "bullish"
    else:
        macd_sig = "Bearish"
        macd_desc = f"MACD line ({macd_val:.3f}) is below the Signal line ({signal_val:.3f}), indicating negative momentum."
        macd_status = "bearish"
    tech_signals.append({"indicator": "MACD Crossover", "value": f"{macd_val:.3f}", "signal": macd_sig, "desc": macd_desc, "status": macd_status})
    
    # Moving Average Signal
    close_price = float(latest_row['Close'])
    ma50_val = float(latest_row['MA50'])
    ma200_val = float(latest_row['MA200'])
    if close_price > ma50_val and ma50_val > ma200_val:
        ma_sig = "Strong Bullish Trend"
        ma_desc = f"Price ({close_price:.2f}) is above the 50-day MA ({ma50_val:.2f}) which is above the 200-day MA ({ma200_val:.2f})."
        ma_status = "bullish"
    elif close_price > ma50_val:
        ma_sig = "Bullish"
        ma_desc = f"Price is trading above its medium-term 50-day moving average ({ma50_val:.2f})."
        ma_status = "bullish"
    elif close_price < ma50_val and ma50_val < ma200_val:
        ma_sig = "Strong Bearish Trend"
        ma_desc = f"Price ({close_price:.2f}) is below both the 50-day MA ({ma50_val:.2f}) and 200-day MA ({ma200_val:.2f})."
        ma_status = "bearish"
    else:
        ma_sig = "Neutral / Consolidation"
        ma_desc = f"Price ({close_price:.2f}) is sandwiching between the 50-day MA ({ma50_val:.2f}) and 200-day MA."
        ma_status = "neutral"
    tech_signals.append({"indicator": "Moving Averages (MA)", "value": f"{close_price:.2f} (Close)", "signal": ma_sig, "desc": ma_desc, "status": ma_status})
    
    # Bollinger Bands Signal
    bb_upper = float(latest_row['BB_Upper'])
    bb_lower = float(latest_row['BB_Lower'])
    if close_price > bb_upper:
        bb_sig = "Bearish Breakout"
        bb_desc = f"Price has closed above the Upper Bollinger Band ({bb_upper:.2f}). Stock might be overextended."
        bb_status = "bearish"
    elif close_price < bb_lower:
        bb_sig = "Bullish Reversal"
        bb_desc = f"Price has dropped below the Lower Bollinger Band ({bb_lower:.2f}). Potential bounce upcoming."
        bb_status = "bullish"
    else:
        bb_sig = "Neutral Range"
        bb_desc = f"Price is safely trading inside the Bollinger volatility band ({bb_lower:.2f} - {bb_upper:.2f})."
        bb_status = "neutral"
    tech_signals.append({"indicator": "Bollinger Bands Volatility", "value": f"{close_price:.2f} (Close)", "signal": bb_sig, "desc": bb_desc, "status": bb_status})

    # Prepare Historical Data for Charting (Last 120 Trading Days)
    chart_df = full_df.tail(120).copy()
    dates = [d.strftime('%Y-%m-%d') for d in chart_df.index]
    prices = [round(float(p), 2) for p in chart_df['Close']]
    ma50_series = [round(float(p), 2) if not np.isnan(p) else None for p in chart_df['MA50']]
    ma200_series = [round(float(p), 2) if not np.isnan(p) else None for p in chart_df['MA200']]
    volumes = [int(v) for v in chart_df['Volume']]
    
    # Stock Statistics
    # Calculate 52-week High/Low
    last_year_df = full_df.tail(252)
    high_52w = float(last_year_df['High'].max())
    low_52w = float(last_year_df['Low'].min())
    
    # Calculate change
    prev_close = float(full_df.iloc[-2]['Close']) if len(full_df) > 1 else close_price
    price_change = close_price - prev_close
    price_change_pct = (price_change / prev_close) * 100

    response_data = {
        "ticker": ticker,
        "stock_name": stock_name,
        "current_price": round(close_price, 2),
        "price_change": round(price_change, 2),
        "price_change_pct": round(price_change_pct, 2),
        "open": round(float(latest_row['Open']), 2),
        "high": round(float(latest_row['High']), 2),
        "low": round(float(latest_row['Low']), 2),
        "volume": int(latest_row['Volume']),
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "prediction": "UP" if prediction == 1 else "DOWN",
        "confidence": confidence,
        "backtest_acc": round(float(test_acc) * 100, 1),
        "backtest_precision": round(float(precision) * 100, 1),
        "backtest_recall": round(float(recall) * 100, 1),
        "training_acc": round(float(train_acc) * 100, 1),
        "training_days": len(train_df),
        "tech_signals": tech_signals,
        "feature_importances": feature_importance_dict,
        "chart_data": {
            "dates": dates,
            "prices": prices,
            "ma50": ma50_series,
            "ma200": ma200_series,
            "volumes": volumes
        }
    }
    
    # Cache the result for today
    PREDICTION_CACHE[cache_key] = response_data
    return response_data
