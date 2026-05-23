import os
import sys

# Ensure parent directory is in path to import stock_app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_app.ml_service import fetch_and_predict, clean_stock_ticker

def main():
    print("=" * 60)
    print("      INDIAN SHARE MARKET PREDICTOR (CLI MODE)      ")
    print("=" * 60)
    
    ticker_input = input("Enter Indian Stock Ticker (e.g., RELIANCE, TATAMOTORS, SBIN, NIFTY): ")
    if not ticker_input.strip():
        ticker_input = "RELIANCE"
        
    try:
        ticker = clean_stock_ticker(ticker_input)
        print(f"\nFetching live data and training RandomForest model for {ticker}...")
        
        result = fetch_and_predict(ticker)
        
        print("\n" + "=" * 50)
        print(f" STOCK: {result['stock_name']} ({result['ticker']})")
        print(f" CURRENT PRICE: ₹{result['current_price']:,} ({result['price_change']:+}, {result['price_change_pct']:.2f}%)")
        print("-" * 50)
        
        pred_color = "UP" if result['prediction'] == "UP" else "DOWN"
        print(f" NEXT-DAY MOVEMENT PREDICTION: {pred_color}")
        print(f" MODEL CONFIDENCE: {result['confidence']}%")
        print("=" * 50)
        
        print("\n[Technical Signals Summary]")
        for sig in result['tech_signals']:
            print(f" • {sig['indicator']}: {sig['signal']} | {sig['desc']}")
            
        print("\n[Top Model Features]")
        for feat in result['feature_importances'][:4]:
            print(f" • {feat['feature']}: {feat['importance']}% importance")
            
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Please verify your internet connection and try a valid ticker symbol.")

if __name__ == "__main__":
    main()
    