from django.shortcuts import render
from django.http import JsonResponse
from .ml_service import fetch_and_predict, POPULAR_INDIAN_STOCKS

def dashboard_home(request):
    """
    Renders the gorgeous dark-themed stock predictor dashboard home.
    """
    # Pass down the popular stock lists to build the grid dynamically in the template
    stock_list = []
    for ticker, name in POPULAR_INDIAN_STOCKS.items():
        # Shorten stock name for UI grid aesthetics
        short_name = name.split(" Ltd.")[0].split(" (")[0]
        if ticker == "^NSEI":
            short_name = "NIFTY 50"
        elif ticker == "^BSESN":
            short_name = "SENSEX"
        elif ticker == "^NSEBANK":
            short_name = "BANK NIFTY"
            
        stock_list.append({
            "ticker": ticker,
            "name": name,
            "short_name": short_name,
            # Generate static class types for specific styling if needed
            "symbol": ticker.replace(".NS", "").replace(".BO", "").replace("^", "")
        })
        
    context = {
        "popular_stocks": stock_list
    }
    return render(request, 'index.html', context)

def predict_stock(request):
    """
    AJAX endpoint: Performs live data fetching and dynamic ML training/prediction,
    returning results in a comprehensive JSON.
    """
    ticker = request.GET.get('ticker', 'RELIANCE.NS').strip()
    if not ticker:
        ticker = 'RELIANCE.NS'
        
    try:
        data = fetch_and_predict(ticker)
        return JsonResponse(data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({
            "error": str(e),
            "status": "failed"
        }, status=400)
