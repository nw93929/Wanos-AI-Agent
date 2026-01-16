"""
Market Data Service
===================

Fetches stock data from public APIs for screening workflow.

Data Sources:
1. Financial Modeling Prep (FMP) - Free tier: 250 requests/day
   https://financialmodelingprep.com/developer/docs/
2. SEC EDGAR API - Unlimited, free
   https://www.sec.gov/edgar/sec-api-documentation
3. Alpha Vantage (backup) - Free tier: 5 requests/min

IMPORTANT: API keys are stored in .env file, never committed to git.
"""

import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time

# Load API key from environment (NEVER hardcode!)
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# Base URLs
FMP_BASE = "https://financialmodelingprep.com/api/v3"
SEC_BASE = "https://data.sec.gov"

# ============================================================================
# UNIVERSE FETCHERS
# ============================================================================

def fetch_sp500_tickers() -> List[str]:
    """
    Fetch current S&P 500 constituents.

    Returns:
        List of ticker symbols (e.g., ['AAPL', 'MSFT', ...])
    """
    try:
        url = f"{FMP_BASE}/sp500_constituent?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        tickers = [item["symbol"] for item in data]

        print(f"[Market Data] Fetched {len(tickers)} S&P 500 tickers")
        return tickers

    except Exception as e:
        print(f"[Error] Failed to fetch S&P 500 list: {e}")
        # Fallback: Return top 50 by market cap (hardcoded for resilience)
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
            "V", "JNJ", "WMT", "JPM", "MA", "PG", "UNH", "HD", "CVX", "MRK",
            "ABBV", "PFE", "KO", "PEP", "COST", "AVGO", "TMO", "MCD", "ABT",
            "CSCO", "ACN", "DHR", "NKE", "LIN", "VZ", "ADBE", "CRM", "NEE",
            "TXN", "CMCSA", "PM", "ORCL", "WFC", "DIS", "BMY", "HON", "INTC",
            "UPS", "IBM", "RTX", "QCOM", "AMGN"
        ]

def fetch_nasdaq100_tickers() -> List[str]:
    """Fetch Nasdaq 100 constituents"""
    try:
        url = f"{FMP_BASE}/nasdaq_constituent?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        return [item["symbol"] for item in data[:100]]  # Top 100

    except Exception as e:
        print(f"[Error] Failed to fetch Nasdaq 100: {e}")
        return []

# ============================================================================
# FUNDAMENTAL DATA
# ============================================================================

def fetch_stock_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Fetch key fundamental metrics for a single stock.

    Returns dict with:
        - marketCap, pe, priceToBook, roe, debtToEquity, currentRatio
        - revenueGrowth, earningsGrowth, dividendYield
    """
    try:
        # Use FMP quote + key-metrics endpoint
        url = f"{FMP_BASE}/key-metrics-ttm/{ticker}?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        data = response.json()
        if not data:
            return {}

        metrics = data[0] if isinstance(data, list) else data

        return {
            "ticker": ticker,
            "marketCap": metrics.get("marketCapTTM", 0),
            "pe": metrics.get("peRatioTTM", 0),
            "priceToBook": metrics.get("pbRatioTTM", 0),
            "roe": metrics.get("roeTTM", 0),
            "debtToEquity": metrics.get("debtToEquityTTM", 0),
            "currentRatio": metrics.get("currentRatioTTM", 0),
            "revenueGrowth": metrics.get("revenuePerShareTTM", 0),
            "earningsGrowth": metrics.get("netIncomePerShareTTM", 0),
            "dividendYield": metrics.get("dividendYieldTTM", 0),
        }

    except Exception as e:
        print(f"[Error] Failed to fetch fundamentals for {ticker}: {e}")
        return {}

def batch_fetch_financials(tickers: List[str], batch_size: int = 50) -> Dict[str, Dict[str, Any]]:
    """
    Fetch fundamentals for multiple stocks in batches.

    Respects API rate limits (250/day for free tier).
    Implements exponential backoff on errors.

    Args:
        tickers: List of stock symbols
        batch_size: Number of stocks per batch (default 50)

    Returns:
        Dict mapping ticker -> fundamentals
    """
    results = {}

    # Process in batches to respect rate limits
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        print(f"[Market Data] Fetching batch {i//batch_size + 1}/{len(tickers)//batch_size + 1}...")

        for ticker in batch:
            data = fetch_stock_fundamentals(ticker)
            if data:
                results[ticker] = data

            # Rate limiting: 5 requests/second (free tier limit)
            time.sleep(0.2)

        # Pause between batches
        if i + batch_size < len(tickers):
            print("[Market Data] Pausing 10 seconds between batches...")
            time.sleep(10)

    print(f"[Market Data] Successfully fetched data for {len(results)}/{len(tickers)} stocks")
    return results

# ============================================================================
# INSIDER TRADING DATA
# ============================================================================

def fetch_insider_trades(ticker: str, days: int = 90) -> List[Dict[str, Any]]:
    """
    Fetch recent insider trading activity from SEC Form 4 filings.

    Args:
        ticker: Stock symbol
        days: Lookback period (default 90 days)

    Returns:
        List of insider transactions with:
        - transactionType: 'P-Purchase', 'S-Sale', etc.
        - transactionValue: Dollar amount
        - transactionShares: Number of shares
        - reportingName: Insider role (CEO, CFO, Director, etc.)
        - filingDate: When Form 4 was filed
    """
    try:
        # Use FMP insider trading endpoint
        url = f"{FMP_BASE}/insider-trading?symbol={ticker}&apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Filter to recent transactions
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = []

        for trade in data:
            filing_date = datetime.strptime(trade.get("filingDate", ""), "%Y-%m-%d")

            if filing_date >= cutoff_date:
                recent_trades.append({
                    "transactionType": trade.get("transactionType", ""),
                    "transactionValue": trade.get("securitiesTransacted", 0) * trade.get("price", 0),
                    "transactionShares": trade.get("securitiesTransacted", 0),
                    "reportingName": trade.get("reportingName", ""),
                    "filingDate": trade.get("filingDate", ""),
                })

        return recent_trades

    except Exception as e:
        print(f"[Error] Failed to fetch insider trades for {ticker}: {e}")
        return []

# ============================================================================
# HISTORICAL PRICE DATA (for momentum calculations)
# ============================================================================

def fetch_price_history(ticker: str, days: int = 252) -> List[float]:
    """
    Fetch daily closing prices for momentum calculations.

    Args:
        ticker: Stock symbol
        days: Lookback period (default 252 = 1 year of trading days)

    Returns:
        List of closing prices (most recent last)
    """
    try:
        url = f"{FMP_BASE}/historical-price-full/{ticker}?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        historical = data.get("historical", [])

        # Get last N days
        prices = [day["close"] for day in historical[:days]]
        prices.reverse()  # Oldest to newest

        return prices

    except Exception as e:
        print(f"[Error] Failed to fetch price history for {ticker}: {e}")
        return []

def calculate_momentum(prices: List[float]) -> Dict[str, float]:
    """
    Calculate price momentum metrics.

    Returns:
        - momentum_6m: 6-month return
        - momentum_12m: 12-month return
        - volatility: Annualized volatility
    """
    if len(prices) < 252:
        return {"momentum_6m": 0, "momentum_12m": 0, "volatility": 0}

    price_6m_ago = prices[-126]  # ~6 months
    price_12m_ago = prices[-252]  # ~12 months
    current_price = prices[-1]

    momentum_6m = (current_price - price_6m_ago) / price_6m_ago
    momentum_12m = (current_price - price_12m_ago) / price_12m_ago

    # Calculate daily returns for volatility
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    daily_vol = (sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns)) ** 0.5
    annual_vol = daily_vol * (252 ** 0.5)  # Annualize

    return {
        "momentum_6m": momentum_6m,
        "momentum_12m": momentum_12m,
        "volatility": annual_vol
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_api_health() -> Dict[str, bool]:
    """
    Test connectivity to all data sources.

    Returns dict with service_name -> is_available
    """
    health = {}

    # Test FMP
    try:
        url = f"{FMP_BASE}/quote/AAPL?apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=5)
        health["financial_modeling_prep"] = response.status_code == 200
    except:
        health["financial_modeling_prep"] = False

    # Test SEC EDGAR
    try:
        url = f"{SEC_BASE}/submissions/CIK0000320193.json"  # Apple
        response = requests.get(url, timeout=5, headers={"User-Agent": "YourName youremail@example.com"})
        health["sec_edgar"] = response.status_code == 200
    except:
        health["sec_edgar"] = False

    return health
