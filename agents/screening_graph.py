"""
Stock Screening Agent Workflow
===============================

This graph implements a multi-stage screening pipeline to identify high-potential stocks
from a large universe using fundamental analysis, insider trading signals, and proven
investment strategies (Warren Buffett, Peter Lynch, etc.).

## Screening Workflow:

1. **Universe Builder**: Fetch list of stocks to screen (S&P 500, Russell 2000, etc.)
2. **Quick Filter**: Apply quantitative filters to narrow down candidates
   - Market cap > $1B
   - Positive earnings
   - Debt/Equity < 1.0
   - ROE > 15%
3. **Insider Activity Analyzer**: Check for significant insider buying
4. **Strategy Scorer**: Score each stock against famous investor criteria
   - Warren Buffett: Economic moat, consistent earnings, low debt
   - Peter Lynch: PEG < 1, growing earnings, understandable business
   - Benjamin Graham: P/B < 1.5, P/E < 15, current ratio > 2
5. **Deep Research**: Run full research workflow on top 10-20 candidates
6. **Portfolio Constructor**: Rank and recommend final portfolio

## Data Sources:
- Financial Modeling Prep API (free tier: financialmodelingprep.com)
- SEC EDGAR API (insider trading Form 4 filings)
- Yahoo Finance (backup/supplementary data)
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

from agents.state_screening import ScreeningState
from agents.prompts_screening import (
    UNIVERSE_BUILDER_SYSTEM,
    QUICK_FILTER_SYSTEM,
    INSIDER_ANALYZER_SYSTEM,
    STRATEGY_SCORER_SYSTEM,
    PORTFOLIO_CONSTRUCTOR_SYSTEM
)
from services.market_data import (
    fetch_sp500_tickers,
    fetch_stock_fundamentals,
    fetch_insider_trades,
    batch_fetch_financials
)
from services.local_models import generate_reasoning_response

# Two-tier model system:
# Tier 1: GPT-5-nano for quick screening (cheap, fast)
screening_model = ChatOpenAI(model="gpt-5-nano", temperature=0)

# Tier 2: Local reasoning model for deep analysis (configured in .env)
# Options: "deepseek-r1-14b" (recommended), "qwen2.5-14b" (balanced)
REASONING_MODEL = os.getenv("REASONING_MODEL", "deepseek-r1-14b")

# ============================================================================
# NODE DEFINITIONS
# ============================================================================

def universe_builder_node(state: ScreeningState) -> dict:
    """
    Build initial universe of stocks to screen.

    Based on user criteria, selects appropriate stock universe:
    - "large cap" -> S&P 500
    - "all US stocks" -> Russell 3000
    - "tech" -> Nasdaq 100
    - "small cap value" -> Russell 2000 value stocks

    Returns list of tickers with basic metadata.
    """
    criteria = state["criteria"]

    # Use screening model (fast/cheap) to interpret criteria
    response = screening_model.invoke([
        {"role": "system", "content": UNIVERSE_BUILDER_SYSTEM},
        {"role": "user", "content": f"User wants to screen stocks with criteria: {criteria}. Which universe should we use?"}
    ])

    # For now, default to S&P 500 (500 stocks)
    # TODO: Parse LLM response to select appropriate index
    tickers = fetch_sp500_tickers()

    return {
        "universe": tickers,
        "universe_size": len(tickers)
    }

def quick_filter_node(state: ScreeningState) -> dict:
    """
    Apply quantitative filters to narrow down candidates.

    Filters applied:
    1. Market cap > $1B (avoid micro-caps)
    2. Positive net income (profitable companies)
    3. Debt-to-Equity < 1.0 (manageable debt)
    4. Return on Equity (ROE) > 15% (efficient capital use)
    5. Current Ratio > 1.5 (liquidity)

    This reduces 500 stocks -> ~50-100 candidates
    """
    tickers = state["universe"]
    max_stocks = state.get("max_stocks", 20)

    print(f"[Quick Filter] Screening {len(tickers)} stocks...")

    # Batch fetch fundamentals for all tickers
    fundamentals = batch_fetch_financials(tickers)

    # Apply filters
    candidates = []
    for ticker, data in fundamentals.items():
        try:
            # Extract metrics (handle missing data)
            market_cap = data.get("marketCap", 0)
            net_income = data.get("netIncome", 0)
            debt_to_equity = data.get("debtToEquity", 999)
            roe = data.get("roe", 0)
            current_ratio = data.get("currentRatio", 0)

            # Apply filters
            if (market_cap > 1_000_000_000 and  # $1B+
                net_income > 0 and              # Profitable
                debt_to_equity < 1.0 and        # Low debt
                roe > 0.15 and                  # 15%+ ROE
                current_ratio > 1.5):           # Good liquidity

                candidates.append({
                    "ticker": ticker,
                    "market_cap": market_cap,
                    "roe": roe,
                    "debt_to_equity": debt_to_equity,
                    "fundamentals": data
                })
        except (KeyError, TypeError):
            # Skip stocks with missing data
            continue

    # Sort by ROE (best performers first)
    candidates = sorted(candidates, key=lambda x: x["roe"], reverse=True)

    # Limit to max_stocks * 3 for next stage (we'll narrow further)
    candidates = candidates[:max_stocks * 3]

    print(f"[Quick Filter] {len(candidates)} candidates passed initial screening")

    return {
        "candidates": candidates,
        "filter_count": len(candidates)
    }

def insider_analyzer_node(state: ScreeningState) -> dict:
    """
    Analyze insider trading activity for each candidate.

    Insider buying signals:
    - Multiple executives buying in past 90 days
    - Large purchase amounts (> $100k)
    - CEO/CFO purchases (most significant)

    Scores each stock 0-100 based on insider activity strength.
    """
    candidates = state["candidates"]

    print(f"[Insider Analysis] Analyzing insider trades for {len(candidates)} stocks...")

    insider_scores = []
    for candidate in candidates:
        ticker = candidate["ticker"]

        # Fetch recent insider trades (past 90 days)
        trades = fetch_insider_trades(ticker, days=90)

        # Calculate insider score
        buy_count = sum(1 for t in trades if t["transactionType"] == "P-Purchase")
        buy_amount = sum(t["transactionValue"] for t in trades if t["transactionType"] == "P-Purchase")
        exec_buys = sum(1 for t in trades if t["transactionType"] == "P-Purchase" and
                       t["reportingName"] in ["CEO", "CFO", "President"])

        # Scoring formula
        score = min(100, (buy_count * 10) + (buy_amount / 100_000) + (exec_buys * 20))

        candidate["insider_score"] = score
        candidate["insider_summary"] = f"{buy_count} purchases, ${buy_amount:,.0f} total"

        insider_scores.append(candidate)

    # Sort by insider score
    insider_scores = sorted(insider_scores, key=lambda x: x["insider_score"], reverse=True)

    print(f"[Insider Analysis] Top insider activity: {insider_scores[0]['ticker']} ({insider_scores[0]['insider_score']:.0f}/100)")

    return {
        "candidates": insider_scores
    }

def strategy_scorer_node(state: ScreeningState) -> dict:
    """
    Score stocks against famous investor strategies.

    Strategies:
    1. Warren Buffett: Economic moat, consistent ROE, low debt, understandable business
    2. Peter Lynch: PEG ratio < 1, earnings growth, industry tailwinds
    3. Benjamin Graham: Deep value (P/B < 1.5, P/E < 15, net-net value)

    Uses LOCAL REASONING MODEL (DeepSeek-R1/QwQ-32B) for deep analysis.
    This is where we use the powerful model for stocks that passed screening.
    """
    candidates = state["candidates"]
    criteria = state["criteria"]

    print(f"[Strategy Scoring] Using {REASONING_MODEL} for deep analysis of {len(candidates)} stocks...")
    print(f"[Strategy Scoring] This may take 30-60 seconds per stock with local GPU inference...")

    scored_candidates = []

    for idx, candidate in enumerate(candidates):
        ticker = candidate["ticker"]
        fundamentals = candidate["fundamentals"]

        print(f"[{idx+1}/{len(candidates)}] Analyzing {ticker} with reasoning model...")

        # Build detailed prompt for reasoning model
        prompt = f"""
Analyze {ticker} as a potential investment using {criteria} investment criteria.

**Fundamental Data:**
- P/E Ratio: {fundamentals.get('pe', 'N/A')}
- P/B Ratio: {fundamentals.get('priceToBook', 'N/A')}
- ROE: {fundamentals.get('roe', 0) * 100:.1f}%
- Debt/Equity: {fundamentals.get('debtToEquity', 'N/A')}
- Revenue Growth: {fundamentals.get('revenueGrowth', 0) * 100:.1f}%
- Earnings Growth: {fundamentals.get('earningsGrowth', 0) * 100:.1f}%
- Current Ratio: {fundamentals.get('currentRatio', 'N/A')}
- Market Cap: ${fundamentals.get('marketCap', 0) / 1e9:.2f}B

**Insider Activity:**
{candidate.get('insider_summary', 'No recent insider activity')}

**Task:**
1. Analyze how well this stock aligns with {criteria} principles
2. Identify key strengths and risks
3. Consider valuation, growth prospects, and competitive position
4. Provide a score from 0-100 based on investment merit

**Output Format:**
Score: [0-100 number]
Reasoning: [2-3 sentence explanation]

Think step-by-step and show your reasoning process.
"""

        # Use local reasoning model for deep analysis
        response = generate_reasoning_response(
            prompt=prompt,
            system_prompt=STRATEGY_SCORER_SYSTEM,
            model_choice=REASONING_MODEL,
            max_new_tokens=1024,
            temperature=0.1
        )

        # Extract score from response
        try:
            # Look for "Score: XX" pattern
            score_line = [line for line in response.split('\n') if 'Score:' in line][0]
            strategy_score = int(''.join(filter(str.isdigit, score_line)))
            strategy_score = min(100, max(0, strategy_score))  # Clamp 0-100
        except (IndexError, ValueError):
            # Fallback: try to extract any number
            import re
            numbers = re.findall(r'\b([0-9]{1,3})\b', response)
            strategy_score = int(numbers[0]) if numbers else 50

        candidate["strategy_score"] = strategy_score
        candidate["reasoning_analysis"] = response  # Store full reasoning

        # Combined score: 50% insider + 50% strategy
        candidate["total_score"] = (candidate["insider_score"] * 0.5) + (strategy_score * 0.5)

        scored_candidates.append(candidate)

        print(f"  → Score: {strategy_score}/100 (Combined: {candidate['total_score']:.1f})")

    # Sort by total score
    scored_candidates = sorted(scored_candidates, key=lambda x: x["total_score"], reverse=True)

    # Keep only top N
    max_stocks = state.get("max_stocks", 10)
    final_candidates = scored_candidates[:max_stocks]

    print(f"[Strategy Scoring] Top pick: {final_candidates[0]['ticker']} (Score: {final_candidates[0]['total_score']:.1f}/100)")

    return {
        "final_candidates": final_candidates,
        "screening_complete": True
    }

def portfolio_constructor_node(state: ScreeningState) -> dict:
    """
    Construct final portfolio recommendation.

    Groups stocks by:
    - Sector diversification
    - Risk profile (high growth vs stable value)
    - Position sizing (based on conviction score)

    Returns markdown report with top 10 recommendations.
    """
    candidates = state["final_candidates"]

    prompt = f"""
Create a portfolio recommendation report for these {len(candidates)} stocks:

{[{c['ticker']: c['total_score']} for c in candidates]}

Include:
1. Top 10 ranked recommendations
2. Sector diversification analysis
3. Risk/return profile for each
4. Suggested position sizing (equal-weight vs conviction-weighted)
5. Summary rationale for portfolio construction
"""

    # Use screening model for portfolio construction (summary task, not deep reasoning)
    response = screening_model.invoke([
        {"role": "system", "content": PORTFOLIO_CONSTRUCTOR_SYSTEM},
        {"role": "user", "content": prompt}
    ])

    return {
        "portfolio_report": response.content
    }

# ============================================================================
# GRAPH ASSEMBLY
# ============================================================================

screening_workflow = StateGraph(ScreeningState)

screening_workflow.add_node("universe_builder", universe_builder_node)
screening_workflow.add_node("quick_filter", quick_filter_node)
screening_workflow.add_node("insider_analyzer", insider_analyzer_node)
screening_workflow.add_node("strategy_scorer", strategy_scorer_node)
screening_workflow.add_node("portfolio_constructor", portfolio_constructor_node)

screening_workflow.add_edge(START, "universe_builder")
screening_workflow.add_edge("universe_builder", "quick_filter")
screening_workflow.add_edge("quick_filter", "insider_analyzer")
screening_workflow.add_edge("insider_analyzer", "strategy_scorer")
screening_workflow.add_edge("strategy_scorer", "portfolio_constructor")
screening_workflow.add_edge("portfolio_constructor", END)

screening_app = screening_workflow.compile()
