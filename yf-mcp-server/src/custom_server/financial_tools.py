#########################################################################
### MCP YAHOO FINANCE TOOLS DEFINITION
#########################################################################
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define custom classes
class StockInfo(BaseModel):
    """Stock information model"""

    symbol: str
    name: str = ""
    current_price: float = 0.0
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None


class HistoricalDataRequest(BaseModel):
    """Request model for historical data"""

    symbol: str
    period: str = Field(
        default="1mo", description="Period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"
    )
    interval: str = Field(
        default="1d",
        description="Interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo",
    )

# Wrap tools into this following function
def register_finance_tools(mcp):
    """ Register my custom MCP tools """

    @mcp.tool()
    async def get_stock_info(symbol: str) -> Dict[str, Any]:
        """
        Get basic stock information including current price, market cap, and key metrics.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

        Returns:
            Dictionary containing stock information
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info

            return {
                "symbol": symbol.upper(),
                "name": info.get("longName", ""),
                "current_price": info.get("currentPrice", 0.0),
                "previous_close": info.get("previousClose"),
                "open": info.get("open"),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "market_cap": info.get("marketCap"),
                # P/E Ratios
                "trailing_pe": info.get("trailingPE"),  # P/E based on past 12 months
                "forward_pe": info.get("forwardPE"),  # P/E based on estimated earnings
                "peg_ratio": info.get("pegRatio"),  # PEG ratio (PE to growth)
                # Valuation Metrics
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "price_to_book": info.get("priceToBook"),
                "enterprise_value": info.get("enterpriseValue"),
                "enterprise_to_revenue": info.get("enterpriseToRevenue"),
                "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
                # Profitability Metrics
                "profit_margins": info.get("profitMargins"),
                "operating_margins": info.get("operatingMargins"),
                "gross_margins": info.get("grossMargins"),
                "ebitda_margins": info.get("ebitdaMargins"),
                # Earnings & Returns
                "earnings_per_share": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
                "return_on_assets": info.get("returnOnAssets"),
                "return_on_equity": info.get("returnOnEquity"),
                # Dividend Information
                "dividend_yield": info.get("dividendYield"),
                "dividend_rate": info.get("dividendRate"),
                "payout_ratio": info.get("payoutRatio"),
                "ex_dividend_date": info.get("exDividendDate"),
                # Financial Health
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "free_cashflow": info.get("freeCashflow"),
                "operating_cashflow": info.get("operatingCashflow"),
                # Growth Metrics
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "revenue_per_share": info.get("revenuePerShare"),
                "book_value": info.get("bookValue"),
                # Trading Metrics
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "52_week_change": info.get("52WeekChange"),
                "volume": info.get("volume"),
                "avg_volume": info.get("averageVolume"),
                "avg_volume_10days": info.get("averageVolume10days"),
                "beta": info.get("beta"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "float_shares": info.get("floatShares"),
                "shares_short": info.get("sharesShort"),
                "short_ratio": info.get("shortRatio"),
                "short_percent_of_float": info.get("shortPercentOfFloat"),
                # Analyst Metrics
                "target_high_price": info.get("targetHighPrice"),
                "target_low_price": info.get("targetLowPrice"),
                "target_mean_price": info.get("targetMeanPrice"),
                "target_median_price": info.get("targetMedianPrice"),
                "recommendation_mean": info.get("recommendationMean"),
                "recommendation_key": info.get("recommendationKey"),
                "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
                # Company Information
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "website": info.get("website"),
                "full_time_employees": info.get("fullTimeEmployees"),
                "business_summary": (
                    info.get("businessSummary", "")[:500] + "..."
                    if info.get("businessSummary", "")
                    else ""
                ),
            }
        except Exception as e:
            logger.error("Error getting stock info for %s: %s", symbol, str(e))
            return {"error": f"Failed to get stock info for {symbol}: {str(e)}"}
    
    @mcp.tool()
    async def get_historical_data(
        symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Get historical stock price data.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
            interval: Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)

        Returns:
            Dictionary containing historical price data
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                return {"error": f"No data found for symbol {symbol}"}

            # Convert DataFrame to dictionary format
            data = []
            for date, row in hist.iterrows():
                data.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]) if "Volume" in row else 0,
                    }
                )

            return {
                "symbol": symbol.upper(),
                "period": period,
                "interval": interval,
                "data": data,
                "count": len(data),
            }
        except Exception as e:
            logger.error("Error getting historical data for %s: %s", symbol, str(e))
            return {"error": f"Failed to get historical data for {symbol}: {str(e)}"}
    
    @mcp.tool()
    async def get_news(symbol: str, count: int = 10) -> Dict[str, Any]:
        """
        Get recent news for a stock.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            count: Number of news articles to return (default: 10)

        Returns:
            Dictionary containing news articles
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            news = ticker.news

            if not news:
                return {
                    "symbol": symbol.upper(),
                    "news": [],
                    "message": "No news available",
                }

            # Limit the number of articles
            news = news[:count]

            news_data = []
            for article in news:
                content = article.get("content", {})
                thumbnail_url = ""
                if content.get("thumbnail") and content["thumbnail"].get("resolutions"):
                    thumbnail_url = content["thumbnail"]["resolutions"][0].get("url", "")

                # Convert pubDate to timestamp if available
                pub_time = 0
                if content.get("pubDate"):
                    try:
                        pub_time = int(
                            datetime.fromisoformat(
                                content["pubDate"].replace("Z", "+00:00")
                            ).timestamp()
                        )
                    except (ValueError, KeyError, AttributeError):
                        pub_time = 0

                news_data.append(
                    {
                        "title": content.get("title", ""),
                        "link": content.get("canonicalUrl", {}).get("url", ""),
                        "publisher": content.get("provider", {}).get("displayName", ""),
                        "providerPublishTime": pub_time,
                        "type": content.get("contentType", ""),
                        "thumbnail": thumbnail_url,
                        "summary": content.get("summary", ""),
                    }
                )

            return {"symbol": symbol.upper(), "news": news_data, "count": len(news_data)}
        except Exception as e:
            logger.error("Error getting news for %s: %s", symbol, str(e))
            return {"error": f"Failed to get news for {symbol}: {str(e)}"}
        
    @mcp.tool()
    async def search_stocks(query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for stocks by company name or ticker symbol.

        This tool searches Yahoo Finance's database for stocks matching your query.
        Works best with specific company names or partial ticker symbols.

        Examples of effective queries:
        - "Microsoft" (company name)
        - "AAPL" (ticker symbol)
        - "Tesla" (company name)
        - "JPM" (partial ticker)

        Note: Complex multi-word queries may return fewer results. For best results,
        search for one company at a time.

        Args:
            query: Search query - company name or ticker symbol (e.g., 'Microsoft', 'AAPL')
            limit: Maximum number of results to return (default: 10, max recommended: 25)

        Returns:
            Dictionary containing search results with symbol, name, type, exchange,
            sector, industry, relevance score, and other metadata
        """
        try:
            # Use yfinance Search class (updated API)
            search_obj = yf.Search(query, max_results=limit)
            search_results = search_obj.quotes

            if not search_results:
                return {"query": query, "results": [], "message": "No results found"}

            results = []
            for result in search_results[:limit]:  # Ensure we don't exceed limit
                results.append(
                    {
                        "symbol": result.get("symbol", ""),
                        "name": result.get("longname", result.get("shortname", "")),
                        "type": result.get("quoteType", ""),
                        "exchange": result.get("exchange", ""),
                        "sector": result.get("sector", ""),
                        "industry": result.get("industry", ""),
                        "score": result.get("score", 0),
                        "is_yahoo_finance": result.get("isYahooFinance", False),
                    }
                )

            return {"query": query, "results": results, "count": len(results)}
        except Exception as e:
            logger.error("Error searching stocks for query '%s': %s", query, str(e))
            return {"error": f"Failed to search stocks for query '{query}': {str(e)}"}
