#####################################################################################################################
### MCP FINANCIAL DATASETS TOOLS DEFINITION
#####################################################################################################################
import json
import os
import httpx
import logging
import sys
from dotenv import load_dotenv
## Load Enviornment variables
#os.chdir('/Workspace/Users/`gabriele.albini@databricks.com`/mcp-demo')

load_dotenv()
my_API_key = os.getenv('my_API_key')
FINANCIAL_DATASETS_API_BASE = "https://api.financialdatasets.ai"

# Helper function to make API requests
async def make_request(url: str) -> dict[str, any] | None:
    """Make a request to the Financial Datasets API with proper error handling."""
    # Load environment variables from .env file
    #load_dotenv()
    
    headers = {}
    """
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key
    """
    api_key = my_API_key

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=45.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"Error": str(e)}
          
def register_finance_tools(mcp):
    """ Register my custom MCP tools """

    # Tool to retrieve live stock prices
    @mcp.tool()
    async def get_current_stock_price(ticker: str) -> str:
        """Get the current / latest price of a company.

        Args:
            ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        """
        # Fetch data from the API
        url = f"{FINANCIAL_DATASETS_API_BASE}/prices/snapshot/?ticker={ticker}"
        data = await make_request(url)

        # Check if data is found
        if not data:
            return "Unable to fetch current price or no current price found."

        # Extract the current price
        snapshot = data.get("snapshot", {})

        # Check if current price is found
        if not snapshot:
            return "Unable to fetch current price or no current price found."

        # Stringify the current price
        return json.dumps(snapshot, indent=2)

    # Tool to retreive stock prices for a defined period
    @mcp.tool()
    async def get_historical_stock_prices(
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "day",
        interval_multiplier: int = 1,
    ) -> str:
        """Gets historical stock prices for a company.

        Args:
            ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
            start_date: Start date of the price data (e.g. 2020-01-01)
            end_date: End date of the price data (e.g. 2020-12-31)
            interval: Interval of the price data (e.g. minute, hour, day, week, month)
            interval_multiplier: Multiplier of the interval (e.g. 1, 2, 3)
        """
        # Fetch data from the API
        url = f"{FINANCIAL_DATASETS_API_BASE}/prices/?ticker={ticker}&interval={interval}&interval_multiplier={interval_multiplier}&start_date={start_date}&end_date={end_date}"
        data = await make_request(url)

        # Check if data is found
        if not data:
            return "Unable to fetch prices or no prices found."

        # Extract the prices
        prices = data.get("prices", [])

        # Check if prices are found
        if not prices:
            return "Unable to fetch prices or no prices found."

        # Stringify the prices
        return json.dumps(prices, indent=2)