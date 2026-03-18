"""
Finance Tools - Financial Datasets API
Source: https://financialdatasets.ai/

Financial market and company fundamentals tools for threshold testing.
"""

import json
import logging
import os
from typing import Any
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv


logger = logging.getLogger("financial-datasets-mcp")
load_dotenv()

FINANCIAL_DATASETS_API_BASE = "https://api.financialdatasets.ai"


async def make_request(url: str) -> dict[str, Any] | None:
    """Make a request to the Financial Datasets API with proper error handling."""
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.exception("Financial API request failed: %s", url)
        return {"error": str(e)}


def build_api_url(path: str, **params: object) -> str:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    return f"{FINANCIAL_DATASETS_API_BASE}{path}" + (f"?{query}" if query else "")


def dump_api_items(data: dict[str, Any] | None, field: str, error_message: str) -> str:
    if not data:
        return error_message

    items = data.get(field, [])
    if not items:
        return error_message

    return json.dumps(items, indent=2)


def dump_api_object(data: dict[str, Any] | None, field: str, error_message: str) -> str:
    if not data:
        return error_message

    item = data.get(field, {})
    if not item:
        return error_message

    return json.dumps(item, indent=2)


def register_tools(mcp):
    """Register all finance tools with the MCP server."""

    @mcp.tool()
    async def get_income_statements(
        ticker: str,
        period: str = "annual",
        limit: int = 4,
    ) -> str:
        """Get income statements for a company."""
        url = build_api_url(
            "/financials/income-statements/",
            ticker=ticker,
            period=period,
            limit=limit,
        )
        data = await make_request(url)
        return dump_api_items(
            data,
            "income_statements",
            "Unable to fetch income statements or no income statements found.",
        )

    @mcp.tool()
    async def get_balance_sheets(
        ticker: str,
        period: str = "annual",
        limit: int = 4,
    ) -> str:
        """Get balance sheets for a company."""
        url = build_api_url(
            "/financials/balance-sheets/",
            ticker=ticker,
            period=period,
            limit=limit,
        )
        data = await make_request(url)
        return dump_api_items(
            data,
            "balance_sheets",
            "Unable to fetch balance sheets or no balance sheets found.",
        )

    @mcp.tool()
    async def get_cash_flow_statements(
        ticker: str,
        period: str = "annual",
        limit: int = 4,
    ) -> str:
        """Get cash flow statements for a company."""
        url = build_api_url(
            "/financials/cash-flow-statements/",
            ticker=ticker,
            period=period,
            limit=limit,
        )
        data = await make_request(url)
        return dump_api_items(
            data,
            "cash_flow_statements",
            "Unable to fetch cash flow statements or no cash flow statements found.",
        )

    @mcp.tool()
    async def get_current_stock_price(ticker: str) -> str:
        """Get the current stock price for a company."""
        url = build_api_url("/prices/snapshot/", ticker=ticker)
        data = await make_request(url)
        return dump_api_object(
            data,
            "snapshot",
            "Unable to fetch current price or no current price found.",
        )

    @mcp.tool()
    async def get_historical_stock_prices(
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "day",
        interval_multiplier: int = 1,
    ) -> str:
        """Get historical stock prices for a company."""
        url = build_api_url(
            "/prices/",
            ticker=ticker,
            interval=interval,
            interval_multiplier=interval_multiplier,
            start_date=start_date,
            end_date=end_date,
        )
        data = await make_request(url)
        return dump_api_items(data, "prices", "Unable to fetch prices or no prices found.")

    @mcp.tool()
    async def get_company_news(ticker: str) -> str:
        """Get recent news for a company."""
        url = build_api_url("/news/", ticker=ticker)
        data = await make_request(url)
        return dump_api_items(data, "news", "Unable to fetch news or no news found.")

    @mcp.tool()
    async def get_available_crypto_tickers() -> str:
        """Get all available crypto tickers."""
        url = build_api_url("/crypto/prices/tickers")
        data = await make_request(url)
        return dump_api_items(
            data,
            "tickers",
            "Unable to fetch available crypto tickers or no available crypto tickers found.",
        )

    @mcp.tool()
    async def get_crypto_prices(
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "day",
        interval_multiplier: int = 1,
    ) -> str:
        """Get historical prices for a crypto currency."""
        url = build_api_url(
            "/crypto/prices/",
            ticker=ticker,
            interval=interval,
            interval_multiplier=interval_multiplier,
            start_date=start_date,
            end_date=end_date,
        )
        data = await make_request(url)
        return dump_api_items(data, "prices", "Unable to fetch prices or no prices found.")

    @mcp.tool()
    async def get_historical_crypto_prices(
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "day",
        interval_multiplier: int = 1,
    ) -> str:
        """Get historical prices for a crypto currency."""
        url = build_api_url(
            "/crypto/prices/",
            ticker=ticker,
            interval=interval,
            interval_multiplier=interval_multiplier,
            start_date=start_date,
            end_date=end_date,
        )
        data = await make_request(url)
        return dump_api_items(data, "prices", "Unable to fetch prices or no prices found.")

    @mcp.tool()
    async def get_current_crypto_price(ticker: str) -> str:
        """Get the current price of a crypto currency."""
        url = build_api_url("/crypto/prices/snapshot/", ticker=ticker)
        data = await make_request(url)
        return dump_api_object(
            data,
            "snapshot",
            "Unable to fetch current price or no current price found.",
        )

    @mcp.tool()
    async def get_sec_filings(
        ticker: str,
        limit: int = 10,
        filing_type: str | None = None,
    ) -> str:
        """Get SEC filings for a company."""
        url = build_api_url(
            "/filings/",
            ticker=ticker,
            limit=limit,
            filing_type=filing_type,
        )
        data = await make_request(url)
        return dump_api_items(
            data,
            "filings",
            "Unable to fetch SEC filings or no SEC filings found.",
        )

    @mcp.tool()
    async def getAnalystEstimates(
        ticker: str,
        period: str = "annual",
        limit: int = 4,
    ) -> str:
        """Get analyst consensus estimates for a company."""
        url = build_api_url(
            "/analyst-estimates/",
            ticker=ticker,
            period=period,
            limit=limit,
        )
        data = await make_request(url)
        return dump_api_items(
            data,
            "analyst_estimates",
            "Unable to fetch analyst estimates or no analyst estimates found.",
        )

    @mcp.tool()
    async def getFinancialMetrics(
        ticker: str,
        period: str = "annual",
        limit: int = 4,
    ) -> str:
        """Get historical financial metrics for a company."""
        url = build_api_url(
            "/financial-metrics/",
            ticker=ticker,
            period=period,
            limit=limit,
        )
        data = await make_request(url)
        return dump_api_items(
            data,
            "financial_metrics",
            "Unable to fetch financial metrics or no financial metrics found.",
        )

    @mcp.tool()
    async def getFinancialMetricsSnapshot(ticker: str) -> str:
        """Get the latest financial metrics snapshot for a company."""
        url = build_api_url("/financial-metrics/snapshot/", ticker=ticker)
        data = await make_request(url)
        return dump_api_object(
            data,
            "snapshot",
            "Unable to fetch financial metrics snapshot or no snapshot found.",
        )

    @mcp.tool()
    async def getSegmentedRevenues(
        ticker: str,
        period: str = "annual",
        limit: int = 4,
    ) -> str:
        """Get segmented revenue data for a company."""
        url = build_api_url(
            "/financials/segmented-revenues/",
            ticker=ticker,
            period=period,
            limit=limit,
        )
        data = await make_request(url)
        return dump_api_items(
            data,
            "segmented_revenues",
            "Unable to fetch segmented revenues or no segmented revenues found.",
        )

    @mcp.tool()
    async def getFilingItems(
        ticker: str,
        filing_type: str,
        filing_date: str,
        item: str,
    ) -> str:
        """Extract a specific section from a filing."""
        url = build_api_url(
            "/filings/items/",
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            item=item,
        )
        data = await make_request(url)
        if not data:
            return "Unable to fetch filing items or no filing items found."

        filing_item = data.get("filing_item")
        if filing_item:
            return json.dumps(filing_item, indent=2)

        items = data.get("filing_items", [])
        if items:
            return json.dumps(items, indent=2)

        return "Unable to fetch filing items or no filing items found."

    @mcp.tool()
    async def getAvailableFilingItems(filing_type: str) -> str:
        """Get a list of extractable items for a filing type."""
        url = build_api_url("/filings/items/available/", filing_type=filing_type)
        data = await make_request(url)
        return dump_api_items(
            data,
            "items",
            "Unable to fetch available filing items or no filing items found.",
        )

    @mcp.tool()
    async def getCompanyFacts(ticker: str) -> str:
        """Get company-level facts such as sector, market cap, and exchange."""
        url = build_api_url("/company/facts/", ticker=ticker)
        data = await make_request(url)
        if not data:
            return "Unable to fetch company facts or no company facts found."

        company_facts = data.get("company_facts")
        if company_facts:
            return json.dumps(company_facts, indent=2)

        facts = data.get("facts")
        if facts:
            return json.dumps(facts, indent=2)

        return "Unable to fetch company facts or no company facts found."

    print("Registered 18 Finance Tools")
