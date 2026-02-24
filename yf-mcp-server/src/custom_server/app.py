from pathlib import Path
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI
from fastapi.responses import FileResponse


#####################################################################################################################
### MCP Config
#####################################################################################################################

# Create an MCP server
mcp = FastMCP("Custom YahooFinance MCP Server on Databricks Apps")


# Add financial module
from custom_server.financial_tools import register_finance_tools
register_finance_tools(mcp)


# Add boilerplate tools
# from custom_server.boilerplate_tools import register_boilerplate_tools
# register_boilerplate_tools(mcp)

#####################################################################################################################
### WEBAPP Config
#####################################################################################################################
STATIC_DIR = Path(__file__).parent / "static"


# Create a ASGI web app that exposes the MCP endpoint
mcp_app = mcp.streamable_http_app()


# Create a FastAPI application `app`
app = FastAPI(
    lifespan=lambda _: mcp.session_manager.run(),
)


# Create a front end route for the app to handle http get requests
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# Mount the mcp server to the FastAPI web app, as a sub-application at the root path
app.mount("/", mcp_app)