## Boilerplate Tools
def register_boilerplate_tools(mcp):
    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b