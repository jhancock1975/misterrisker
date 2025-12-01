"""Exceptions for the Researcher MCP Server."""


class ResearcherAPIError(Exception):
    """Exception raised for API errors in the Researcher MCP Server.
    
    Attributes:
        message: Error message describing what went wrong
        tool_name: Name of the tool that caused the error (if applicable)
        api_source: The API that caused the error (finnhub, openai, etc.)
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        api_source: str | None = None
    ):
        """Initialize ResearcherAPIError.
        
        Args:
            message: Error message
            tool_name: Name of the tool that caused the error
            api_source: The API source (finnhub, openai, etc.)
        """
        self.message = message
        self.tool_name = tool_name
        self.api_source = api_source
        super().__init__(self.message)


class ResearcherConfigError(Exception):
    """Exception raised for configuration errors.
    
    Attributes:
        message: Error message describing the configuration issue
        missing_key: The missing configuration key (if applicable)
    """
    
    def __init__(self, message: str, missing_key: str | None = None):
        """Initialize ResearcherConfigError.
        
        Args:
            message: Error message
            missing_key: The missing configuration key
        """
        self.message = message
        self.missing_key = missing_key
        super().__init__(self.message)
