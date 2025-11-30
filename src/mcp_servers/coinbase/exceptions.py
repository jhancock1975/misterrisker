"""Custom exceptions for Coinbase MCP Server."""


class CoinbaseAPIError(Exception):
    """Exception raised for Coinbase API errors.
    
    Attributes:
        message: Error message describing what went wrong
        status_code: HTTP status code from the API response (optional)
        error_code: Coinbase-specific error code (optional)
    """
    
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None
    ):
        """Initialize CoinbaseAPIError.
        
        Args:
            message: Error message
            status_code: HTTP status code
            error_code: Coinbase-specific error code
        """
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        parts = [self.message]
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        return " | ".join(parts)


class CoinbaseAuthError(CoinbaseAPIError):
    """Exception raised for Coinbase authentication errors.
    
    This is a specific type of API error that indicates
    authentication or authorization failures.
    """
    
    def __init__(
        self,
        message: str = "Authentication failed",
        status_code: int | None = 401,
        error_code: str | None = "AUTH_ERROR"
    ):
        """Initialize CoinbaseAuthError.
        
        Args:
            message: Error message (default: "Authentication failed")
            status_code: HTTP status code (default: 401)
            error_code: Error code (default: "AUTH_ERROR")
        """
        super().__init__(message, status_code, error_code)
