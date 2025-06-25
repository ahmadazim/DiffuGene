import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", format_str: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_str: Custom format string for log messages
        
    Returns:
        Configured logger
    """
    if format_str is None:
        format_str = "%(asctime)s [%(levelname)s] %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
