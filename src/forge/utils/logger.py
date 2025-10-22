"""
Production-style Logger with colored output for TrainForge.

This module provides a comprehensive logging solution with colored console output
and optional file logging capabilities.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Get the original formatted message
        formatted_message = super().format(record)
        
        # Add color based on log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        return f"{color}{formatted_message}{reset}"


class Logger:
    """
    Production-style Logger with colored output.
    
    Features:
    - Colored console output (INFO: green, WARNING: yellow, ERROR: red)
    - Optional file logging
    - Configurable log levels
    - Thread-safe logging
    - Proper formatting with timestamps
    """
    
    def __init__(
        self,
        name: str = "TrainForge",
        level: Union[str, int] = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        console_output: bool = True,
        file_output: bool = False
    ):
        """
        Initialize the Logger.
        
        Args:
            name (str): Logger name
            level (Union[str, int]): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file (Optional[Union[str, Path]]): Path to log file
            console_output (bool): Enable console output
            file_output (bool): Enable file output
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_level(level))
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Set up console handler with colors
        if console_output:
            self._setup_console_handler()
        
        # Set up file handler if requested
        if file_output and log_file:
            self._setup_file_handler(log_file)
    
    def _get_level(self, level: Union[str, int]) -> int:
        """Convert string level to logging level constant."""
        if isinstance(level, str):
            return getattr(logging, level.upper(), logging.INFO)
        return level
    
    def _setup_console_handler(self) -> None:
        """Set up colored console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: Union[str, Path]) -> None:
        """Set up file handler without colors."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message (green color)."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message (yellow color)."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message (red color)."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)
    
    def set_level(self, level: Union[str, int]) -> None:
        """Change the logging level."""
        self.logger.setLevel(self._get_level(level))
    
    def add_file_handler(self, log_file: Union[str, Path]) -> None:
        """Add a file handler to the logger."""
        self._setup_file_handler(log_file)
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger


# Convenience function to create a default logger instance
def get_logger(
    name: str = "TrainForge",
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    file_output: bool = False
) -> Logger:
    """
    Create and return a Logger instance.
    
    Args:
        name (str): Logger name
        level (Union[str, int]): Logging level
        log_file (Optional[Union[str, Path]]): Path to log file
        console_output (bool): Enable console output
        file_output (bool): Enable file output
    
    Returns:
        Logger: Configured logger instance
    """
    return Logger(
        name=name,
        level=level,
        log_file=log_file,
        console_output=console_output,
        file_output=file_output
    )


# Create a default logger instance for immediate use
default_logger = get_logger()