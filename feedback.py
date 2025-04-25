"""
Feedback Module

This module provides elegant logging and progress tracking utilities for optimization algorithms.

Features:
1. SimpleLogger: A clean, customizable logger with color support
2. ProgressManager: Custom tqdm progress bars with vanishing capability for simple tasks
"""
import logging
import sys
import time
from typing import Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum
import os

# Use colorama for cross-platform terminal colors
from colorama import init, Fore, Back, Style
from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm

# Initialize colorama
init(autoreset=True)

class LogLevel(Enum):
    """Log levels with associated colors and prefixes."""
    DEBUG = (logging.DEBUG, Fore.CYAN, "DEBUG")
    INFO = (logging.INFO, Fore.GREEN, "INFO")
    SUCCESS = (25, Fore.GREEN + Style.BRIGHT, "SUCCESS")  # Custom level between INFO and WARNING
    WARNING = (logging.WARNING, Fore.YELLOW, "WARNING")
    ERROR = (logging.ERROR, Fore.RED, "ERROR")
    CRITICAL = (logging.CRITICAL, Fore.RED + Style.BRIGHT, "CRITICAL")
    
    def __init__(self, level_num, color, prefix):
        self.level_num = level_num
        self.color = color
        self.prefix = prefix


class Logger:
    """
    An elegant, simple logger with color support.
    
    Features:
    - Color-coded log levels
    - Optional timestamps
    - Customizable formatting
    - File logging capability
    """
    
    def __init__(self, 
                 name: str = "Logger", 
                 level: LogLevel = LogLevel.INFO, 
                 show_time: bool = True,
                 log_file: Optional[str] = None,
                 file_level: Optional[LogLevel] = None):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            level: Minimum log level to display
            show_time: Whether to include timestamps in log messages
            log_file: Optional path to write logs to a file
            file_level: Log level for file output (defaults to same as console level)
        """
        self.name = name
        self.level = level
        self.show_time = show_time
        self.log_file = log_file
        self.file_level = file_level or level
        
        # Register custom log level SUCCESS
        if not hasattr(logging, 'SUCCESS'):
            logging.SUCCESS = LogLevel.SUCCESS.level_num
            logging.addLevelName(logging.SUCCESS, LogLevel.SUCCESS.prefix)
        
        # Setup Python's logging module
        self._setup()
    
    def _setup(self):
        """Configure the underlying Python logger."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)  # Capture all logs
        self.logger.handlers = []  # Clear any existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level.level_num)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.file_level.level_num)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(file_handler)
    
    def _format(self, level: LogLevel, message: str) -> str:
        """Format a log message with appropriate colors and prefixes."""
        timestamp = f"{Fore.BLUE}[{time.strftime('%Y-%m-%d %H:%M:%S')}]{Style.RESET_ALL} " if self.show_time else ""
        prefix = f"{level.color}[{level.prefix}]{Style.RESET_ALL}"
        return f"{timestamp}{prefix} {message}"
    
    def log(self, level: LogLevel, message: str, *args, **kwargs):
        """Log a message at the specified level."""
        if level.level_num >= self.level.level_num:
            formatted_message = self._format(level, message)
            print(formatted_message, *args, **kwargs)
            
            # Also log to the Python logger for file output
            self.logger.log(level.level_num, message)
    
    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self.log(LogLevel.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self.log(LogLevel.INFO, message, *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        """Log a success message."""
        self.log(LogLevel.SUCCESS, message, *args, **kwargs)
    
    def warn(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self.log(LogLevel.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self.log(LogLevel.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log a critical message."""
        self.log(LogLevel.CRITICAL, message, *args, **kwargs)


@dataclass
class ProgressConfig:
    """Configuration options for progress bars."""
    desc: str = ""
    total: Optional[int] = None
    leave: bool = True  # Whether to leave the progress bar after completion
    is_main: bool = False  # Whether this is a main progress bar (custom param)
    color: Optional[str] = None  # Color for the progress bar (custom param)
    unit: str = "it"  # Unit name
    position: Optional[int] = None  # Position in case of nested bars
    file: Optional[Any] = None
    dynamic_ncols: bool = True  # Adapt to terminal width changes
    smoothing: float = 0.3  # Smoothing factor for ETA calculation
    bar_format: Optional[str] = None  # Custom bar format
    initial: int = 0  # Initial counter value
    disable: bool = False  # Whether to disable the progress bar
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for tqdm initialization."""
        # Only include parameters that tqdm recognizes
        tqdm_params = {
            "desc": self.desc,
            "total": self.total,
            "leave": self.leave,
            "unit": self.unit,
            "position": self.position,
            "file": self.file,
            "dynamic_ncols": self.dynamic_ncols,
            "smoothing": self.smoothing,
            "bar_format": self.bar_format,
            "initial": self.initial,
            "disable": self.disable
        }
        
        # Include color if specified (convert to tqdm's color parameter)
        if self.color:
            tqdm_params["colour"] = self.color
            
        # Remove None values to use tqdm defaults
        return {k: v for k, v in tqdm_params.items() if v is not None}


class Progress:
    """
    Manages customized tqdm progress bars.
    
    Features:
    - Main progress bars remain visible after completion
    - Simple progress bars vanish after completion
    - Consistent styling and color support
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        self.bars = []
        self.logger = logger or Logger()
    
    def bar(self, config: Optional[ProgressConfig] = None, **kwargs) -> Any:
        """
        Create a new progress bar with the specified configuration.
        
        Args:
            config: Progress bar configuration
            **kwargs: Additional parameters to pass to tqdm
            
        Returns:
            A tqdm progress bar instance
        """
        # Use provided config or create a default one
        cfg = config or ProgressConfig()
        
        # Update with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        
        # Simple progress bars (non-main) shouldn't be left after completion
        if not cfg.is_main and cfg.leave is True:
            cfg.leave = False
        
        # Get a clean dictionary of parameters for tqdm
        tqdm_params = cfg.as_dict()
        
        # Create and return the progress bar
        progress_bar = auto_tqdm(**tqdm_params)
        self.bars.append(progress_bar)
        return progress_bar
    
    def _create_config(self, desc: str, total: int, **kwargs) -> ProgressConfig:
        """Create a base progress config with common settings."""
        is_main = kwargs.pop('is_main', False)
        return ProgressConfig(
            desc=desc,
            total=total,
            leave=is_main,  # Only leave the bar if it's a main bar
            is_main=is_main,
            **kwargs
        )
    
    def main(self, desc: str, total: int, **kwargs) -> Any:
        """
        Create a main progress bar that will remain visible after completion.
        
        Args:
            desc: Description of the progress
            total: Total number of iterations
            **kwargs: Additional parameters to pass to tqdm
            
        Returns:
            A tqdm progress bar instance
        """
        # Default settings for main progress bars
        if 'color' not in kwargs:
            kwargs['color'] = 'green'
        
        kwargs['is_main'] = True
        config = self._create_config(desc, total, **kwargs)
        return self.bar(config)
    
    def simple(self, desc: str, total: int, **kwargs) -> Any:
        """
        Create a simple progress bar that will vanish after completion.
        
        Args:
            desc: Description of the progress
            total: Total number of iterations
            **kwargs: Additional parameters to pass to tqdm
            
        Returns:
            A tqdm progress bar instance
        """
        kwargs['is_main'] = False
        config = self._create_config(desc, total, **kwargs)
        return self.bar(config)
    
    def close(self):
        """Close all active progress bars."""
        for bar in self.bars:
            bar.close()
        self.bars = []


# Create global instances for convenience
log = Logger()
progress = Progress(log)


# Example usage
if __name__ == "__main__":
    # Logger examples
    logger = Logger("Example")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warn("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Progress bar examples
    import time
    
    # Main progress bar (will remain visible)
    with progress.main("Main Progress", 3) as main_pbar:
        for i in range(3):
            # Simple progress bar (will vanish)
            with progress.simple(f"Subtask {i+1}", 100) as pbar:
                for j in range(100):
                    time.sleep(0.01)
                    pbar.update(1)
            main_pbar.update(1)
    
    logger.success("All tasks completed successfully!") 