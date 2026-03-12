import logging
import sys
from app.core.config import get_settings

settings = get_settings()

class ColoredFormatter(logging.Formatter):
    """Custom logging formatter that adds colors based on log levels and logger names."""
    
    # ANSI escape codes for colors
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    CYAN = "\x1b[36;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + RESET,
        logging.INFO: BLUE + "%(asctime)s" + RESET + " - " + CYAN + "%(name)s" + RESET + " - " + GREEN + "%(levelname)s" + RESET + " - %(message)s",
        logging.WARNING: YELLOW + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + RESET,
        logging.ERROR: RED + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        
        # Special coloring for Agent logs to make them stand out
        if "app.agents" in record.name:
            log_fmt = log_fmt.replace("%(name)s", "\x1b[35;1m" + record.name + self.RESET) # Magenta Bold
        
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())
    
    # Configure root logger
    logging.root.setLevel(settings.LOG_LEVEL)
    logging.root.handlers = [handler]
    
    # Suppress verbose loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce LLM call noise
