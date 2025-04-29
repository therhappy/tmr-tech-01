"""
Application package initialization.
Ensures configuration is loaded from .env_config before any submodules.
"""

import logging

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import and load configuration
from app.config import load_config, get_log_level

# Load environment variables from .env_config
load_config()

# Update logging level from configuration
logging_level = get_log_level()
logging.getLogger().setLevel(logging_level)
logger.info(f"Application initialized with logging level: {logging_level}")