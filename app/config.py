"""
Configuration utility for loading environment variables from .env_config file.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_file=None):
    """
    Load configuration variables from .env_config file into environment variables.
    
    Args:
        config_file (str, optional): Path to the config file. Defaults to None,
                                     which will use the .env_config in the project root.
    """
    if config_file is None:
        # Try to find the .env_config file in the project root
        project_root = Path(__file__).parent.parent
        config_file = project_root / '.env_config'
    
    if not os.path.exists(config_file):
        logger.warning(f"Config file {config_file} not found. Using default environment variables.")
        return
    
    logger.info(f"Loading configuration from {config_file}")
    
    # Read the .env_config file and set environment variables
    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse key-value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Don't override existing environment variables
                if key not in os.environ:
                    os.environ[key] = value
                    logger.debug(f"Set environment variable: {key}")

def get_config(key, default=None):
    """
    Get a configuration value from environment variables.
    
    Args:
        key (str): The configuration key to retrieve.
        default (any, optional): Default value if key is not found. Defaults to None.
        
    Returns:
        str: The configuration value or default if not found.
    """
    return os.environ.get(key, default)

# Config accessor functions for commonly used values
def get_api_host():
    return get_config('API_HOST', '0.0.0.0')

def get_api_port():
    return int(get_config('API_PORT', '8000'))

def get_log_level():
    return get_config('LOG_LEVEL', 'INFO')

def is_eval_mode():
    """
    Check if the application is running in test mode.
    
    Returns:
        bool: True if test mode is enabled, False otherwise.
    """
    test_mode = get_config('EVAL_MODE', 'false').lower()
    return test_mode == 'true'
