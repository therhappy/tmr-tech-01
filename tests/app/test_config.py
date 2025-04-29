"""
Tests for the configuration module.
"""

import os
import pytest
from pathlib import Path
from app.config import (
    load_config, 
    get_config, 
    get_api_host, 
    get_api_port, 
    get_log_level,
)


@pytest.fixture
def temp_env_config(tmp_path):
    """Create a temporary .env_config file for testing."""
    config_file = tmp_path / ".env_config"
    with open(config_file, "w") as f:
        f.write("TEST_VAR=test_value\n")
        f.write("API_HOST=test.host\n")
        f.write("API_PORT=9999\n")
        f.write("LLM_MODEL_NAME=test-model\n")
        f.write("LLM_MODEL_PATH=/path/to/model\n")
        f.write("CONVERSATION_STORAGE_PATH=/path/to/storage\n")
        f.write("LOG_LEVEL=DEBUG\n")
        f.write("PINECONE_API_KEY=test-pinecone-key\n")
        f.write("OPENAI_PROJECT=test-project\n")
        f.write("OPENAI_API_KEY=test-openai-key\n")
        f.write("# This is a comment\n")
        f.write("\n")  # Empty line
    return str(config_file)


def test_load_config(temp_env_config):
    """Test that configuration is loaded from a file."""
    # Clear any existing TEST_VAR to ensure test is reliable
    if "TEST_VAR" in os.environ:
        del os.environ["TEST_VAR"]
    
    load_config(temp_env_config)
    assert os.environ.get("TEST_VAR") == "test_value"


def test_get_config():
    """Test getting configuration values."""
    os.environ["TEST_KEY"] = "test_value"
    assert get_config("TEST_KEY") == "test_value"
    assert get_config("NON_EXISTENT_KEY", "default") == "default"


def test_config_accessor_functions():
    """Test the config accessor functions."""
    # Set environment variables for testing
    os.environ["API_HOST"] = "test.host"
    os.environ["API_PORT"] = "9999"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    assert get_api_host() == "test.host"
    assert get_api_port() == 9999
    assert get_log_level() == "DEBUG"


def test_config_accessor_defaults():
    """Test the default values for config accessor functions."""
    # Remove environment variables to test defaults
    env_vars = [
        "API_HOST", "API_PORT", "LLM_MODEL_NAME", "LLM_MODEL_PATH",
        "CONVERSATION_STORAGE_PATH", "LOG_LEVEL", "PINECONE_API_KEY",
        "OPENAI_PROJECT", "OPENAI_API_KEY"
    ]
    
    # Save original values
    original_values = {}
    for var in env_vars:
        if var in os.environ:
            original_values[var] = os.environ[var]
            del os.environ[var]
    
    try:
        # Test default values
        assert get_api_host() == "0.0.0.0"
        assert get_api_port() == 8000
        assert get_log_level() == "INFO"

    finally:
        # Restore original values
        for var, value in original_values.items():
            os.environ[var] = value

def test_load_config_file_not_found():
    """Test behavior when config file is not found."""
    non_existent_file = "/path/that/does/not/exist/.env_config"
    # This should not raise an exception but log a warning
    load_config(non_existent_file)
    # Since we can't easily check log messages, we'll just verify the function completes
    assert True

def test_load_config_without_param():
    """Test loading config without specifying a file path."""
    # Save original project root config if exists
    original_config = None
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / '.env_config'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            original_config = f.read()
    
    # Create a temporary config in the project root
    try:
        with open(config_path, 'w') as f:
            f.write("TEST_DEFAULT_LOAD=success\n")
        
        # Clear the variable if it exists
        if "TEST_DEFAULT_LOAD" in os.environ:
            del os.environ["TEST_DEFAULT_LOAD"]
        
        # Load config without parameter
        load_config()
        
        # Check if variable was loaded
        assert os.environ.get("TEST_DEFAULT_LOAD") == "success"
    finally:
        # Clean up: restore original or remove temporary file
        if original_config is not None:
            with open(config_path, 'w') as f:
                f.write(original_config)
        else:
            if config_path.exists():
                config_path.unlink()