"""
Tests for the download_model module.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from app.deployment.download_model import download_model, main


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer."""
    with patch('app.deployment.download_model.SentenceTransformer') as mock_st:
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance
        yield mock_st, mock_instance


@pytest.fixture
def mock_os_functions():
    """Mock OS functions used in download_model."""
    with patch('app.deployment.download_model.os.path.exists') as mock_exists, \
         patch('app.deployment.download_model.os.makedirs') as mock_makedirs, \
         patch('app.deployment.download_model.os.path.dirname') as mock_dirname, \
         patch('app.deployment.download_model.os.chdir') as mock_chdir:
        
        # Mock dirname to return predictable paths
        mock_dirname.side_effect = lambda path: f"{path}_parent"
        
        yield {
            'exists': mock_exists,
            'makedirs': mock_makedirs,
            'dirname': mock_dirname,
            'chdir': mock_chdir
        }


def test_download_model_directory_exists(mock_sentence_transformer, mock_os_functions):
    """Test download_model when the target directory already exists."""
    # Configure mock
    mock_st, mock_model = mock_sentence_transformer
    mock_os_functions['exists'].return_value = True
    
    # Call function
    download_model("./test_model_path", "test_model_name")
    
    # Verify directory check but not creation
    mock_os_functions['exists'].assert_called_once_with("./test_model_path")
    mock_os_functions['makedirs'].assert_not_called()
    
    # Verify model loading and saving
    mock_st.assert_called_once_with("test_model_name")
    mock_model.save_pretrained.assert_called_once_with("./test_model_path")
    
    # Verify directory change to app root
    assert mock_os_functions['chdir'].called


def test_download_model_directory_not_exists(mock_sentence_transformer, mock_os_functions):
    """Test download_model when the target directory does not exist."""
    # Configure mock
    mock_st, mock_model = mock_sentence_transformer
    mock_os_functions['exists'].return_value = False
    
    # Call function
    download_model("./test_model_path", "test_model_name")
    
    # Verify directory check and creation
    mock_os_functions['exists'].assert_called_once_with("./test_model_path")
    mock_os_functions['makedirs'].assert_called_once_with("./test_model_path")
    
    # Verify model loading and saving
    mock_st.assert_called_once_with("test_model_name")
    mock_model.save_pretrained.assert_called_once_with("./test_model_path")


def test_download_model_changes_directory(mock_sentence_transformer, mock_os_functions):
    """Test that download_model changes to the app root directory."""
    # Configure mock
    mock_os_functions['exists'].return_value = True
    
    # Mock abspath to return a predictable path
    with patch('app.deployment.download_model.os.path.abspath') as mock_abspath:
        mock_abspath.return_value = "/mock/path/to/file"
        
        # Call function
        download_model("./test_model_path", "test_model_name")
        
        # Verify directory change logic
        assert mock_os_functions['dirname'].call_count == 3
        # We expect to navigate up two levels from the script location
        assert mock_os_functions['chdir'].called


def test_main_with_default_config():
    """Test main function when using default config values."""
    with patch('app.deployment.download_model.get_config') as mock_get_config, \
         patch('app.deployment.download_model.download_model') as mock_download_model, \
         patch('app.deployment.download_model.print') as mock_print:
        
        # Mock get_config to return default values
        mock_get_config.side_effect = lambda key, default: default
        
        # Call main function
        main()
        
        # Verify config was requested with correct parameters
        mock_get_config.assert_any_call("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        mock_get_config.assert_any_call("EMBEDDING_MODEL_PATH", "./models")
        
        # Verify download_model was called with correct parameters
        mock_download_model.assert_called_once_with("./models", "BAAI/bge-m3")
        
        # Verify print statements
        assert mock_print.call_count == 2


def test_main_with_custom_config():
    """Test main function when using custom config values."""
    with patch('app.deployment.download_model.get_config') as mock_get_config, \
         patch('app.deployment.download_model.download_model') as mock_download_model, \
         patch('app.deployment.download_model.print') as mock_print:
        
        # Mock get_config to return custom values
        mock_get_config.side_effect = lambda key, default: {
            "EMBEDDING_MODEL_NAME": "custom/model", 
            "EMBEDDING_MODEL_PATH": "/custom/path"
        }.get(key, default)
        
        # Call main function
        main()
        
        # Verify config was requested with correct parameters
        mock_get_config.assert_any_call("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        mock_get_config.assert_any_call("EMBEDDING_MODEL_PATH", "./models")
        
        # Verify download_model was called with custom parameters
        mock_download_model.assert_called_once_with("/custom/path", "custom/model")
        
        # Verify print statements
        mock_print.assert_any_call("Downloading model custom/model to /custom/path...")
        mock_print.assert_any_call("Done")
