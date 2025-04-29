"""
Submodule to download a HuggingFace-hosted model to local storage
"""

from sentence_transformers import SentenceTransformer
import os
try:
    # When imported as part of the package
    from app.config import get_config
except ImportError:
    # When run as a script
    import sys
    import pathlib
    # Add the parent directory to sys.path
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
    from app.config import get_config


def download_model(model_path, model_name):
    """Download a Hugging Face model and tokenizer to the specified directory"""
    
    # Go at app root level
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory of the current directory
    parent_dir = os.path.dirname(current_dir)
    # Get the parent directory of the parent directory
    parent_of_parent_dir = os.path.dirname(parent_dir)
    # Change the working directory
    os.chdir(parent_of_parent_dir)
    
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)
    
    model = SentenceTransformer(model_name)
    model.save_pretrained(model_path)


def main() -> None:
     # Load the model name & path from the config
    model_name = get_config("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    model_path = get_config("EMBEDDING_MODEL_PATH", "./models")
    
    print("Downloading model {} to {}...".format(model_name, model_path))
    
    # Download the model
    download_model(model_path, model_name)
    
    print("Done")


if __name__ == "__main__":
   main()
