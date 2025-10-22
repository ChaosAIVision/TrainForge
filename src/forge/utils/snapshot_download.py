from typing import List, Dict , Optional
from hugggingface_hub import snapshot_download


def download_model(model_name: str, cache_dir: Optional[str] = None) -> str:
    """
    Download a model from the Hugging Face Hub.

    Args:
        model_name (str): The name of the model to download.
        cache_dir (Optional[str], optional): The directory to cache the downloaded model. Defaults to None.

    Returns:
        str: The path to the downloaded model.
    """
    return snapshot_download(model_name, cache_dir=cache_dir)


