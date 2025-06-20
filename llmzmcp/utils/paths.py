import os
import shutil
from pathlib import Path


def clear_pycache():
    """
    Remove all __pycache__ folders for cleanup before push

    Returns:
        None
    """
    file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    proj_dir = file_dir.parents[0]
    for root, dirs, files in os.walk(proj_dir, topdown=False):
        if "__pycache__" in root:
            shutil.rmtree(root)
    return None

def get_repo_dir():
    """
    Get the repo root directory
    """
    path = Path(os.path.realpath(__file__)).parents[2]
    return path