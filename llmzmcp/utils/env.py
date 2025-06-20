import os
from pathlib import Path

from dotenv import load_dotenv

from llmzmcp.utils.paths import get_repo_dir

cwd = get_repo_dir()
env_path = f"{cwd}/.env"
load_dotenv(dotenv_path=f"{env_path}")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")