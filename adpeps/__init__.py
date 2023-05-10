import os
from importlib.metadata import version
from pathlib import Path

__version__ = version("ad-peps")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = Path(ROOT_DIR).parent
