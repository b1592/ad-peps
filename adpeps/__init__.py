import os
from pathlib import Path

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = Path(ROOT_DIR).parent
