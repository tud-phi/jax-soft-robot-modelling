import importlib.metadata

from .integration import *
from .symbolic_derivation import *
from .systems import *

__version__ = importlib.metadata.version("equinox")
