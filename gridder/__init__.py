"""
Gridder - A Python package for geological data gridding and interpolation.

This module provides tools for gridding geological data points onto regular grids,
with support for various interpolation methods and geological inpainting techniques.
"""

import os
import sys

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import main classes
try:
    from .gridder import Geo_Gridder
except ImportError:
    # Fallback for cases where relative import fails
    from gridder import Geo_Gridder

# Package metadata
__version__ = "1.0.0"
__author__ = "Marios Karaoulis"
__email__ = "marios@example.com"

# Define what gets imported with "from gridder import *"
__all__ = ['Geo_Gridder']