"""
VtkClass - A Python package for geological VTK data processing and visualization.

This module provides tools for working with VTK files in geological applications,
including 3D grid generation, data interpolation, and visualization.
"""

import os
import sys

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import main classes
try:
    from .VtkClass import VtkClass
except ImportError:
    # Fallback for cases where relative import fails
    from VtkClass import VtkClass

# Package metadata
__version__ = "0.6.0.0"
__author__ = "Marios Karaoulis"
__email__ = "marios@example.com"

# Define what gets imported with "from vtkclass import *"
__all__ = ['VtkClass']