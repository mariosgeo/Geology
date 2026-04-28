"""
Geology - Professional Geological Model Inpainting Toolkit

A comprehensive research and development package for geological model reconstruction
using advanced inpainting techniques and machine learning methods.

This package provides tools for:
- Geological model reconstruction from sparse borehole data
- Biharmonic inpainting with geological constraints
- One-vs-all classification for geological units
- 3D geological visualization and analysis
- Uncertainty quantification for geological predictions
"""

__version__ = "1.0.0"
__author__ = "Marios Karaoulis"
__email__ = "marios.karaoulis@example.com"

# Import main functionality
from .core import *

# Package-level convenience functions
def get_version():
    """Get the package version."""
    return __version__

def get_info():
    """Get package information."""
    return {
        'name': 'Geology',
        'version': __version__,
        'author': __author__,
        'description': 'Professional geological model inpainting toolkit',
        'url': 'https://github.com/mariosgeo/Geology'
    }

# Make main classes available at package level
try:
    import sys
    import os
    
    # Add gridder to path
    gridder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gridder')
    if gridder_path not in sys.path:
        sys.path.insert(0, gridder_path)
    
    # Add geo_vtk to path  
    geo_vtk_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'geo_vtk', 'src')
    if geo_vtk_path not in sys.path:
        sys.path.insert(0, geo_vtk_path)
    
    # Import main classes
    from gridder import Geo_Gridder
    from vtkclass.VtkClass import VtkClass
    import geo_utils
    
    # Make available at package level
    __all__ = ['Geo_Gridder', 'VtkClass', 'geo_utils', 'get_version', 'get_info']
    
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    __all__ = ['get_version', 'get_info']