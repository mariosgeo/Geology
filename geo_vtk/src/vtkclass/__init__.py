"""
VtkClass - Professional Geological VTK Data Visualization Package

A comprehensive Python package for converting geological and geophysical data
to VTK format for 3D visualization and analysis. Supports borehole data,
3D geological models, raster data, and complex geological structures.

This module provides the main VtkClass interface for creating VTK files
from various geological data formats. Compatible with ParaView, VisIt,
Mayavi, and other VTK-based visualization software.

Classes
-------
VtkClass : Main class for geological VTK data conversion
    Primary interface for converting geological data to VTK format

Functions  
---------
Various utility functions for geological data processing and visualization

Examples
--------
>>> from vtkclass import VtkClass
>>> vtk = VtkClass()
>>> vtk.make_borehole_as_cube_multi('borehole.vtk', data, radius=1.0)
>>> vtk.make_3d_grid_to_vtk('model.vtk', grid_data, x_coords, y_coords, z_coords)
"""

import os
import sys

# Package version information
__version__ = "1.0.0"
__author__ = "Marios Karaoulis"
__email__ = "marios.karaoulis@example.com"
__license__ = "MIT"

# Package metadata
__title__ = "GeoVTK"
__description__ = "Professional geological VTK data visualization toolkit"
__url__ = "https://github.com/mariosgeo/Geology"

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import main classes with error handling
try:
    from .VtkClass import VtkClass
    __all__ = ['VtkClass']
except ImportError as e:
    # Fallback for cases where relative import fails
    import warnings
    warnings.warn(f"Could not import VtkClass: {e}. "
                  "Please check that all dependencies are installed.", 
                  ImportWarning)
    
    # Try absolute import as fallback
    try:
        from VtkClass import VtkClass
        __all__ = ['VtkClass']
    except ImportError:
        # Define empty __all__ if imports fail
        __all__ = []
        VtkClass = None

# Package-level constants
SUPPORTED_FORMATS = [
    'vtk',      # VTK legacy format
    'vtu',      # VTK XML unstructured grid
    'vtr',      # VTK XML rectilinear grid  
    'vts',      # VTK XML structured grid
]

GEOLOGICAL_APPLICATIONS = [
    'borehole_visualization',
    'geological_modeling', 
    'geophysical_surveys',
    'raster_conversion',
    'surface_generation',
    'point_cloud_processing'
]

def get_version():
    """Return the package version."""
    return __version__

def get_package_info():
    """Return comprehensive package information."""
    return {
        'name': __title__,
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'url': __url__,
        'supported_formats': SUPPORTED_FORMATS,
        'applications': GEOLOGICAL_APPLICATIONS
    }
    from VtkClass import VtkClass

# Package metadata
__version__ = "0.6.0.0"
__author__ = "Marios Karaoulis"
__email__ = "marios@example.com"

# Define what gets imported with "from vtkclass import *"
__all__ = ['VtkClass']