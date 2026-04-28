"""
GeoVTK - Professional Geological VTK Data Visualization Library

A comprehensive Python package for converting geological and geophysical data 
to VTK format for 3D visualization and scientific analysis.

Key Components
--------------
vtkclass : Main VTK conversion classes
    - VtkClass: Primary interface for geological VTK data conversion
    
geo_utils : Geological data processing utilities  
    - Color mapping and conversion functions
    - Multi-dimensional data processing
    - Geological colormap definitions
    - Time series data handling

Applications
------------
- Borehole data visualization as 3D cylinders or cubes
- 3D geological model generation from grid data
- GeoTIFF raster data conversion to VTK surfaces  
- Complex geological structure creation
- Multi-property geological visualization with uncertainty
- Geophysical survey data visualization

Supported Data Types
-------------------
- Borehole logs and geological formations
- 3D geological grids and models
- GeoTIFF and raster elevation data
- Point cloud and scattered geological data
- Time-series geological monitoring data
- Multi-attribute geophysical datasets

Examples
--------
>>> # Import main components
>>> from geovtk import VtkClass, geo_utils
>>> 
>>> # Create VTK converter instance
>>> vtk = VtkClass()
>>> 
>>> # Convert borehole data to VTK
>>> vtk.make_borehole_as_cube_multi('borehole.vtk', borehole_data, radius=1.0)
>>> 
>>> # Use geological colormaps
>>> resistivity_colors = geo_utils.loke()
>>> color_indices = geo_utils.index_to_cmap(1, 1000, resistivity_data, 256)
"""

import sys
import warnings

# Package version and metadata
__version__ = "1.0.0"
__title__ = "GeoVTK"
__description__ = "Professional geological VTK data visualization library"
__author__ = "Marios Karaoulis"
__license__ = "MIT"
__url__ = "https://github.com/mariosgeo/Geology"

# Import main modules with comprehensive error handling
_import_errors = []

# Import VtkClass
try:
    from .vtkclass import VtkClass
    _vtk_available = True
except ImportError as e:
    _import_errors.append(f"VtkClass import failed: {e}")
    try:
        # Fallback for direct execution
        from vtkclass import VtkClass
        _vtk_available = True
    except ImportError as e2:
        _import_errors.append(f"VtkClass fallback import failed: {e2}")
        VtkClass = None
        _vtk_available = False

# Import geo_utils
try:
    from . import geo_utils
    _utils_available = True
except ImportError as e:
    _import_errors.append(f"geo_utils import failed: {e}")
    try:
        # Fallback for direct execution
        import geo_utils
        _utils_available = True
    except ImportError as e2:
        _import_errors.append(f"geo_utils fallback import failed: {e2}")
        geo_utils = None
        _utils_available = False

# Define public API
__all__ = []
if _vtk_available:
    __all__.append('VtkClass')
if _utils_available:
    __all__.append('geo_utils')

# Add utility functions to __all__
__all__.extend(['get_version', 'get_package_info', 'check_dependencies'])

# Issue warnings for failed imports
if _import_errors:
    warnings.warn(
        f"Some GeoVTK components could not be imported:\n" + 
        "\n".join(_import_errors) + 
        "\nPlease check that all dependencies are installed.",
        ImportWarning,
        stacklevel=2
    )

def get_version():
    """Return the GeoVTK package version."""
    return __version__

def get_package_info():
    """Return comprehensive package information."""
    return {
        'name': __title__,
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'license': __license__,
        'url': __url__,
        'components': {
            'VtkClass': _vtk_available,
            'geo_utils': _utils_available
        }
    }

def check_dependencies():
    """Check and report status of key dependencies."""
    dependencies = {}
    
    # Check core scientific libraries
    for lib in ['numpy', 'pandas', 'matplotlib', 'scipy']:
        try:
            __import__(lib)
            dependencies[lib] = True
        except ImportError:
            dependencies[lib] = False
    
    # Check geospatial libraries
    for lib in ['osgeo.gdal', 'osgeo.gdalnumeric']:
        try:
            __import__(lib)
            dependencies[lib.split('.')[-1]] = True
        except ImportError:
            dependencies[lib.split('.')[-1]] = False
    
    return dependencies

# Package-level constants for geological applications
SUPPORTED_VTK_FORMATS = [
    'vtk',      # VTK legacy format
    'vtu',      # VTK XML unstructured grid
    'vtr',      # VTK XML rectilinear grid  
    'vts',      # VTK XML structured grid
]

GEOLOGICAL_DATA_TYPES = [
    'borehole_logs',
    'geological_grids',
    'geotiff_rasters', 
    'point_clouds',
    'time_series',
    'geophysical_surveys'
]