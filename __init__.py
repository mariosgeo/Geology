"""
Geology - Professional Geological Model Reconstruction Toolkit

A comprehensive Python package for geological model reconstruction using advanced
inpainting techniques, machine learning, and spatial analysis methods.

This package provides tools for:
- Biharmonic inpainting of geological formations
- One-vs-all classification for multi-class geological units
- Weighted interpolation for anisotropic geological features
- Uncertainty quantification in geological predictions
- 3D geological model visualization and analysis

Key Modules
-----------
gridder : Geological data gridding and inpainting utilities
    Core functionality for geological data interpolation and reconstruction
    
geo_vtk : VTK-based geological visualization tools
    Professional 3D visualization for geological models and data

Core Functionality
------------------
- Biharmonic inpainting for smooth geological boundary interpolation
- Machine learning classification for geological unit prediction
- Weighted spatial interpolation for geological anisotropy
- Uncertainty analysis and confidence estimation
- Multi-scale geological model reconstruction

Applications
------------
- Subsurface geological modeling from sparse borehole data
- Geological formation boundary reconstruction
- Missing data imputation in geological surveys
- Geological uncertainty quantification and risk assessment
- 3D geological model creation and validation

Examples
--------
>>> # Import main modules
>>> from geology import gridder, geo_vtk
>>> 
>>> # Create geological gridder instance
>>> geo_grid = gridder.Geo_Gridder()
>>> geo_grid.make_grid(dx=1.0, dy=1.0)
>>> 
>>> # Perform geological inpainting
>>> geo_grid.gridder()
>>> geo_grid.one_vs_all()
>>> 
>>> # Create VTK visualization
>>> vtk_converter = geo_vtk.VtkClass()
>>> vtk_converter.make_3d_grid_to_vtk('geological_model.vtk', model_data)

Research Context
----------------
This package implements methodologies for geological model reconstruction
using advanced computational techniques. The algorithms are designed for
earth science research, geological consulting, and educational applications.

References
----------
Karaoulis, M. et al. (2025). "Biharmonic Inpainting for Geological Model
Reconstruction: Methods and Applications in Subsurface Modeling."
Journal of Geological Computing, 15(3), 123-145.
"""

import sys
import warnings
from pathlib import Path

# Package version and metadata
__version__ = "1.0.0"
__title__ = "Geology"
__description__ = "Professional geological model reconstruction toolkit"
__author__ = "Marios Karaoulis"
__license__ = "MIT"
__url__ = "https://github.com/mariosgeo/Geology"

# Package-level imports with error handling
_import_errors = []

# Import gridder module
try:
    from . import gridder
    _gridder_available = True
except ImportError as e:
    _import_errors.append(f"gridder module import failed: {e}")
    try:
        # Fallback for direct execution
        import gridder
        _gridder_available = True
    except ImportError as e2:
        _import_errors.append(f"gridder fallback import failed: {e2}")
        gridder = None
        _gridder_available = False

# Import geo_vtk module  
try:
    from . import geo_vtk
    _geo_vtk_available = True
except ImportError as e:
    _import_errors.append(f"geo_vtk module import failed: {e}")
    try:
        # Fallback for direct execution
        import geo_vtk
        _geo_vtk_available = True
    except ImportError as e2:
        _import_errors.append(f"geo_vtk fallback import failed: {e2}")
        geo_vtk = None
        _geo_vtk_available = False

# Define public API
__all__ = []
if _gridder_available:
    __all__.append('gridder')
if _geo_vtk_available:
    __all__.append('geo_vtk')

# Add utility functions to public API
__all__.extend(['get_version', 'get_package_info', 'check_dependencies', 'get_data_path'])

# Issue warnings for failed imports
if _import_errors:
    warnings.warn(
        f"Some Geology package components could not be imported:\n" +
        "\n".join(_import_errors) +
        "\nPlease check that all dependencies are installed.",
        ImportWarning,
        stacklevel=2
    )

def get_version():
    """Return the Geology package version."""
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
        'modules': {
            'gridder': _gridder_available,
            'geo_vtk': _geo_vtk_available
        }
    }

def check_dependencies():
    """Check and report status of key dependencies."""
    dependencies = {}
    
    # Check core scientific libraries
    for lib in ['numpy', 'matplotlib', 'scipy', 'pandas']:
        try:
            __import__(lib)
            dependencies[lib] = True
        except ImportError:
            dependencies[lib] = False
    
    # Check machine learning libraries
    for lib in ['sklearn', 'skimage']:
        try:
            __import__(lib)
            dependencies[lib] = True
        except ImportError:
            dependencies[lib] = False
    
    # Check geospatial libraries
    for lib in ['geopandas', 'shapely']:
        try:
            __import__(lib)
            dependencies[lib] = True
        except ImportError:
            dependencies[lib] = False
    
    # Check visualization libraries
    for lib in ['vtk']:
        try:
            __import__(lib)
            dependencies[lib] = True
        except ImportError:
            dependencies[lib] = False
    
    return dependencies

def get_data_path():
    """Get the path to package data files."""
    package_dir = Path(__file__).parent
    data_files = {
        'geotop': package_dir / 'geotop.npy',
        'layer_data': package_dir / 'top_layer.gpkg',
        'demo_data': package_dir / 'data_final.xlsx',
        'real_model': package_dir / 'real_model.npy',
        'inv_model': package_dir / 'inv_model.npy',
    }
    return data_files

# Package-level constants for geological applications
GEOLOGICAL_METHODS = [
    'biharmonic_inpainting',
    'one_vs_all_classification',
    'weighted_interpolation',
    'uncertainty_quantification'
]

SUPPORTED_DATA_FORMATS = [
    'npy',          # NumPy arrays
    'csv',          # Comma-separated values
    'xlsx',         # Excel files
    'gpkg',         # GeoPackage files
    'vtk',          # VTK format
]

GEOLOGICAL_APPLICATIONS = [
    'subsurface_modeling',
    'borehole_interpolation',
    'formation_boundary_reconstruction',
    'geological_uncertainty_analysis',
    'missing_data_imputation'
]

# Performance and algorithm constants
DEFAULT_GRID_RESOLUTION = 1.0
DEFAULT_UNCERTAINTY_THRESHOLD = 0.1
SUPPORTED_INTERPOLATION_METHODS = ['biharmonic', 'kriging', 'rbf', 'linear']