"""
Geological VTK Tools - Source Package

This package provides tools for geological data processing and VTK file generation.
"""

# Import main modules
try:
    from .vtkclass import VtkClass
    from . import geo_utils
except ImportError:
    # Fallback for direct execution
    try:
        from vtkclass import VtkClass
        import geo_utils
    except ImportError:
        pass

__version__ = "0.6.0.0"
__all__ = ['VtkClass', 'geo_utils']