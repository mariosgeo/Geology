"""
Core functionality for the Geology package.

This module provides the main interface for geological modeling and inpainting.
"""

import numpy as np
import os
import sys

def setup_paths():
    """Setup paths to access gridder and geo_vtk modules."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Add gridder path
    gridder_path = os.path.join(base_dir, 'gridder')
    if gridder_path not in sys.path:
        sys.path.insert(0, gridder_path)
    
    # Add geo_vtk path
    geo_vtk_path = os.path.join(base_dir, 'geo_vtk', 'src')
    if geo_vtk_path not in sys.path:
        sys.path.insert(0, geo_vtk_path)

# Setup paths when module is imported
setup_paths()

def load_sample_data():
    """Load sample geological data for testing."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Try to load geotop.npy if it exists
    geotop_path = os.path.join(base_dir, 'geotop.npy')
    if os.path.exists(geotop_path):
        return np.load(geotop_path)
    else:
        # Return dummy data if no sample data available
        return np.random.rand(100, 100, 10)

def create_geological_model():
    """Create a new geological model instance."""
    try:
        from gridder import Geo_Gridder
        return Geo_Gridder()
    except ImportError:
        raise ImportError("Could not import Geo_Gridder. Please ensure gridder module is available.")

def create_vtk_converter():
    """Create a new VTK converter instance."""
    try:
        from vtkclass.VtkClass import VtkClass
        return VtkClass()
    except ImportError:
        raise ImportError("Could not import VtkClass. Please ensure geo_vtk module is available.")

# Main functionality exports
__all__ = ['setup_paths', 'load_sample_data', 'create_geological_model', 'create_vtk_converter']