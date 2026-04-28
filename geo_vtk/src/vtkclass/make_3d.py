# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:31:44 2024

@author: marios

3D VTK data processing utilities.
"""

import sys
import os

# Add relative paths for dependencies
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
repo_root = os.path.dirname(os.path.dirname(parent_dir))

# Add gridder and tools directories relative to repository structure
gridder_path = os.path.join(repo_root, 'gridder')
if os.path.exists(gridder_path) and gridder_path not in sys.path:
    sys.path.insert(0, gridder_path)

# Import dependencies if available
try:
    from xyz_of_vtk import xyz_of_vtk
except ImportError:
    print("Warning: xyz_of_vtk module not found. Some functionality may be limited.")
    xyz_of_vtk = None

# Example usage (commented out to avoid execution on import)
# if xyz_of_vtk:
#     data, cell_data, log_res, res, sens, x_center, y_center, z_center, xg, yg, prediction_data = xyz_of_vtk(r'temp\00_2d_norm.vtk')




