"""
Geological inpainting pipeline wrapper for Geo_Gridder.
Implements anisotropic biharmonic one-vs-all inpainting.
"""

import sys
import os
import numpy as np
from typing import Dict, Any, Tuple

# Ensure gridder is importable
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gridder_dir = os.path.join(repo_root, 'gridder')
if gridder_dir not in sys.path:
    sys.path.insert(0, gridder_dir)

import alphashape
import shapely

from gridder import Geo_Gridder


def compute_alphashape_mask(
    points: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    alpha: float = 1.0 / 150.0
) -> Tuple[np.ndarray, list, Any]:
    """
    Compute an alpha shape around training points and evaluate
    a boolean mask for regular 2D grid coordinates (xg, yg).
    
    Returns:
      inside_mask: 2D boolean array (True inside alpha shape, False outside)
      boundary_lines: list of 2D numpy arrays [2, N] of (x, y) coordinates for plotting outlines
      ashape: the Shapely Polygon or MultiPolygon object
    """
    if len(points) < 3:
        inside_mask = np.ones(xg.shape, dtype=bool)
        return inside_mask, [], None

    ashape = None
    try:
        if alpha is not None and alpha > 0:
            ashape = alphashape.alphashape(points, float(alpha))
            if ashape is None or ashape.is_empty or not hasattr(ashape, 'geom_type'):
                ashape = alphashape.alphashape(points, 0.0)
        else:
            ashape = alphashape.alphashape(points, 0.0)
    except Exception:
        try:
            ashape = alphashape.alphashape(points, 0.0)
        except Exception:
            inside_mask = np.ones(xg.shape, dtype=bool)
            return inside_mask, [], None

    # Determine inside/outside mask using shapely.contains_xy
    try:
        inside_mask = shapely.contains_xy(ashape, xg, yg)
    except Exception:
        # Fallback using path containment
        from matplotlib.path import Path
        if ashape.geom_type == 'Polygon':
            poly_path = Path(np.array(ashape.exterior.coords))
            inside_mask = poly_path.contains_points(np.c_[xg.ravel(), yg.ravel()]).reshape(xg.shape)
        elif ashape.geom_type == 'MultiPolygon':
            inside_mask = np.zeros(xg.shape, dtype=bool)
            coords_flat = np.c_[xg.ravel(), yg.ravel()]
            for poly in ashape.geoms:
                p_path = Path(np.array(poly.exterior.coords))
                inside_mask |= p_path.contains_points(coords_flat).reshape(xg.shape)
        else:
            inside_mask = np.ones(xg.shape, dtype=bool)

    # Extract boundary outlines for plotting
    boundary_lines = []
    if ashape.geom_type == 'Polygon':
        boundary_lines.append(np.array(ashape.exterior.xy))
    elif ashape.geom_type == 'MultiPolygon':
        for poly in ashape.geoms:
            boundary_lines.append(np.array(poly.exterior.xy))

    return inside_mask, boundary_lines, ashape


def run_geological_inpainting(
    training_points: np.ndarray,
    training_data: np.ndarray,
    dx: float = 25.0,
    dy: float = 0.5,
    x_weight: float = 1.0,
    y_weight: float = 3.0,
    apply_alphashape: bool = True,
    alpha: float = 1.0 / 150.0
) -> Dict[str, Any]:
    """
    Run the Geo_Gridder pipeline:
      1. Initialize Geo_Gridder with discrete 2D points (arc, z) and lithology labels.
      2. Construct regular grid using specified dx and dy.
      3. Grid observations using categorical mode aggregation.
      4. Perform anisotropic biharmonic one-vs-all inpainting.
      5. Optionally mask cells outside the alpha shape boundary (setting them to NaN).
      
    Returns dictionary with:
      'xg', 'yg': 2D coordinate meshgrids
      'sparse_grid': raw gridded data with NaNs (gg.bs)
      'prediction': inpainted continuous geological classification (gg.prediction_data)
      'uncertainty': class prediction uncertainty % (gg.uncertainty_data)
      'xs', 'ys': 1D grid axes
      'dx', 'dy': grid spacing
      'x_weight', 'y_weight': anisotropy weights
      'inside_mask': boolean 2D mask of valid cells inside alpha shape
      'alpha_shape_lines': list of boundary line coordinate arrays
      'apply_alphashape': bool indicating if alphashape masking was applied
      'alpha': alpha parameter used
    """
    if len(training_points) == 0:
        raise ValueError("No training points provided for inpainting.")

    # Instantiate Geo_Gridder with mode aggregation (for lithology units)
    gg = Geo_Gridder(training_points, training_data, method='mode')

    # Construct regular 2D grid
    gg.make_grid(dx=float(dx), dy=float(dy))

    # Grid discrete borehole samples into cells
    gg.gridder()

    # Inpaint unknown cells with anisotropic biharmonic one-vs-all
    gg.one_vs_all(x_weight=float(x_weight), y_weight=float(y_weight))

    prediction = gg.prediction_data.astype(float).copy()
    uncertainty = getattr(gg, 'uncertainty_data', np.zeros_like(prediction)).astype(float).copy()

    # Alpha shape computation and masking
    inside_mask = None
    boundary_lines = []
    ashape = None

    if apply_alphashape:
        inside_mask, boundary_lines, ashape = compute_alphashape_mask(
            training_points, gg.xg, gg.yg, alpha=alpha
        )
        if inside_mask is not None:
            # Mask predictions and uncertainty outside alpha shape to NaN
            prediction[~inside_mask] = np.nan
            uncertainty[~inside_mask] = np.nan

    unique_classes = np.unique(training_data).tolist()

    return {
        'xg': gg.xg,
        'yg': gg.yg,
        'xs': gg.xs,
        'ys': gg.ys,
        'sparse_grid': gg.bs,
        'prediction': prediction,
        'uncertainty': uncertainty,
        'dx': dx,
        'dy': dy,
        'x_weight': x_weight,
        'y_weight': y_weight,
        'num_points': len(training_data),
        'unique_classes': unique_classes,
        'grid_shape': gg.xg.shape,
        'inside_mask': inside_mask,
        'alpha_shape_lines': boundary_lines,
        'alpha_shape': ashape,
        'apply_alphashape': apply_alphashape,
        'alpha': alpha,
    }

