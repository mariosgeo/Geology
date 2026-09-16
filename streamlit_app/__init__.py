"""
Streamlit Geological Inpainting Package.
"""

from .geotop_colors import (
    GEOTOP_CLASSES,
    GEOTOP_LABELS,
    GEOTOP_CMAP,
    GEOTOP_HEX_MAP,
    GEOTOP_NAME_MAP,
    get_class_name,
    get_class_color
)
from .data_loader import load_borehole_csv
from .projection import (
    load_or_create_profile,
    project_boreholes_to_profile,
    discretize_intervals_to_points
)
from .inpainter import run_geological_inpainting, compute_alphashape_mask

__all__ = [
    'GEOTOP_CLASSES',
    'GEOTOP_LABELS',
    'GEOTOP_CMAP',
    'GEOTOP_HEX_MAP',
    'GEOTOP_NAME_MAP',
    'get_class_name',
    'get_class_color',
    'load_borehole_csv',
    'load_or_create_profile',
    'project_boreholes_to_profile',
    'discretize_intervals_to_points',
    'run_geological_inpainting',
    'compute_alphashape_mask',
    'GEOTOP_COLORMAP',
    'run_inpainting',
]

GEOTOP_COLORMAP = GEOTOP_CMAP
run_inpainting = run_geological_inpainting
