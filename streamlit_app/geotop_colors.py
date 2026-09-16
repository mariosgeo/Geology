"""
GeoTOP / IMOD standard geological classification colormaps and labels.
"""

import numpy as np
from matplotlib.colors import ListedColormap

# Standard IMOD / GeoTOP 10-lithology classification
GEOTOP_CLASSES = {
    0: {"name": "Anthropogenic", "color_rgb": [200, 200, 200], "hex": "#c8c8c8"},
    1: {"name": "Peat", "color_rgb": [157, 78, 64], "hex": "#9d4e40"},
    2: {"name": "Clay", "color_rgb": [0, 146, 0], "hex": "#009200"},
    3: {"name": "Silty clay / Silt", "color_rgb": [194, 207, 92], "hex": "#c2cf5c"},
    4: {"name": "Nothing / Background", "color_rgb": [255, 255, 255], "hex": "#ffffff"},
    5: {"name": "Fine sand / Sand", "color_rgb": [255, 255, 0], "hex": "#ffff00"},
    6: {"name": "Medium sand", "color_rgb": [243, 225, 6], "hex": "#f3e106"},
    7: {"name": "Coarse sand", "color_rgb": [231, 195, 22], "hex": "#e7c316"},
    8: {"name": "Gravel", "color_rgb": [216, 163, 32], "hex": "#d8a320"},
    9: {"name": "Shells", "color_rgb": [95, 95, 255], "hex": "#5f5fff"},
}

# Label array ordered by class index 0..9
GEOTOP_LABELS = [GEOTOP_CLASSES[i]["name"] for i in range(10)]

# Matplotlib ListedColormap
GEOTOP_RGB_NORM = np.array([GEOTOP_CLASSES[i]["color_rgb"] for i in range(10)], dtype=float) / 255.0
GEOTOP_CMAP = ListedColormap(GEOTOP_RGB_NORM, name="GeoTOP_IMOD")

# Plotly discrete color mapping
GEOTOP_HEX_MAP = {i: GEOTOP_CLASSES[i]["hex"] for i in range(10)}
GEOTOP_NAME_MAP = {i: GEOTOP_CLASSES[i]["name"] for i in range(10)}

# Text lithology mapper (useful when CSV has strings like 'sand', 'clay', etc.)
TEXT_TO_CLASS = {
    'anthropogenic': 0,
    'antropogeen': 0,
    'fill': 0,
    'peat': 1,
    'veen': 1,
    'clay': 2,
    'klei': 2,
    'silt': 3,
    'silty clay': 3,
    'siltig': 3,
    'nothing': 4,
    'none': 4,
    'background': 4,
    'fine sand': 5,
    'sand': 5,
    'zand': 5,
    'fine': 5,
    'medium sand': 6,
    'sand (m)': 6,
    'medium': 6,
    'coarse sand': 7,
    'sand (c)': 7,
    'coarse': 7,
    'gravel': 8,
    'grind': 8,
    'rocks': 8,
    'shells': 9,
    'schelpen': 9,
}


def get_class_name(class_id: int or float) -> str:
    """Return friendly name for a GeoTOP integer class code."""
    try:
        cid = int(round(float(class_id)))
        return GEOTOP_CLASSES.get(cid, {}).get("name", f"Class {cid}")
    except (ValueError, TypeError):
        return "Unknown"


def get_class_color(class_id: int or float) -> str:
    """Return hex color for a GeoTOP integer class code."""
    try:
        cid = int(round(float(class_id)))
        return GEOTOP_CLASSES.get(cid, {}).get("hex", "#888888")
    except (ValueError, TypeError):
        return "#888888"
