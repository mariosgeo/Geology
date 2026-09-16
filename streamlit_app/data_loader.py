"""
Borehole data loading and parsing module.
Handles CSV files matching test.csv and borehole_file.csv formats.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .geotop_colors import TEXT_TO_CLASS


def load_borehole_csv(file_or_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a borehole CSV file into:
      1) interval_df: detailed intervals with normalized columns:
         ['name', 'x', 'y', 'z_top', 'z_bottom', 'class_id', 'class_name']
      2) summary_df: 1 row per unique borehole:
         ['name', 'x', 'y', 'z_max', 'z_min', 'thickness', 'layer_count']
    """
    if isinstance(file_or_path, str):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)

    # Normalize column names (strip whitespace and lower/case-insensitive matching)
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)

    # Detect borehole ID column
    name_col = None
    for cand in ['name', 'borehole', 'borehole_id', 'hole_id', 'id', 'cpt']:
        for c in df.columns:
            if c.lower() == cand:
                name_col = c
                break
        if name_col:
            break

    # Detect X and Y coordinates
    x_col = None
    y_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ['x', 'x_coord', 'easting', 'rd_x'] and x_col is None:
            x_col = c
        elif cl in ['y', 'y_coord', 'northing', 'rd_y'] and y_col is None:
            y_col = c

    if x_col is None or y_col is None:
        raise ValueError("Could not find X and Y coordinate columns in CSV.")

    # If no name column found, construct from unique coordinate pairs
    if name_col is None:
        coords_series = df[x_col].astype(str) + "_" + df[y_col].astype(str)
        unique_coords = coords_series.unique()
        coord_to_name = {c: f"BH_{i+1:04d}" for i, c in enumerate(unique_coords)}
        df['name'] = coords_series.map(coord_to_name)
        name_col = 'name'

    # Detect top and bottom vertical boundaries
    # Priority 1: NAP_start / NAP_end (elevations in NAP, common in NL geological models)
    z_top_col = None
    z_bot_col = None
    if 'NAP_start' in df.columns and 'NAP_end' in df.columns:
        z_top_col = 'NAP_start'
        z_bot_col = 'NAP_end'
    elif 'depth_start' in df.columns and 'depth_end' in df.columns:
        # If elevation column exists, convert depth to elevation; else negative depth
        if 'elev' in df.columns or 'surface_elev' in df.columns:
            elev_c = 'elev' if 'elev' in df.columns else 'surface_elev'
            df['z_top'] = df[elev_c] - df['depth_start']
            df['z_bottom'] = df[elev_c] - df['depth_end']
        else:
            df['z_top'] = -df['depth_start']
            df['z_bottom'] = -df['depth_end']
        z_top_col = 'z_top'
        z_bot_col = 'z_bottom'
    elif 'up' in df.columns and 'down' in df.columns:
        # borehole_file.csv style
        if 'elev' in df.columns:
            df['z_top'] = df['elev'] - df['up']
            df['z_bottom'] = df['elev'] - df['down']
        else:
            df['z_top'] = -df['up']
            df['z_bottom'] = -df['down']
        z_top_col = 'z_top'
        z_bot_col = 'z_bottom'
    elif 'top' in df.columns and 'bottom' in df.columns:
        df['z_top'] = df['top']
        df['z_bottom'] = df['bottom']
        z_top_col = 'z_top'
        z_bot_col = 'z_bottom'

    if z_top_col is None or z_bot_col is None:
        raise ValueError("Could not find top and bottom interval boundaries (e.g. NAP_start/NAP_end or depth_start/depth_end).")

    # Detect lithology/class column
    class_col = None
    for cand in ['GTP number', 'classification_for_VTK_number', 'gtp_number', 'class_int', 'lithology', 'major_class']:
        for c in df.columns:
            if c.lower() == cand.lower():
                class_col = c
                break
        if class_col:
            break

    if class_col is None:
        raise ValueError("Could not find geological class column (e.g. 'GTP number' or 'major_class').")

    # Normalize class_id to integer 0..9
    raw_classes = df[class_col]
    if pd.api.types.is_numeric_dtype(raw_classes):
        class_ids = raw_classes.fillna(-1).astype(float)
    else:
        # Map text strings
        def map_text(val):
            if pd.isna(val):
                return -1.0
            s = str(val).strip().lower()
            return float(TEXT_TO_CLASS.get(s, -1))
        class_ids = raw_classes.apply(map_text)

    # Build clean interval DataFrame
    clean_df = pd.DataFrame({
        'name': df[name_col].astype(str),
        'x': df[x_col].astype(float),
        'y': df[y_col].astype(float),
        'z_top': np.maximum(df[z_top_col].astype(float), df[z_bot_col].astype(float)),
        'z_bottom': np.minimum(df[z_top_col].astype(float), df[z_bot_col].astype(float)),
        'class_id': class_ids,
    })

    # Drop intervals with missing coordinates or elevations
    clean_df = clean_df.dropna(subset=['x', 'y', 'z_top', 'z_bottom'])
    clean_df = clean_df[clean_df['class_id'] >= 0]
    clean_df['class_id'] = clean_df['class_id'].astype(int)

    # Compute summary per borehole
    summary_list = []
    for bh_name, group in clean_df.groupby('name'):
        x_val = group['x'].iloc[0]
        y_val = group['y'].iloc[0]
        z_max = group['z_top'].max()
        z_min = group['z_bottom'].min()
        summary_list.append({
            'name': str(bh_name),
            'x': float(x_val),
            'y': float(y_val),
            'z_max': float(z_max),
            'z_min': float(z_min),
            'thickness': float(z_max - z_min),
            'layer_count': int(len(group))
        })

    summary_df = pd.DataFrame(summary_list)

    return clean_df, summary_df
