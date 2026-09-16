"""
Profile projection and depth discretization engine.
Projects 3D/spatial boreholes onto a 2D vertical cross-section plane (demo.ipynb style).
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import shapely.geometry as geom
from sklearn.decomposition import PCA
import scipy.interpolate


def load_or_create_profile(
    borehole_summary: pd.DataFrame,
    profile_path: Optional[str] = None
) -> Tuple[geom.LineString, np.ndarray]:
    """
    Load a reference profile (like profile2.txt) or compute a best-fit profile line
    from borehole coordinates.
    
    Returns:
      (shapely.geometry.LineString, profile_coords_Nx2)
    """
    coords = borehole_summary[['x', 'y']].drop_duplicates().values
    if len(coords) < 2:
        raise ValueError("Need at least 2 boreholes to define a profile line.")

    # Check if a profile file is supplied or profile2.txt exists
    if profile_path is None:
        default_p2 = os.path.join(os.getcwd(), 'profile2.txt')
        if os.path.exists(default_p2):
            profile_path = default_p2

    use_file = False
    if profile_path and os.path.exists(profile_path):
        try:
            raw_p = np.loadtxt(profile_path)
            # Check if coordinates overlap with borehole bounding box
            bx_min, bx_max = coords[:, 0].min(), coords[:, 0].max()
            by_min, by_max = coords[:, 1].min(), coords[:, 1].max()
            px_min, px_max = raw_p[:, 0].min(), raw_p[:, 0].max()
            py_min, py_max = raw_p[:, 1].min(), raw_p[:, 1].max()

            # If profile covers or overlaps the boreholes, use it
            overlap_x = max(0, min(bx_max, px_max) - max(bx_min, px_min))
            overlap_y = max(0, min(by_max, py_max) - max(by_min, py_min))
            if overlap_x > 0 or overlap_y > 0:
                use_file = True
                order = np.argsort(raw_p[:, 0])
                raw_p = raw_p[order]
                # interpolate at 2m spacing
                xs = np.arange(np.min(raw_p[:, 0]), np.max(raw_p[:, 0]), 2.0)
                ys = np.interp(xs, raw_p[:, 0], raw_p[:, 1])
                profile_pts = np.c_[xs, ys]
                line = geom.LineString(profile_pts)
                return line, profile_pts
        except Exception:
            use_file = False

    # Otherwise, automatically fit a spline/PCA profile through the boreholes
    pca = PCA(n_components=1)
    proj_1d = pca.fit_transform(coords).ravel()
    order = np.argsort(proj_1d)
    coords_sorted = coords[order]

    # If few points, use simple piecewise linear
    if len(coords_sorted) <= 3:
        profile_pts = coords_sorted
        line = geom.LineString(profile_pts)
        return line, profile_pts

    # Fit smooth spline
    try:
        tck, u = scipy.interpolate.splprep([coords_sorted[:, 0], coords_sorted[:, 1]], s=len(coords_sorted) * 2)
        unew = np.linspace(0, 1, 300)
        x_spline, y_spline = scipy.interpolate.splev(unew, tck)
        profile_pts = np.c_[x_spline, y_spline]
    except Exception:
        profile_pts = coords_sorted

    line = geom.LineString(profile_pts)
    return line, profile_pts


def project_boreholes_to_profile(
    borehole_summary: pd.DataFrame,
    profile_line: geom.LineString
) -> pd.DataFrame:
    """
    Project each borehole's (X, Y) onto the profile LineString.
    Computes:
      - 'arc': distance along profile line (meters)
      - 'dist_to_profile': perpendicular distance from borehole to profile line (meters)
      - 'proj_x', 'proj_y': projected coordinates on the line
    """
    df = borehole_summary.copy()
    arcs = []
    dists = []
    proj_xs = []
    proj_ys = []

    for _, row in df.iterrows():
        pt = geom.Point(row['x'], row['y'])
        arc_val = float(profile_line.project(pt))
        dist_val = float(profile_line.distance(pt))
        proj_pt = profile_line.interpolate(arc_val)

        arcs.append(arc_val)
        dists.append(dist_val)
        proj_xs.append(float(proj_pt.x))
        proj_ys.append(float(proj_pt.y))

    df['arc'] = arcs
    df['dist_to_profile'] = dists
    df['proj_x'] = proj_xs
    df['proj_y'] = proj_ys

    return df


def discretize_intervals_to_points(
    interval_df: pd.DataFrame,
    projected_summary: pd.DataFrame,
    selected_names: Optional[List[str]] = None,
    dz: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Discretize borehole intervals into 2D points (arc, z) with lithology class.
    
    Parameters:
      interval_df: DataFrame with ['name', 'z_top', 'z_bottom', 'class_id']
      projected_summary: DataFrame with ['name', 'arc']
      selected_names: List of borehole names to include (if None, includes all)
      dz: Vertical sampling step in meters (default 0.2 m, positive float)
      
    Returns:
      training_points: (N, 2) array of [arc, z]
      training_data: (N,) array of integer class IDs
      samples_df: DataFrame of discretized points
    """
    if selected_names is not None:
        int_sub = interval_df[interval_df['name'].isin(selected_names)].copy()
        sum_sub = projected_summary[projected_summary['name'].isin(selected_names)].copy()
    else:
        int_sub = interval_df.copy()
        sum_sub = projected_summary.copy()

    if len(int_sub) == 0:
        return np.empty((0, 2)), np.empty((0,)), pd.DataFrame()

    merged = int_sub.merge(sum_sub[['name', 'arc']], on='name')
    merged = merged.sort_values(by=['arc', 'z_top'], ascending=[True, False])

    step = abs(dz)
    pts_list = []

    for _, row in merged.iterrows():
        top = float(row['z_top'])
        bot = float(row['z_bottom'])
        arc = float(row['arc'])
        cid = int(row['class_id'])

        if top < bot:
            top, bot = bot, top

        # Sample from top down to bot
        depths = np.arange(top, bot, -step)
        if len(depths) == 0:
            depths = np.array([(top + bot) / 2.0])

        arcs_arr = np.full_like(depths, arc)
        cids_arr = np.full_like(depths, cid)

        pts_list.append(np.c_[arcs_arr, depths, cids_arr])

    if len(pts_list) == 0:
        return np.empty((0, 2)), np.empty((0,)), pd.DataFrame()

    all_pts = np.vstack(pts_list)
    training_points = all_pts[:, :2]
    training_data = all_pts[:, 2].astype(int)

    samples_df = pd.DataFrame({
        'arc': training_points[:, 0],
        'z': training_points[:, 1],
        'class_id': training_data
    })

    return training_points, training_data, samples_df
