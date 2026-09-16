"""
Geological Inpainting & Borehole Analysis Studio
Streamlit Web Application for Borehole Analysis and 2D Inpainting.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Patch

# Add current folder to sys.path so modules can be imported directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from streamlit_app.geotop_colors import (
    GEOTOP_CLASSES,
    GEOTOP_LABELS,
    GEOTOP_CMAP,
    GEOTOP_HEX_MAP,
    GEOTOP_NAME_MAP,
    get_class_name,
    get_class_color
)
from streamlit_app.data_loader import load_borehole_csv
from streamlit_app.projection import (
    load_or_create_profile,
    project_boreholes_to_profile,
    discretize_intervals_to_points
)
from streamlit_app.inpainter import run_geological_inpainting

# Configure Streamlit page
st.set_page_config(
    page_title="Geological Inpainting Studio",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 12px 18px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .legend-item {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Application Title
st.markdown('<div class="main-header">⛏️ Geological Inpainting & Borehole Analysis Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Interactive borehole selection on 2D map, profile plane projection, '
    'GeoTOP lithology coloring, and biharmonic one-vs-all inpainting.'
    '</div>',
    unsafe_allow_html=True
)

# Initialize Session State
if 'data_source' not in st.session_state:
    st.session_state['data_source'] = None
if 'clean_df' not in st.session_state:
    st.session_state['clean_df'] = None
if 'summary_df' not in st.session_state:
    st.session_state['summary_df'] = None
if 'projected_summary' not in st.session_state:
    st.session_state['projected_summary'] = None
if 'profile_line' not in st.session_state:
    st.session_state['profile_line'] = None
if 'profile_pts' not in st.session_state:
    st.session_state['profile_pts'] = None
if 'selected_boreholes' not in st.session_state:
    st.session_state['selected_boreholes'] = []
if 'inpaint_results' not in st.session_state:
    st.session_state['inpaint_results'] = None

# ==========================================
# SIDEBAR: DATA LOADING & CONTROLS
# ==========================================
st.sidebar.header("📁 Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload Borehole CSV (e.g. test.csv)",
    type=["csv"],
    help="CSV file containing borehole data (e.g., name, x, y, NAP_start, NAP_end, GTP number)"
)

col_b1, col_b2 = st.sidebar.columns(2)
load_default = col_b1.button("📂 Load test.csv", use_container_width=True)
load_borehole = col_b2.button("📄 borehole_file", use_container_width=True)

# Load data logic
data_to_load = None
data_name = None

if uploaded_file is not None:
    data_to_load = uploaded_file
    data_name = uploaded_file.name
elif load_default:
    default_csv = os.path.join(current_dir, 'test.csv')
    if os.path.exists(default_csv):
        data_to_load = default_csv
        data_name = "test.csv (Default)"
    else:
        st.sidebar.error("test.csv not found in current directory.")
elif load_borehole:
    bh_csv = os.path.join(current_dir, 'borehole_file.csv')
    if os.path.exists(bh_csv):
        data_to_load = bh_csv
        data_name = "borehole_file.csv"
    else:
        st.sidebar.error("borehole_file.csv not found.")
elif st.session_state['clean_df'] is None:
    # Auto-load test.csv on initial startup if present
    default_csv = os.path.join(current_dir, 'test.csv')
    if os.path.exists(default_csv):
        data_to_load = default_csv
        data_name = "test.csv (Auto-loaded)"

if data_to_load is not None:
    try:
        with st.spinner(f"Loading and parsing {data_name}..."):
            clean_df, summary_df = load_borehole_csv(data_to_load)
            line, line_pts = load_or_create_profile(summary_df)
            proj_summary = project_boreholes_to_profile(summary_df, line)

            st.session_state['clean_df'] = clean_df
            st.session_state['summary_df'] = summary_df
            st.session_state['profile_line'] = line
            st.session_state['profile_pts'] = line_pts
            st.session_state['projected_summary'] = proj_summary
            st.session_state['data_source'] = data_name
            # Default selection: first 25 or near profile
            near_bh = proj_summary[proj_summary['dist_to_profile'] < 150]['name'].tolist()
            if len(near_bh) > 0:
                st.session_state['selected_boreholes'] = near_bh[:30]
            else:
                st.session_state['selected_boreholes'] = proj_summary['name'].iloc[:25].tolist()
            st.session_state['inpaint_results'] = None
        st.sidebar.success(f"Loaded: {data_name}")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Check if data is loaded
if st.session_state['clean_df'] is None:
    st.info("👋 Welcome! Please upload a borehole CSV file or click **Load test.csv** in the sidebar to get started.")
    st.stop()

clean_df = st.session_state['clean_df']
summary_df = st.session_state['summary_df']
projected_summary = st.session_state['projected_summary']
profile_pts = st.session_state['profile_pts']

# Sidebar Metrics
st.sidebar.markdown(f"**Dataset:** `{st.session_state['data_source']}`")
st.sidebar.markdown(
    f"- Total Intervals: **{len(clean_df):,}**\n"
    f"- Total Boreholes: **{len(summary_df):,}**\n"
    f"- Elevation Range: **{clean_df['z_bottom'].min():.1f}m** to **{clean_df['z_top'].max():.1f}m**"
)

st.sidebar.divider()

# ==========================================
# SIDEBAR: PARAMETERS FOR PROJECTION & INPAINTING
# ==========================================
st.sidebar.header("⚙️ Inpainting Parameters")

# Checkbox for Plane Projection
plot_all_on_plane = st.sidebar.checkbox(
    "Plot all boreholes on plane (demo.ipynb style)",
    value=False,
    help="When checked, projects all boreholes within buffer onto the profile plane. When unchecked, only uses selected boreholes."
)

if plot_all_on_plane:
    max_dist_slider = st.sidebar.slider(
        "Max Distance to Profile (m)",
        min_value=10.0,
        max_value=2000.0,
        value=150.0,
        step=10.0,
        help="Include boreholes located within this perpendicular distance from the profile line."
    )
else:
    max_dist_slider = 500.0

st.sidebar.subheader("Grid Resolution")
dx_val = st.sidebar.slider(
    "dx (Horizontal spacing, m)",
    min_value=2.0,
    max_value=100.0,
    value=25.0,
    step=1.0,
    help="Grid cell width along the profile (arc distance)."
)

dy_val = st.sidebar.slider(
    "dy (Vertical spacing, m)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.05,
    help="Grid cell height along elevation (z)."
)

dz_sampling = st.sidebar.slider(
    "Interval Sampling Step dz (m)",
    min_value=0.05,
    max_value=1.0,
    value=0.2,
    step=0.05,
    help="Sampling resolution for converting layer intervals into point samples."
)

st.sidebar.subheader("Anisotropy Weights")
x_weight = st.sidebar.slider(
    "Horizontal Anisotropy (x_weight)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Weight for horizontal continuity in biharmonic equation."
)

y_weight = st.sidebar.slider(
    "Vertical Anisotropy (y_weight)",
    min_value=0.1,
    max_value=10.0,
    value=3.0,
    step=0.1,
    help="Weight for vertical continuity in biharmonic equation (default 3.0 favors vertical continuity)."
)

st.sidebar.subheader("Alpha Shape Boundary")
apply_alphashape = st.sidebar.checkbox(
    "Blank outside Alpha Shape",
    value=True,
    help="Blank out (set to NaN) all grid cells outside the borehole data alpha shape after inpainting is done."
)

if apply_alphashape:
    alpha_preset = st.sidebar.selectbox(
        "Alpha parameter preset",
        options=[
            "Standard (1/150 ≈ 0.0067, demo.ipynb)",
            "Convex Hull (Alpha = 0)",
            "Tight (1/75 ≈ 0.0133)",
            "Loose (1/300 ≈ 0.0033)",
            "Custom"
        ],
        index=0,
        help="Controls the tightness of the alpha shape boundary enclosing the boreholes."
    )
    if alpha_preset.startswith("Standard"):
        alpha_val = 1.0 / 150.0
    elif alpha_preset.startswith("Convex"):
        alpha_val = 0.0
    elif alpha_preset.startswith("Tight"):
        alpha_val = 1.0 / 75.0
    elif alpha_preset.startswith("Loose"):
        alpha_val = 1.0 / 300.0
    else:
        alpha_val = st.sidebar.number_input(
            "Custom Alpha value (1/distance)",
            min_value=0.0,
            max_value=0.5,
            value=float(1.0 / 150.0),
            step=0.001,
            format="%.5f"
        )

    show_alpha_outline = st.sidebar.checkbox(
        "Show Alpha Shape outline on plots",
        value=True,
        help="Draw the alpha shape boundary line over the cross-sections."
    )
else:
    alpha_val = 0.0
    show_alpha_outline = False

# Sidebar Quick Selection
st.sidebar.subheader("Borehole Selection")
col_s1, col_s2 = st.sidebar.columns(2)
if col_s1.button("Select All", use_container_width=True):
    st.session_state['selected_boreholes'] = summary_df['name'].tolist()
if col_s2.button("Clear", use_container_width=True):
    st.session_state['selected_boreholes'] = []

if st.sidebar.button("Select Near Profile (< 150m)", use_container_width=True):
    near_names = projected_summary[projected_summary['dist_to_profile'] < 150]['name'].tolist()
    st.session_state['selected_boreholes'] = near_names

# Sidebar Multi-select filter
selected_names_sidebar = st.sidebar.multiselect(
    "Choose boreholes manually:",
    options=summary_df['name'].tolist(),
    default=st.session_state['selected_boreholes'][:100] if len(st.session_state['selected_boreholes']) <= 100 else [],
    help="Search or select individual boreholes by name."
)
if selected_names_sidebar:
    st.session_state['selected_boreholes'] = list(set(st.session_state['selected_boreholes'] + selected_names_sidebar))


# ==========================================
# MAIN PAGE: TABS FOR MAP, PROFILE & INPAINTING
# ==========================================
tab_map, tab_profile, tab_inpaint = st.tabs([
    "🗺️ 1. Borehole Map & Mouse Selection",
    "📊 2. Stratigraphy & Plane Projection",
    "🎨 3. Image Inpainting"
])

# ----------------------------------------------------
# TAB 1: BOREHOLE MAP & SELECTION
# ----------------------------------------------------
with tab_map:
    st.subheader("Borehole Map (Use Mouse to Box/Lasso Select)")
    st.markdown(
        "Click the **Box Select** or **Lasso Select** tool in the top-right toolbar of the map below "
        "and drag across boreholes with your mouse to select them."
    )

    # Prepare map dataframe
    map_df = projected_summary.copy()
    current_selected_set = set(st.session_state['selected_boreholes'])
    map_df['Selected'] = map_df['name'].apply(lambda n: "Selected" if n in current_selected_set else "Unselected")

    # Build Plotly Figure
    fig_map = go.Figure()

    # Add reference profile line
    if profile_pts is not None:
        fig_map.add_trace(go.Scatter(
            x=profile_pts[:, 0],
            y=profile_pts[:, 1],
            mode='lines',
            line=dict(color='#ef4444', width=3, dash='dash'),
            name='Profile Line',
            hoverinfo='name'
        ))

    # Add unselected points
    unselected_df = map_df[map_df['Selected'] == "Unselected"]
    if len(unselected_df) > 0:
        fig_map.add_trace(go.Scatter(
            x=unselected_df['x'],
            y=unselected_df['y'],
            mode='markers',
            marker=dict(
                color='#94a3b8',
                size=6,
                opacity=0.6
            ),
            text=unselected_df['name'],
            customdata=np.c_[unselected_df['name'], unselected_df['z_max'], unselected_df['thickness'], unselected_df['dist_to_profile']],
            hovertemplate="<b>%{customdata[0]}</b><br>X: %{x:.1f}, Y: %{y:.1f}<br>Surface: %{customdata[1]:.1f}m<br>Thickness: %{customdata[2]:.1f}m<br>Dist to line: %{customdata[3]:.1f}m<extra></extra>",
            name='Unselected'
        ))

    # Add selected points
    selected_df = map_df[map_df['Selected'] == "Selected"]
    if len(selected_df) > 0:
        fig_map.add_trace(go.Scatter(
            x=selected_df['x'],
            y=selected_df['y'],
            mode='markers',
            marker=dict(
                color='#2563eb',
                size=9,
                opacity=0.9,
                line=dict(color='#ffffff', width=1)
            ),
            text=selected_df['name'],
            customdata=np.c_[selected_df['name'], selected_df['z_max'], selected_df['thickness'], selected_df['dist_to_profile']],
            hovertemplate="<b>%{customdata[0]} (Selected)</b><br>X: %{x:.1f}, Y: %{y:.1f}<br>Surface: %{customdata[1]:.1f}m<br>Thickness: %{customdata[2]:.1f}m<br>Dist to line: %{customdata[3]:.1f}m<extra></extra>",
            name='Selected'
        ))

    fig_map.update_layout(
        xaxis_title="X Coordinate (m)",
        yaxis_title="Y Coordinate (m)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        dragmode='select',  # defaults to box select for easy mouse interaction
        height=580,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Display plotly chart with interactive selection
    selection_event = st.plotly_chart(
        fig_map,
        on_select="rerun",
        selection_mode=["points", "box", "lasso"],
        key="borehole_map",
        use_container_width=True
    )

    # Update selection from mouse event
    if selection_event and "selection" in selection_event and "points" in selection_event["selection"]:
        clicked_pts = selection_event["selection"]["points"]
        if clicked_pts:
            picked_names = []
            for p in clicked_pts:
                # check customdata or text
                if "customdata" in p and p["customdata"] is not None:
                    picked_names.append(str(p["customdata"][0]))
                elif "text" in p and p["text"] is not None:
                    picked_names.append(str(p["text"]))
            if picked_names:
                st.session_state['selected_boreholes'] = list(set(picked_names))
                st.rerun()

    # Selection Summary Badge
    num_selected = len(st.session_state['selected_boreholes'])
    st.info(f"📍 **Currently Selected Boreholes:** {num_selected:,} of {len(summary_df):,} total.")


# ----------------------------------------------------
# TAB 2: BOREHOLE STRATIGRAPHY & PLANE PROJECTION
# ----------------------------------------------------
with tab_profile:
    st.subheader("Borehole Cross-Section & GeoTOP Lithology Visualization")

    # Determine which boreholes to plot based on checkbox
    if plot_all_on_plane:
        active_summary = projected_summary[projected_summary['dist_to_profile'] <= max_dist_slider].copy()
        st.caption(f"Showing **all {len(active_summary)} boreholes** within {max_dist_slider:.0f}m of the profile line (plane projection).")
    else:
        active_names = st.session_state['selected_boreholes']
        if len(active_names) == 0:
            st.warning("No boreholes selected. Please select boreholes on the map (Tab 1) or check 'Plot all boreholes on plane'.")
            active_summary = projected_summary.iloc[:10].copy()
        else:
            active_summary = projected_summary[projected_summary['name'].isin(active_names)].copy()
        st.caption(f"Showing **{len(active_summary)} selected boreholes** projected along the profile line.")

    # GeoTOP Legend bar
    st.markdown("**GeoTOP / IMOD Classification Palette:**")
    legend_cols = st.columns(5)
    for i in range(10):
        col_idx = i % 5
        c_info = GEOTOP_CLASSES[i]
        text_color = "#000000" if i in [0, 3, 4, 5, 6, 7] else "#ffffff"
        legend_cols[col_idx].markdown(
            f"<div class='legend-item' style='background-color:{c_info['hex']}; color:{text_color}; border:1px solid #cbd5e1;'>"
            f"<b>{i}</b>: {c_info['name']}</div>",
            unsafe_allow_html=True
        )

    st.write("")

    if len(active_summary) > 0:
        active_intervals = clean_df[clean_df['name'].isin(active_summary['name'])].copy()
        active_intervals = active_intervals.merge(active_summary[['name', 'arc', 'dist_to_profile']], on='name')
        active_intervals = active_intervals.sort_values(by=['arc', 'z_top'], ascending=[True, False])

        # Matplotlib Profile Projection Plot
        fig_prof, ax_prof = plt.subplots(figsize=(15, 6.5))

        # Determine column bar width based on profile span
        arc_min = active_summary['arc'].min()
        arc_max = active_summary['arc'].max()
        arc_span = max(10.0, arc_max - arc_min)
        bar_width = max(1.5, arc_span / (len(active_summary) * 3.5))

        # Plot each interval as a colored rectangle
        for _, row in active_intervals.iterrows():
            arc = row['arc']
            z_top = row['z_top']
            z_bot = row['z_bottom']
            cid = int(row['class_id'])
            color = get_class_color(cid)
            height = z_top - z_bot

            rect = plt.Rectangle(
                (arc - bar_width / 2.0, z_bot),
                bar_width,
                height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.3
            )
            ax_prof.add_patch(rect)

        # Plot ground surface line (interpolated top elevation)
        surf_df = active_summary.sort_values(by='arc')
        ax_prof.plot(surf_df['arc'], surf_df['z_max'], color='#334155', linestyle='--', linewidth=1.2, label='Ground Surface')

        ax_prof.set_xlim(arc_min - bar_width * 2, arc_max + bar_width * 2)
        ax_prof.set_ylim(active_intervals['z_bottom'].min() - 2, active_intervals['z_top'].max() + 3)
        ax_prof.set_xlabel("Distance Along Profile (arc, meters)", fontsize=11, fontweight='bold')
        ax_prof.set_ylabel("Elevation (NAP / meters)", fontsize=11, fontweight='bold')
        title_mode = "All Boreholes on Plane" if plot_all_on_plane else "Selected Boreholes on Plane"
        ax_prof.set_title(f"2D Projected Profile ({title_mode}) — {len(active_summary)} Boreholes", fontsize=13, fontweight='bold')
        ax_prof.grid(True, linestyle=':', alpha=0.6)
        ax_prof.legend(loc='upper right')

        st.pyplot(fig_prof)
        plt.close(fig_prof)

        # Expandable Stratigraphic Columns (Side-by-Side viewer)
        with st.expander("🔍 View Individual Stratigraphic Columns Side-by-Side", expanded=False):
            sample_bhs = active_summary['name'].iloc[:15].tolist()
            picked_bhs = st.multiselect("Select boreholes to view column details:", options=active_summary['name'].tolist(), default=sample_bhs)

            if picked_bhs:
                fig_cols, ax_cols = plt.subplots(figsize=(max(8, len(picked_bhs) * 1.1), 5))
                for idx, bh_n in enumerate(picked_bhs):
                    b_df = active_intervals[active_intervals['name'] == bh_n]
                    for _, row in b_df.iterrows():
                        z_top = row['z_top']
                        z_bot = row['z_bottom']
                        cid = int(row['class_id'])
                        color = get_class_color(cid)
                        ax_cols.add_patch(plt.Rectangle((idx - 0.35, z_bot), 0.7, z_top - z_bot, facecolor=color, edgecolor='black', linewidth=0.5))

                ax_cols.set_xlim(-0.8, len(picked_bhs) - 0.2)
                ax_cols.set_xticks(range(len(picked_bhs)))
                ax_cols.set_xticklabels(picked_bhs, rotation=45, ha='right', fontsize=9)
                ax_cols.set_ylabel("Elevation (m)", fontsize=10, fontweight='bold')
                ax_cols.set_title("Stratigraphic Columns of Selected Boreholes", fontsize=11, fontweight='bold')
                ax_cols.grid(True, axis='y', linestyle=':', alpha=0.5)
                st.pyplot(fig_cols)
                plt.close(fig_cols)


# ----------------------------------------------------
# TAB 3: IMAGE INPAINTING
# ----------------------------------------------------
with tab_inpaint:
    st.subheader("Biharmonic One-vs-All Image Inpainting")
    st.markdown(
        "Reconstruct a continuous 2D geological cross-section from the discrete borehole observations "
        "using anisotropic biharmonic one-vs-all inpainting."
    )

    # Determine input boreholes
    if plot_all_on_plane:
        target_summary = projected_summary[projected_summary['dist_to_profile'] <= max_dist_slider].copy()
        target_names = target_summary['name'].tolist()
    else:
        target_names = st.session_state['selected_boreholes']
        if len(target_names) == 0:
            target_names = projected_summary['name'].iloc[:20].tolist()

    alpha_status = f"Active (α = {alpha_val:.5f})" if apply_alphashape else "Disabled"
    st.markdown(
        f"**Input Configuration:**\n"
        f"- Target Boreholes: **{len(target_names)}**\n"
        f"- Grid Spacing: **dx = {dx_val:.1f} m**, **dy = {dy_val:.2f} m**\n"
        f"- Anisotropy Weights: **x_weight = {x_weight:.1f}**, **y_weight = {y_weight:.1f}**\n"
        f"- Sampling step: **dz = {dz_sampling:.2f} m**\n"
        f"- Blank Outside Alpha Shape: **{alpha_status}**"
    )

    run_btn = st.button("🚀 Run Geological Inpainting", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Running biharmonic one-vs-all inpainting and applying Alpha Shape boundary..."):
            t_start = time.time()
            try:
                # 1. Discretize intervals into 2D points (arc, z)
                training_points, training_data, samples_df = discretize_intervals_to_points(
                    clean_df,
                    projected_summary,
                    selected_names=target_names,
                    dz=dz_sampling
                )

                if len(training_points) == 0:
                    st.error("No valid points generated from the selected boreholes. Check elevation ranges.")
                else:
                    # 2. Run Geo_Gridder pipeline with post-inpainting alphashape blanking
                    results = run_geological_inpainting(
                        training_points,
                        training_data,
                        dx=dx_val,
                        dy=dy_val,
                        x_weight=x_weight,
                        y_weight=y_weight,
                        apply_alphashape=apply_alphashape,
                        alpha=alpha_val
                    )
                    results['elapsed_time'] = time.time() - t_start
                    st.session_state['inpaint_results'] = results
                    st.success(f"Inpainting complete in {results['elapsed_time']:.2f} seconds!")
            except Exception as ex:
                st.error(f"Inpainting error: {ex}")
                import traceback
                st.code(traceback.format_exc())

    # Render Results if available
    res = st.session_state.get('inpaint_results', None)
    if res is not None:
        xg = res['xg']
        yg = res['yg']
        sparse_grid = res['sparse_grid']
        prediction = res['prediction']
        uncertainty = res['uncertainty']

        st.divider()
        st.subheader("Inpainting Results")

        # Configure colormaps to render blank NaN cells as clean white
        plot_cmap = GEOTOP_CMAP.copy()
        plot_cmap.set_bad(color='white')

        unc_cmap = plt.cm.viridis.copy()
        unc_cmap.set_bad(color='white')

        # Visualizations (Matplotlib figure matching demo.ipynb)
        fig_res, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)

        # Plot 1: Sparse Gridded Observations
        mat1 = ax1.pcolor(xg, yg, sparse_grid, vmin=-0.5, vmax=9.5, cmap=plot_cmap, shading='auto')
        ax1.set_title("1. Sparse Gridded Boreholes (Input Observations)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Elevation (m)", fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.5)
        cbar1 = fig_res.colorbar(mat1, ax=ax1, orientation='vertical', pad=0.01)
        cbar1.set_ticks(range(10))
        cbar1.set_ticklabels(GEOTOP_LABELS, fontsize=8)

        # Plot 2: Inpainted Continuous Geological Profile
        mat2 = ax2.pcolor(xg, yg, prediction, vmin=-0.5, vmax=9.5, cmap=plot_cmap, shading='auto')
        mask_tag = " (Blanked Outside Alpha Shape)" if res.get('apply_alphashape') else ""
        ax2.set_title(
            f"2. Inpainted Geological Profile{mask_tag} (one_vs_all, dx={res['dx']:.1f}m, dy={res['dy']:.2f}m, "
            f"x_weight={res['x_weight']:.1f}, y_weight={res['y_weight']:.1f})",
            fontsize=12,
            fontweight='bold'
        )
        ax2.set_ylabel("Elevation (m)", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.5)
        cbar2 = fig_res.colorbar(mat2, ax=ax2, orientation='vertical', pad=0.01)
        cbar2.set_ticks(range(10))
        cbar2.set_ticklabels(GEOTOP_LABELS, fontsize=8)

        # Plot 3: Prediction Uncertainty
        mat3 = ax3.pcolor(xg, yg, uncertainty, vmin=0, vmax=100, cmap=unc_cmap, shading='auto')
        ax3.set_title(f"3. Classification Uncertainty (%){mask_tag}", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Profile Arc Distance (m)", fontsize=10)
        ax3.set_ylabel("Elevation (m)", fontsize=10)
        ax3.grid(True, linestyle=':', alpha=0.5)
        cbar3 = fig_res.colorbar(mat3, ax=ax3, orientation='vertical', pad=0.01)
        cbar3.set_label("Uncertainty %", fontsize=9)

        # Draw Alpha Shape boundary outlines if requested and available
        if show_alpha_outline and res.get('alpha_shape_lines'):
            for idx, line in enumerate(res['alpha_shape_lines']):
                lbl = 'Alpha Shape Boundary' if idx == 0 else ""
                ax1.plot(line[0], line[1], color='#0f172a', linewidth=1.5, linestyle='-', label=lbl)
                ax2.plot(line[0], line[1], color='#0f172a', linewidth=1.8, linestyle='-', label=lbl)
                ax3.plot(line[0], line[1], color='red', linewidth=1.8, linestyle='--', label=lbl)
            ax1.legend(loc='upper right', framealpha=0.85)
            ax2.legend(loc='upper right', framealpha=0.85)
            ax3.legend(loc='upper right', framealpha=0.85)

        st.pyplot(fig_res)
        plt.close(fig_res)

        # Summary Metrics
        st.write("")
        if res.get('inside_mask') is not None:
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric("Grid Dimensions", f"{xg.shape[1]} × {xg.shape[0]}")
            m_col2.metric("Total Grid Cells", f"{xg.size:,}")
            inside_cells = np.count_nonzero(res['inside_mask'])
            m_col3.metric("Inside Alpha Shape", f"{inside_cells:,} ({inside_cells / xg.size * 100:.1f}%)")
            blank_cells = xg.size - inside_cells
            m_col4.metric("Blanked Outside", f"{blank_cells:,} ({blank_cells / xg.size * 100:.1f}%)")
            m_col5.metric("Execution Time", f"{res['elapsed_time']:.2f}s")
        else:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Grid Dimensions", f"{xg.shape[1]} × {xg.shape[0]}")
            m_col2.metric("Total Grid Cells", f"{xg.size:,}")
            known_cells = np.count_nonzero(~np.isnan(sparse_grid))
            m_col3.metric("Observed Cells", f"{known_cells:,} ({known_cells / xg.size * 100:.1f}%)")
            m_col4.metric("Execution Time", f"{res['elapsed_time']:.2f}s")

        # Class Distribution Table
        with st.expander("📊 Inpainted Class Proportions & Breakdown", expanded=False):
            valid_pred = prediction[~np.isnan(prediction)].astype(int)
            counts = pd.Series(valid_pred).value_counts().sort_index()
            dist_df = pd.DataFrame({
                'Class Code': counts.index,
                'Lithology Name': [get_class_name(c) for c in counts.index],
                'Grid Cells': counts.values,
                'Volume / Area %': (counts.values / len(valid_pred) * 100).round(2)
            })
            st.dataframe(dist_df, use_container_width=True)

        # Download Inpainted Grid
        st.write("")
        export_df = pd.DataFrame(prediction, index=np.round(res['ys'], 2), columns=np.round(res['xs'], 2))
        csv_data = export_df.to_csv().encode('utf-8')
        st.download_button(
            label="💾 Download Inpainted Grid as CSV",
            data=csv_data,
            file_name="inpainted_geological_profile.csv",
            mime="text/csv",
            use_container_width=True
        )
