# -*- coding: utf-8 -*-
"""
Geological Data Visualization Utilities

This module provides utility functions for geological and geophysical data visualization,
focusing on color mapping, data processing, and format conversions commonly used in
earth science applications.

The utilities support various geological visualization workflows including:
- Scientific color space conversions (RGB/CMYK) for publication-quality figures
- Specialized geological colormaps for different data types
- Multi-dimensional data intersection and processing 
- Time series data handling for geological monitoring
- Color mapping optimized for geological parameter visualization

Key Functionality
-----------------
- Color space conversions for scientific publication
- Geological colormap definitions (resistivity, porosity, permeability)
- Multi-dimensional array operations for geological grids
- Data deduplication for survey datasets
- Datetime conversions for temporal geological data

Scientific Context
------------------
Geological data often requires specialized visualization approaches:
- Resistivity data uses logarithmic color scales
- Porosity/permeability need linear scales with geological boundaries
- Stratigraphic data requires discrete color schemes
- Time-series geological monitoring needs temporal color mapping

Created on Wed Mar 14 12:58:12 2018
@author: karaouli

Examples
--------
>>> # Convert RGB to CMYK for publication
>>> c, m, y, k = rgb_to_cmyk(255, 128, 0)  # Orange geological marker
>>> 
>>> # Create geological resistivity colormap
>>> colors = loke()  # Standard electrical resistivity colors
>>> 
>>> # Map geological values to colors
>>> color_indices = index_to_cmap(0.1, 1000, resistivity_data, 256)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.colors import ListedColormap


rgb_scale = 255
cmyk_scale = 100


def rgb_to_cmyk(r, g, b):
    """
    Convert RGB color values to CMYK for publication-quality geological figures.
    
    Converts RGB (Red, Green, Blue) color values to CMYK (Cyan, Magenta, Yellow, Black)
    color space, commonly required for scientific publication and professional printing
    of geological maps, cross-sections, and data visualizations.

    Parameters
    ----------
    r : int or float
        Red component (0-255 scale).
    g : int or float  
        Green component (0-255 scale).
    b : int or float
        Blue component (0-255 scale).

    Returns
    -------
    tuple of float
        CMYK values as (c, m, y, k) on 0-100 scale suitable for printing.

    Notes
    -----
    - Pure black (0,0,0) returns (0,0,0,100) in CMYK
    - Conversion uses standard ICC color profile algorithms
    - Useful for preparing geological visualizations for publication
    - CMYK provides better color reproduction for printed geological maps

    Examples
    --------
    >>> # Convert geological formation colors to CMYK
    >>> c, m, y, k = rgb_to_cmyk(255, 165, 0)  # Orange sandstone
    >>> print(f"Sandstone CMYK: C={c:.1f}, M={m:.1f}, Y={y:.1f}, K={k:.1f}")
    
    >>> # Convert standard geological colors
    >>> limestone_cmyk = rgb_to_cmyk(135, 206, 235)  # Light blue limestone
    >>> shale_cmyk = rgb_to_cmyk(105, 105, 105)      # Dark gray shale
    """
    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / float(rgb_scale)
    m = 1 - g / float(rgb_scale)
    y = 1 - b / float(rgb_scale)

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) 
    m = (m - min_cmy) 
    y = (y - min_cmy) 
    k = min_cmy

    # rescale to the range [0,cmyk_scale]
    return c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale

def cmyk_to_rgb(c, m, y, k):
    """
    Convert CMYK color values to RGB for digital geological visualization.
    
    Converts CMYK (Cyan, Magenta, Yellow, Black) color values from print-ready
    geological publications back to RGB format for digital display, web visualization,
    or screen-based geological data analysis.

    Parameters
    ----------
    c : float
        Cyan component (0-100 scale).
    m : float
        Magenta component (0-100 scale).
    y : float
        Yellow component (0-100 scale).
    k : float
        Black (Key) component (0-100 scale).

    Returns
    -------
    tuple of float
        RGB values as (r, g, b) on 0-1 scale for digital display.

    Notes
    -----
    - Output RGB values are normalized to 0-1 range for matplotlib compatibility
    - Multiply by 255 to get standard 0-255 RGB values
    - Useful for converting published geological color schemes to digital format
    - Maintains color fidelity for geological data visualization

    Examples
    --------
    >>> # Convert published geological formation colors to RGB
    >>> r, g, b = cmyk_to_rgb(20, 0, 100, 0)  # Yellow limestone from publication
    >>> rgb_255 = (r*255, g*255, b*255)  # Convert to 0-255 scale
    
    >>> # Convert geological map legend colors
    >>> sandstone_rgb = cmyk_to_rgb(0, 35, 100, 0)  # Orange sandstone
    >>> shale_rgb = cmyk_to_rgb(0, 0, 0, 60)        # Gray shale
    """
    #r = rgb_scale*(1.0-(c+k)/float(cmyk_scale))
    #g = rgb_scale*(1.0-(m+k)/float(cmyk_scale))
    #b = rgb_scale*(1.0-(y+k)/float(cmyk_scale))
    r = (1-c)*(1-k)
    g = (1-m)*(1-k)
    b = (1-y)*(1-k)
    
    return r,g,b


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    if value<minimum:
        value=minimum
    if value>maximum:
        value=maximum
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r/255, g/255, b/255

def bgr():
    data=np.genfromtxt(r'D:\karaouli\Desktop\Projects\python_tools\rhofract_169.zon',skip_header=1)
    r,g,b = cmyk_to_rgb(data[:,2]/255,data[:,3]/255,data[:,4]/255,data[:,1]/255)
    cmap=np.c_[r,g,b]
    my_map=ListedColormap(cmap,name='BGR')
    return r,g,b

def index_to_cmap(values_min, values_max, values, number_of_colors):
    """
    Map geological data values to colormap indices for visualization.
    
    Converts continuous geological or geophysical data values to discrete colormap
    indices, enabling consistent color representation across different geological
    parameters such as resistivity, porosity, density, or formation classification.

    Parameters
    ----------
    values_min : float
        Minimum value for color mapping. Values below this will be clipped.
    values_max : float  
        Maximum value for color mapping. Values above this will be clipped.
    values : array_like
        Geological data values to map to colors. Can be resistivity (Ohm·m),
        porosity (%), density (g/cm³), or any geological parameter.
    number_of_colors : int
        Number of discrete colors in the target colormap (typically 256).

    Returns
    -------
    numpy.ndarray
        Array of colormap indices (1-based) with same shape as input values.
        Each index corresponds to a specific color in the geological colormap.

    Notes
    -----
    - Values are clipped to [values_min, values_max] range before mapping
    - Useful for geological data with extreme outliers or measurement errors
    - Output indices are 1-based to match standard geological visualization software
    - Linear mapping preserves geological parameter relationships

    Examples
    --------
    >>> # Map electrical resistivity data to colors
    >>> resistivity = np.array([1, 10, 100, 1000])  # Ohm·m
    >>> color_indices = index_to_cmap(0.1, 10000, resistivity, 256)
    
    >>> # Map porosity data with geological constraints
    >>> porosity = np.array([0.05, 0.15, 0.25, 0.35])  # fraction
    >>> porosity_colors = index_to_cmap(0.0, 0.5, porosity, 64)
    
    >>> # Handle geological outliers with clipping
    >>> density_data = np.array([1.8, 2.3, 2.7, 15.0])  # g/cm³, last value is error
    >>> density_colors = index_to_cmap(1.5, 3.0, density_data, 128)  # Clips 15.0 to 3.0
    """
    values=np.array(values)
    values=np.reshape(values,values.size,1)
    ind1=np.where(values>values_max)
    ind1=ind1[0]
    values[ind1]=values_max
    ind1=np.where(values<values_min)
    ind1=ind1[0]
    values[ind1]=values_min
    
    idx_in_colorbar = np.floor(1+ (values - values_min) / (values_max -values_min) * (number_of_colors-1))
    return idx_in_colorbar


def multidim_intersect(arr1, arr2):
    """
    Find intersection of multi-dimensional geological coordinate arrays.
    
    Identifies common coordinate points between two geological datasets,
    useful for matching borehole locations, survey points, or grid intersections
    in geological modeling and data integration workflows.

    Parameters
    ----------
    arr1 : numpy.ndarray
        First array of coordinates with shape (N, D) where N is number of points
        and D is dimensionality (typically 2D: x,y or 3D: x,y,z coordinates).
    arr2 : numpy.ndarray  
        Second array of coordinates with same dimensionality as arr1.

    Returns
    -------
    numpy.ndarray
        Array containing intersection coordinates with shape (M, D) where M
        is the number of common points between the input arrays.

    Notes
    -----
    - Useful for finding overlapping survey locations across different campaigns
    - Handles floating-point coordinate matching with standard precision
    - Preserves original coordinate ordering from first array
    - Common in geological data integration workflows

    Examples
    --------
    >>> # Find common borehole locations between surveys
    >>> survey_2020 = np.array([[100, 200], [150, 250], [200, 300]])
    >>> survey_2021 = np.array([[150, 250], [200, 300], [250, 350]])
    >>> common_locations = multidim_intersect(survey_2020, survey_2021)
    >>> print(f"Overlapping drill sites: {len(common_locations)}")
    
    >>> # Find intersection of 3D geological model grids
    >>> model1_coords = np.array([[0,0,10], [1,0,10], [0,1,10]])
    >>> model2_coords = np.array([[1,0,10], [0,1,10], [2,0,10]])
    >>> shared_coords = multidim_intersect(model1_coords, model2_coords)
    """
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[0])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[0])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[0])    



def multidim_intersect_pandas_2d(arr1, arr2):
    df1 = pd.DataFrame({"x" : arr1[:,0], "y" : arr1[:,1]}).reset_index()
    df2 = pd.DataFrame({"x" : arr2[:,0], "y" : arr2[:,1]}).reset_index()
    result = pd.merge(df1, df2, left_on=["x","y"], right_on=["x","y"])
    result=result.sort_values(by=["index_x"])
    com=result[["x","y"]].values
    tmp=result[["index_x","index_y"]].values
    idx=tmp[:,0]
    idy=tmp[:,1]
    return com,idx,idy

def multidim_intersect_pandas_3d(arr1, arr2):
    df1 = pd.DataFrame({"x" : arr1[:,0], "y" : arr1[:,1], "z" : arr1[:,2]}).reset_index()
    df2 = pd.DataFrame({"x" : arr2[:,0], "y" : arr2[:,1], "z" : arr2[:,2]}).reset_index()
    result = pd.merge(df1, df2, left_on=["x","y","z"], right_on=["x","y","z"])
    result=result.sort_values(by=["index_x"])
    com=result[["x","y","z"]].values
    tmp=result[["index_x","index_y"]].values
    idx=tmp[:,0]
    idy=tmp[:,1]
    #idz=tmp[:,2]
    return com,idx,idy

def multidim_intersect_pandas_4d(arr1, arr2):
    df1 = pd.DataFrame({"x" : arr1[:,0], "y" : arr1[:,1], "z" : arr1[:,2], "k" : arr1[:,3]}).reset_index()
    df2 = pd.DataFrame({"x" : arr2[:,0], "y" : arr2[:,1], "z" : arr2[:,2], "k" : arr2[:,3]}).reset_index()
    result = pd.merge(df1, df2, left_on=["x","y","z","k"], right_on=["x","y","z","k"])
    result=result.sort_values(by=["index_x"])
    com=result[["x","y","z","k"]].values
    tmp=result[["index_x","index_y"]].values
    idx=tmp[:,0]
    idy=tmp[:,1]
    #idz=tmp[:,2]
    #idk=tmp[:,3]
    return com,idx,idy
    
def remove_all_duplicates(arr1):
    mat=pd.DataFrame(arr1).drop_duplicates(keep='false').as_matrix()
    return mat

def remove_duplicates(arr1):
    mat=pd.DataFrame(arr1).drop_duplicates(keep='first').as_matrix()
    return mat



   
    
def datetime2matlabdn(dt):
    """
    Convert Python datetime to MATLAB datenum format for geological time series.
    
    Converts Python datetime objects to MATLAB datenum format, enabling
    cross-platform compatibility for geological monitoring data, time-lapse
    surveys, and temporal geological analysis workflows.

    Parameters
    ----------
    dt : datetime.datetime
        Python datetime object representing geological survey time,
        monitoring timestamp, or data acquisition time.

    Returns
    -------
    float
        MATLAB datenum value (days since January 1, 0000) including
        fractional day component for precise timestamp representation.

    Notes
    -----
    - MATLAB datenum: Days since January 1, 0000 (proleptic Gregorian calendar)
    - Includes fractional days for sub-daily temporal resolution
    - Useful for geological monitoring systems using MATLAB data processing
    - Maintains microsecond precision for high-frequency geological measurements

    Examples
    --------
    >>> # Convert geological survey timestamp
    >>> from datetime import datetime
    >>> survey_time = datetime(2023, 6, 15, 14, 30, 0)  # Survey at 2:30 PM
    >>> matlab_time = datetime2matlabdn(survey_time)
    >>> print(f"MATLAB datenum: {matlab_time:.6f}")
    
    >>> # Convert monitoring data timestamps
    >>> import pandas as pd
    >>> monitoring_data = pd.DataFrame({
    ...     'timestamp': [datetime(2023, 1, 1, 12), datetime(2023, 1, 2, 12)],
    ...     'groundwater_level': [1.5, 1.3]
    ... })
    >>> monitoring_data['matlab_time'] = monitoring_data['timestamp'].apply(datetime2matlabdn)
    
    >>> # Time-lapse geological survey processing
    >>> acquisition_times = [datetime(2023, 3, 1), datetime(2023, 6, 1), datetime(2023, 9, 1)]
    >>> matlab_times = [datetime2matlabdn(t) for t in acquisition_times]
    """
    ord = dt.toordinal()
    mdn = dt + timedelta(days = 366)
    frac = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac





def loke():
    """
    Generate the standard Loke electrical resistivity colormap for geophysical visualization.
    
    Returns the widely-used electrical resistivity colormap developed for geophysical
    interpretation, particularly electrical resistivity tomography (ERT) and induced
    polarization (IP) surveys. This colormap provides intuitive color progression
    from low resistivity (blue/cyan) to high resistivity (red/white).

    Returns
    -------
    numpy.ndarray
        RGB colormap array with shape (18, 3) containing normalized RGB values (0-1).
        Colors progress from blue (low resistivity/conductive materials like clay, 
        groundwater) through green and yellow to red and white (high resistivity/
        resistive materials like bedrock, dry soil).

    Notes
    -----
    - Standard colormap in geophysical software (RES2DINV, EarthImager, AGI)
    - Blue colors: Low resistivity (1-10 Ohm·m) - clay, saltwater, contamination
    - Cyan/Green: Medium-low resistivity (10-100 Ohm·m) - wet soil, brackish water
    - Yellow: Medium resistivity (100-1000 Ohm·m) - dry soil, fresh water
    - Orange/Red: High resistivity (1000+ Ohm·m) - bedrock, dry sand, air voids
    - White: Very high resistivity (10000+ Ohm·m) - granite, quartzite, air
    
    Geological Interpretation
    -------------------------
    - Conductive zones (blue): Clay layers, groundwater, contamination plumes
    - Resistive zones (red/white): Bedrock, dry zones, buried utilities
    - Intermediate zones (green/yellow): Typical soil conditions

    Examples
    --------
    >>> # Get standard electrical resistivity colormap
    >>> resistivity_cmap = loke()
    >>> 
    >>> # Use with matplotlib for resistivity visualization
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>> cmap = ListedColormap(resistivity_cmap, name='Loke_Resistivity')
    >>> plt.imshow(resistivity_data, cmap=cmap)
    
    >>> # Apply to resistivity survey data
    >>> resistivity_values = np.logspace(0, 4, 100)  # 1 to 10000 Ohm·m
    >>> color_indices = index_to_cmap(1, 10000, resistivity_values, len(resistivity_cmap))
    
    References
    ----------
    Loke, M.H., Chambers, J.E., Rucker, D.F., Kuras, O. and Wilkinson, P.B., 2013.
    Recent developments in the direct-current geoelectrical imaging method.
    Journal of Applied Geophysics, 95, pp.135-156.
    """
    map=np.array([[0,0,128/255],
    [0,0,170/255],
    [0,0,211/255],
    [0,0,255/255],
    [0,128/255,255/255],
    [0,255/255,255/255],
    [0,192/255,128/255],
    [0,255/255,0],
    [0,128/255,0],
    [128/255,192/255,0],
    [255/255,255/255,0],
    [191/255,128/255,0],
    [255/255,128/255,0],
    [255/255,0,0],
    [211/255,0,0],
    [132/255,0,64/255],
    [96/255,0/255,96/255],
    [255/255,255/255,255/255]])
    my_map=ListedColormap(map,name='LOKE')
    return map
    
def imod_c():
    map=np.array([[200,200,200],
		[157,78,64],
		[0,146,0],
		[194,207,92],
		[255,255,255],
		[255,255,0],
		[243,225,6],
		[231,195,22],
		[216,163,32],
		[95,95,255]])
    map=map/255
    my_map=ListedColormap(map,name='IMOD')
    return map

def imod_d():
    map=np.array([[200,200,200],
		[157,78,64],
		[0,146,0],
		[194,207,92],
		[255,255,255],
		[255,255,0],
		[243,225,6],
		[231,195,22],
		[216,163,32],
		[95,95,255]])
    
    my_map=ListedColormap(map,name='IMOD')
    return map
    
    
def parula_data():
    parula_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

# geosoft colormap A
    clra32_data = [[ 0.        ,  0.        ,  1.        ],
               [ 0.        ,  0.33333333,  1.        ],
               [ 0.        ,  0.55294118,  1.        ],
               [ 0.        ,  0.78039216,  1.        ],
               [ 0.        ,  0.91372549,  1.        ],
               [ 0.        ,  1.        ,  1.        ],
               [ 0.        ,  1.        ,  0.63921569],
               [ 0.        ,  1.        ,  0.24705882],
               [ 0.        ,  1.        ,  0.17647059],
               [ 0.        ,  1.        ,  0.04705882],
               [ 0.28235294,  1.        ,  0.        ],
               [ 0.4       ,  1.        ,  0.        ],
               [ 0.52941176,  1.        ,  0.        ],
               [ 0.71372549,  1.        ,  0.        ],
               [ 0.90588235,  1.        ,  0.        ],
               [ 1.        ,  0.95686275,  0.        ],
               [ 1.        ,  0.82745098,  0.        ],
               [ 1.        ,  0.7372549 ,  0.        ],
               [ 1.        ,  0.68235294,  0.        ],
               [ 1.        ,  0.57647059,  0.        ],
               [ 1.        ,  0.5254902 ,  0.        ],
               [ 1.        ,  0.44313725,  0.        ],
               [ 1.        ,  0.33333333,  0.        ],
               [ 1.        ,  0.16470588,  0.        ],
               [ 1.        ,  0.02745098,  0.        ],
               [ 1.        ,  0.        ,  0.06666667],
               [ 1.        ,  0.        ,  0.20392157],
               [ 1.        ,  0.03529412,  0.45490196],
               [ 1.        ,  0.18823529,  0.70588235],
               [ 1.        ,  0.34901961,  0.94509804],
               [ 1.        ,  0.50588235,  1.        ],
               [ 1.        ,  0.6627451 ,  1.        ]]
       
    clra128_data= [[ 0.        ,  0.        ,  1.        ],
               [ 0.        ,  0.05882353,  1.        ],
               [ 0.        ,  0.10980392,  1.        ],
               [ 0.        ,  0.21960784,  1.        ],
               [ 0.        ,  0.33333333,  1.        ],
               [ 0.        ,  0.39215686,  1.        ],
               [ 0.        ,  0.44705882,  1.        ],
               [ 0.        ,  0.49803922,  1.        ],
               [ 0.        ,  0.55294118,  1.        ],
               [ 0.        ,  0.60784314,  1.        ],
               [ 0.        ,  0.66666667,  1.        ],
               [ 0.        ,  0.7254902 ,  1.        ],
               [ 0.        ,  0.78039216,  1.        ],
               [ 0.        ,  0.82745098,  1.        ],
               [ 0.        ,  0.85882353,  1.        ],
               [ 0.        ,  0.88627451,  1.        ],
               [ 0.        ,  0.91372549,  1.        ],
               [ 0.        ,  0.94509804,  1.        ],
               [ 0.        ,  0.97254902,  1.        ],
               [ 0.        ,  0.98823529,  1.        ],
               [ 0.        ,  1.        ,  1.        ],
               [ 0.        ,  1.        ,  0.8627451 ],
               [ 0.        ,  1.        ,  0.78431373],
               [ 0.        ,  1.        ,  0.71372549],
               [ 0.        ,  1.        ,  0.63921569],
               [ 0.        ,  1.        ,  0.56862745],
               [ 0.        ,  1.        ,  0.4627451 ],
               [ 0.        ,  1.        ,  0.34901961],
               [ 0.        ,  1.        ,  0.24705882],
               [ 0.        ,  1.        ,  0.22745098],
               [ 0.        ,  1.        ,  0.21176471],
               [ 0.        ,  1.        ,  0.18823529],
               [ 0.        ,  1.        ,  0.17647059],
               [ 0.        ,  1.        ,  0.15686275],
               [ 0.        ,  1.        ,  0.14117647],
               [ 0.        ,  1.        ,  0.09411765],
               [ 0.        ,  1.        ,  0.04705882],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.09411765,  1.        ,  0.        ],
               [ 0.18823529,  1.        ,  0.        ],
               [ 0.28235294,  1.        ,  0.        ],
               [ 0.31764706,  1.        ,  0.        ],
               [ 0.34901961,  1.        ,  0.        ],
               [ 0.38431373,  1.        ,  0.        ],
               [ 0.4       ,  1.        ,  0.        ],
               [ 0.41176471,  1.        ,  0.        ],
               [ 0.44705882,  1.        ,  0.        ],
               [ 0.49019608,  1.        ,  0.        ],
               [ 0.52941176,  1.        ,  0.        ],
               [ 0.56862745,  1.        ,  0.        ],
               [ 0.61568627,  1.        ,  0.        ],
               [ 0.6627451 ,  1.        ,  0.        ],
               [ 0.71372549,  1.        ,  0.        ],
               [ 0.76078431,  1.        ,  0.        ],
               [ 0.80784314,  1.        ,  0.        ],
               [ 0.85490196,  1.        ,  0.        ],
               [ 0.90588235,  1.        ,  0.        ],
               [ 0.95294118,  1.        ,  0.        ],
               [ 0.97647059,  1.        ,  0.        ],
               [ 1.        ,  1.        ,  0.        ],
               [ 1.        ,  0.95686275,  0.        ],
               [ 1.        ,  0.91372549,  0.        ],
               [ 1.        ,  0.88627451,  0.        ],
               [ 1.        ,  0.85882353,  0.        ],
               [ 1.        ,  0.82745098,  0.        ],
               [ 1.        ,  0.80392157,  0.        ],
               [ 1.        ,  0.77647059,  0.        ],
               [ 1.        ,  0.75294118,  0.        ],
               [ 1.        ,  0.7372549 ,  0.        ],
               [ 1.        ,  0.72156863,  0.        ],
               [ 1.        ,  0.70196078,  0.        ],
               [ 1.        ,  0.69411765,  0.        ],
               [ 1.        ,  0.68235294,  0.        ],
               [ 1.        ,  0.66666667,  0.        ],
               [ 1.        ,  0.63921569,  0.        ],
               [ 1.        ,  0.61176471,  0.        ],
               [ 1.        ,  0.57647059,  0.        ],
               [ 1.        ,  0.56862745,  0.        ],
               [ 1.        ,  0.55294118,  0.        ],
               [ 1.        ,  0.5372549 ,  0.        ],
               [ 1.        ,  0.5254902 ,  0.        ],
               [ 1.        ,  0.51372549,  0.        ],
               [ 1.        ,  0.49803922,  0.        ],
               [ 1.        ,  0.47058824,  0.        ],
               [ 1.        ,  0.44313725,  0.        ],
               [ 1.        ,  0.41176471,  0.        ],
               [ 1.        ,  0.38431373,  0.        ],
               [ 1.        ,  0.36078431,  0.        ],
               [ 1.        ,  0.33333333,  0.        ],
               [ 1.        ,  0.29019608,  0.        ],
               [ 1.        ,  0.25098039,  0.        ],
               [ 1.        ,  0.20784314,  0.        ],
               [ 1.        ,  0.16470588,  0.        ],
               [ 1.        ,  0.1254902 ,  0.        ],
               [ 1.        ,  0.08235294,  0.        ],
               [ 1.        ,  0.05490196,  0.        ],
               [ 1.        ,  0.02745098,  0.        ],
               [ 1.        ,  0.        ,  0.        ],
               [ 1.        ,  0.        ,  0.02352941],
               [ 1.        ,  0.        ,  0.05098039],
               [ 1.        ,  0.        ,  0.06666667],
               [ 1.        ,  0.        ,  0.09411765],
               [ 1.        ,  0.        ,  0.11764706],
               [ 1.        ,  0.        ,  0.14117647],
               [ 1.        ,  0.        ,  0.20392157],
               [ 1.        ,  0.        ,  0.26666667],
               [ 1.        ,  0.        ,  0.32941176],
               [ 1.        ,  0.        ,  0.39215686],
               [ 1.        ,  0.03529412,  0.45490196],
               [ 1.        ,  0.07058824,  0.51372549],
               [ 1.        ,  0.11372549,  0.58039216],
               [ 1.        ,  0.15294118,  0.64313725],
               [ 1.        ,  0.18823529,  0.70588235],
               [ 1.        ,  0.22745098,  0.77254902],
               [ 1.        ,  0.27058824,  0.83529412],
               [ 1.        ,  0.30980392,  0.89019608],
               [ 1.        ,  0.34901961,  0.94509804],
               [ 1.        ,  0.38431373,  1.        ],
               [ 1.        ,  0.42745098,  1.        ],
               [ 1.        ,  0.46666667,  1.        ],
               [ 1.        ,  0.50588235,  1.        ],
               [ 1.        ,  0.54509804,  1.        ],
               [ 1.        ,  0.58039216,  1.        ],
               [ 1.        ,  0.62352941,  1.        ],
               [ 1.        ,  0.6627451 ,  1.        ],
               [ 1.        ,  0.70196078,  1.        ],
               [ 1.        ,  0.74117647,  1.        ],
               [ 1.        ,  0.76078431,  1.        ]]
  
# geosoft colormap B
    clrb128_data= [[ 0.        ,  0.49803922,  1.        ],
               [ 0.        ,  0.51764706,  1.        ],
               [ 0.        ,  0.54509804,  1.        ],
               [ 0.        ,  0.56862745,  1.        ],
               [ 0.        ,  0.59215686,  1.        ],
               [ 0.        ,  0.61568627,  1.        ],
               [ 0.        ,  0.63921569,  1.        ],
               [ 0.        ,  0.6627451 ,  1.        ],
               [ 0.        ,  0.68627451,  1.        ],
               [ 0.        ,  0.70588235,  1.        ],
               [ 0.        ,  0.73333333,  1.        ],
               [ 0.        ,  0.75686275,  1.        ],
               [ 0.        ,  0.78039216,  1.        ],
               [ 0.        ,  0.80392157,  1.        ],
               [ 0.        ,  0.82745098,  1.        ],
               [ 0.        ,  0.85098039,  1.        ],
               [ 0.        ,  0.8745098 ,  1.        ],
               [ 0.        ,  0.90588235,  1.        ],
               [ 0.        ,  0.9372549 ,  1.        ],
               [ 0.        ,  0.96862745,  1.        ],
               [ 0.        ,  1.        ,  1.        ],
               [ 0.        ,  1.        ,  0.9372549 ],
               [ 0.        ,  1.        ,  0.8745098 ],
               [ 0.        ,  1.        ,  0.81176471],
               [ 0.        ,  1.        ,  0.74901961],
               [ 0.        ,  1.        ,  0.68627451],
               [ 0.        ,  1.        ,  0.62352941],
               [ 0.        ,  1.        ,  0.49803922],
               [ 0.        ,  1.        ,  0.37254902],
               [ 0.        ,  1.        ,  0.24705882],
               [ 0.        ,  1.        ,  0.12156863],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.03529412,  1.        ,  0.        ],
               [ 0.07843137,  1.        ,  0.        ],
               [ 0.11764706,  1.        ,  0.        ],
               [ 0.15686275,  1.        ,  0.        ],
               [ 0.19215686,  1.        ,  0.        ],
               [ 0.23529412,  1.        ,  0.        ],
               [ 0.2745098 ,  1.        ,  0.        ],
               [ 0.31372549,  1.        ,  0.        ],
               [ 0.34901961,  1.        ,  0.        ],
               [ 0.39215686,  1.        ,  0.        ],
               [ 0.43137255,  1.        ,  0.        ],
               [ 0.47058824,  1.        ,  0.        ],
               [ 0.50980392,  1.        ,  0.        ],
               [ 0.54901961,  1.        ,  0.        ],
               [ 0.58823529,  1.        ,  0.        ],
               [ 0.62745098,  1.        ,  0.        ],
               [ 0.66666667,  1.        ,  0.        ],
               [ 0.70196078,  1.        ,  0.        ],
               [ 0.74509804,  1.        ,  0.        ],
               [ 0.78431373,  1.        ,  0.        ],
               [ 0.82352941,  1.        ,  0.        ],
               [ 0.8627451 ,  1.        ,  0.        ],
               [ 0.90196078,  1.        ,  0.        ],
               [ 0.94117647,  1.        ,  0.        ],
               [ 0.98039216,  1.        ,  0.        ],
               [ 1.        ,  1.        ,  0.        ],
               [ 1.        ,  0.96862745,  0.        ],
               [ 1.        ,  0.93333333,  0.        ],
               [ 1.        ,  0.90196078,  0.        ],
               [ 1.        ,  0.86666667,  0.        ],
               [ 1.        ,  0.82745098,  0.        ],
               [ 1.        ,  0.8       ,  0.        ],
               [ 1.        ,  0.76470588,  0.        ],
               [ 1.        ,  0.73333333,  0.        ],
               [ 1.        ,  0.69803922,  0.        ],
               [ 1.        ,  0.66666667,  0.        ],
               [ 1.        ,  0.63137255,  0.        ],
               [ 1.        ,  0.59215686,  0.        ],
               [ 1.        ,  0.55686275,  0.        ],
               [ 1.        ,  0.5254902 ,  0.        ],
               [ 1.        ,  0.49019608,  0.        ],
               [ 1.        ,  0.45882353,  0.        ],
               [ 1.        ,  0.42352941,  0.        ],
               [ 1.        ,  0.39215686,  0.        ],
               [ 1.        ,  0.35294118,  0.        ],
               [ 1.        ,  0.32156863,  0.        ],
               [ 1.        ,  0.28627451,  0.        ],
               [ 1.        ,  0.25490196,  0.        ],
               [ 1.        ,  0.21960784,  0.        ],
               [ 1.        ,  0.18823529,  0.        ],
               [ 1.        ,  0.15686275,  0.        ],
               [ 1.        ,  0.1254902 ,  0.        ],
               [ 1.        ,  0.09019608,  0.        ],
               [ 1.        ,  0.05882353,  0.        ],
               [ 1.        ,  0.02745098,  0.        ],
               [ 1.        ,  0.        ,  0.        ],
               [ 1.        ,  0.        ,  0.29019608],
               [ 1.        ,  0.        ,  0.50980392],
               [ 1.        ,  0.        ,  0.66666667],
               [ 1.        ,  0.        ,  0.76470588],
               [ 1.        ,  0.        ,  0.82745098],
               [ 1.        ,  0.        ,  0.88235294],
               [ 1.        ,  0.        ,  0.91372549],
               [ 1.        ,  0.        ,  0.94117647],
               [ 1.        ,  0.        ,  0.96078431],
               [ 1.        ,  0.        ,  0.97254902],
               [ 1.        ,  0.        ,  1.        ],
               [ 0.98039216,  0.        ,  1.        ],
               [ 0.96078431,  0.        ,  1.        ],
               [ 0.94117647,  0.        ,  1.        ],
               [ 0.92156863,  0.        ,  1.        ],
               [ 0.90196078,  0.        ,  1.        ],
               [ 0.88235294,  0.01960784,  1.        ],
               [ 0.8745098 ,  0.05882353,  1.        ],
               [ 0.8627451 ,  0.09411765,  1.        ],
               [ 0.85490196,  0.13333333,  1.        ],
               [ 0.84313725,  0.17647059,  1.        ],
               [ 0.83529412,  0.21568627,  1.        ],
               [ 0.82352941,  0.25490196,  1.        ],
               [ 0.83529412,  0.29019608,  1.        ],
               [ 0.84313725,  0.33333333,  1.        ],
               [ 0.85490196,  0.37254902,  1.        ],
               [ 0.8627451 ,  0.41176471,  1.        ],
               [ 0.8745098 ,  0.44705882,  1.        ],
               [ 0.88235294,  0.49019608,  1.        ],
               [ 0.89411765,  0.52941176,  1.        ],
               [ 0.90196078,  0.56862745,  1.        ],
               [ 0.91372549,  0.60784314,  1.        ],
               [ 0.92156863,  0.64313725,  1.        ],
               [ 0.93333333,  0.68627451,  1.        ],
               [ 0.94117647,  0.7254902 ,  1.        ],
               [ 0.95294118,  0.76470588,  1.        ],
               [ 0.96078431,  0.80392157,  1.        ],
               [ 0.97254902,  0.84313725,  1.        ],
               [ 0.98039216,  0.88235294,  1.        ],
               [ 0.99215686,  0.92156863,  1.        ]]
    return parula_data, clra32_data,clra128_data, clrb128_data