# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:47:39 2020
In this demo, we read a griddeted DEM from a geotif 
or we provide the data as 2d matrix, and we make a vtk,
This is usefull, if the coordaintes stroe3d in the geotiff are in different 
projection
@author: karaouli
"""

from osgeo import gdalnumeric
from osgeo import gdal
from vtkclass import VtkClass
import numpy as np

# Intiallize our class
int1=VtkClass()

# First we read a geotif and we generate a vtk file
# we need to provide only the name of the file. The outfulll s generated automatically, in the same folder
# Notice that the code will interpolate the nan number

int1.tiff_to_vtk_3d('..\\data\\2d_data\\ahn3_05m_dtm_merge.tif')

# Alternativly, you can plot is as a surface (usefull when you want to embades another image)
int1.tiff_to_vtk('..\\data\\2d_data\\ahn3_05m_dtm_merge.tif')


# You can also call the function by providing the data as matrix.
# This is useful, when you want ot change the coordainte system
# YOu cave to manually provide x<coords,y_coords, x_res,y_res, ulx,uly
# See example to clarify

# Additionally, you can pass more data to plot. 
# in this case the second matrix, will be used for coloring.
# say you want to emade in DEM, land use. Mase sure size is the same


raster = gdal.Open('..\\data\\2d_data\\ahn3_05m_dtm_merge.tif')
ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
x_coords = ulx + np.arange(0,raster.RasterXSize) * xres +  (xres / 2) #add half the cell size
y_coords = uly + np.arange(0,raster.RasterYSize) * yres +  (yres / 2) #add half the cell size

int1.tiff_to_vtk_3d_data('..\\data\\2d_data\\dem_1.vtk',raster,x_coords,y_coords,xres,yres,0)

# Alternativly, you can plot is as a surface (usefull when you want to embades another image)
int1.tiff_to_vtk_data('..\\data\\2d_data\\dem_2.vtk',raster,x_coords,y_coords,xres,yres,0)
