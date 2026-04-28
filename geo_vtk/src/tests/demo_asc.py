# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:40:04 2020

@author: karaouli
"""

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

int1.make_asc_to_vtk('..\\data\\arcgis\\d50_1m.asc','..\\data\\arcgis\\d50_1m.vtk',v=0)



# Plot is a surface 
