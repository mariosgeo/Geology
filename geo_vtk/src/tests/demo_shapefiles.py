# -*- coding: utf-8 -*-
"""
In this demo, we utilize geopandas to read a polygon written in a shape file
For every polygon, we will create a vtk
"""

import geopandas
from vtkclass import VtkClass
import numpy as np



# Intialize the class
int1=VtkClass()

# Read a polygon shape file. This file has two polygons
df = geopandas.read_file('..\\data\\arcgis\\gebied_S7-20.shp')

no_of_polygones=df.shape[0]
# create a polygon in vtk, for every one
for i in range(0,no_of_polygones):
    # get the coordinates from each polygon. Since VTK is 3d, we have to add zeros on z plane
    polygon_perimeter=np.asarray(df.iloc[i].geometry.boundary.coords.xy)
    int1.make_2d_plane('../data/vtk/polygon_%d.vtk'%i,np.c_[polygon_perimeter.T,np.zeros((polygon_perimeter.shape[1],1))],i)
    
# read now a shape file with points
df = geopandas.read_file('..\\data\\arcgis\\uitgevoerde_boringen_NITGnr_S7-20.shp')
# This file has aan x,y columt that represent points. 
x=df['X'].values
y=df['Y'].values
z=np.zeros(x.shape)
int1.make_xyz_points('../data/vtk/points.vtk',np.c_[x,y,z])


# Finally, let's read an .asc file that containts elevation