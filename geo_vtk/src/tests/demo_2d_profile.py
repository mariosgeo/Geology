# -*- coding: utf-8 -*-
"""
In this example, we have a 2d profile that represent a geophysical image.
We want to project it to a 3d space.

There are several options and input files. We will start with simple 2d matrices, 
anmd increase complexity in due time

Created on Thu Feb  6 16:21:27 2020

@author: karaouli
"""


import numpy as np
from vtkclass import VtkClass



# Intiallize our class
int1=VtkClass()

# first let's create some randome 2d data.
data=np.random.rand(50,200)
# Let's pretend that this is a profile, where along the x axis you have data (200 points)
# along the profile, the z-axis represents inforation with depth (50 points). 

# say that the data on the x axis are every 1m, depth is information every 2m.
# you need to provde the x-axis and y-axis and z-xis


# First, let's plot it as a typical 2D profile (along x-axis). Y-axis then is
# set to zero 
x_spacing=1
z_spacing=2
x=np.arange(0,x_spacing*data.shape[1],x_spacing)
y=np.zeros((data.shape[1],1))
z=-np.arange(0,z_spacing*data.shape[0],z_spacing)

# let's plot in vtk
int1.make_2d_profiles('..\\data\\vtk\\simple_2d_profile.vtk', data, x, y, z)



# You can use actual coordiantes.
# Say that you know the begging and the end of the line, in x,y coordinates.
# then spacing is calulated automatically as also the correspongs coordiantes
x=np.array([-10,100])
y=np.array([-70,220])
z=np.array([0,-65])

int1.make_2d_profiles('..\\data\\vtk\\simple_2d_profile_fom_start_end_points.vtk', data, x, y, z)

# you might know few points along the profile... Placed them in order
x=np.array([-10,100,120,180])
y=np.array([-70,-90,140,220])
z=np.array([0,-65])

int1.make_2d_profiles('..\\data\\vtk\\simple_2d_profile_fom_multiple_points.vtk', data, x, y, z)
# Notice that in this scenario z is consant. Say if you had seg-y files to plot, z is the sampling rage




# A more simple approach, is that cells have variable diemsnions in x and y, and
# you want to add an image as texture. More help soon on this topic
x=np.array([-10,100,120,180,222])
y=np.array([-70,-90,140,220,340])
z=np.array([0,-65])

int1.make_2d_plane_for_texture('..\\data\\vtk\\simple_2d_profile_for_texture.vtk', x, y, z)


# A more generic approach is when we load x,y,z data from file, and we have variable depths and distances.
# Also each location is associated with an elevation
# We need to make a 2d matrix od the data, 
# an x,y, positioning
# the thickness of the cells 
# the elevation of the cell
# We will use as example file in the data\2d_data\em_csv
# This file represents a continuoes em survey
# each line corresponts to inversion results for that location
# each line is thus a sounding
# In this file, we have 12 layers as inversion resutls.
# Thus, we have 12 resistivity values (one per layer)
# Each layer thickenss is defined by the up boundary (12 columns)
# and the bottom boundary (12 columns)
# Adjust in you case accordingly
data=np.genfromtxt('..\\data\\2d_data\\em_data.csv',delimiter=',',skip_header=1)
x=data[:,0]
y=data[:,1] 
topo=data[:,2]
values=data[:,3:15].T
top=data[:,15:27].T
bot=data[:,27:39].T

# This is the most generic mesh, where for each x,y location, a seperate thickenss and elevation
# is defined. In this example though, all layer have the same thickness, thus it will produce the same
# results with the more simplieid verison (see below)
int1.make_2d_profiles_with_elevation('..\\data\\vtk\\generic_2d_profile.vtk',values,x,y,top,bot,topo)


# if the thickness of all layers is constant, then you need to define the the z only as a
# vecotr, with the same number of elements,as colums of the data matrix data

top2=np.matrix(top[:,0]).T
bot2=np.matrix(bot[:,0]).T
int1.make_2d_profiles_with_elevation('..\\data\\vtk\\generic_2d_profile_v2.vtk',values,x,y,top2,bot2,topo)



# Another scenraio is that you want to plot x,y,z data where while ther are gridded (spaging is contanst)
# ,they are stored as xyz file. There might be missing also data from the grid (in ERT the trapezoild shape)
# In this case theyt will not be ploted. 
# Addintially, the data repseresnets a 2D profile, but actually some points (from GPS) are known along the profile
# Such a scenario is the file ert_resutls.xyz witht he inversion model, and the ert_coordiantes.txt with few points along the profile

# The dx,dz will be calualted automatically. There are many diffrent ways to do that, 
# so look this as a guiide and adjust depending on your needs. 
data=np.genfromtxt('..\\data\\2d_data\ert_resutls.xyz',delimiter=',',skip_header=1)
# This file now has x,depth, resistivity as value, which are gridded.
# Load the coordinates.
coordiantes=np.genfromtxt('..\\data\\2d_data\ert_coordiantes.txt',delimiter=',',skip_header=0)
# This vector has few (al least 2) points along the profile, with x,y and elevation
x=coordiantes[:,0]
y=coordiantes[:,1]
elev=coordiantes[:,2]
int1.make_2d_profiles_with_elevation_scatter_data('..\\data\\vtk\\generic_2d_profile_missing.vtk',data,x,y,elev)
# or just plot in 2d
int1.make_2d_profiles_with_elevation_scatter_data('..\\data\\vtk\\generic_2d_profile_missing2.vtk',data)


# Finally, the more generic function, where the code will create a mesh, based on the x,y,z data,
# that are highly scattered
# The parameter trim removes traingles from the mesh, with higy diffeernt shape.
# More information soon
trim=0.01
int1.make_2d_profiles_from_xyz('..\\data\\vtk\\generic_2d_tringular.vtk',data,trim,x,y,elev)
