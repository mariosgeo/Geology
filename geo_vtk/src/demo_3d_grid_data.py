# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:44:15 2020

In this example, we will read a 3d model, as per mulitple 2d slices.
i.e. each slice is gfridded data per depth

Note that the dimensions of the elevatio and 2d slices should be the same (mxn)

To save space, everythin is saved as a numpy array. 
@author: karaouli
"""

from vtkclass import VtkClass
import numpy as np
import glob
# Intiallize our class
int1=VtkClass()




# first lest's read all the 2d slices and create a 3d matrix out of it
yt=glob.glob('..\\data\\3d_data\\bs_*')


for k in range(0,len(yt)):
    data=np.load(yt[k]) # load data slice by slice
    data=np.flipud(data) # Ignore this statmene, data were stored updide down
    # alocate matrix here, to gen the dimension of the data set
    if k==0:
        ic_3d=np.zeros((data.shape[0],data.shape[1],len(yt)))
    ic_3d=np.zeros((data.shape[0],data.shape[1],len(yt)))
    ic_3d[:,:,k]=data

    

# read the matrix that has the elevation. This is optinaol, if there is no elevation, 
# set this matrix to zeros
mosaic=np.load('..\\data\\3d_data\\\mosaic_5_focus.npy')
mosaic[mosaic>1e3]=0  # some house keeping



# Now let's read the coordantes for BOTH the elavation map and data
out_trans=np.load('..\\data\\3d_data\\map_coordinates.npy')
x_ahn3=np.arange(out_trans[2],out_trans[2]+mosaic.shape[1]*out_trans[0],out_trans[0])
y_ahn3=np.arange(out_trans[5],out_trans[5]+mosaic.shape[0]*out_trans[4],out_trans[4])

# In this example we know the z depth of each slice (constant 0.5 m spacing). If we load mxnxk slices,
# then the z represtns the top and bottom per layer. In other workds
# the 1st layer, has top and bottom surface of z[0] and z[1]
# This means that the z must me k+1 dimensions
# Notice that the spacing does not need to be constant
z_plots=np.arange(0,-2.5,-0.5)


# let's make the 3d vtk file now. Be patien here, is a bog data set
int1.make_3d_grid_to_vtk('..\\data\\vtk\\3d.vtk',ic_3d,x_ahn3,y_ahn3,z_plots,mosaic,name='IC')
