# -*- coding: utf-8 -*-
"""
In this demo scipt, we will create a 2d line, a 2d plane, a 3d cube and a 3d cylinder.
Created on Tue Feb  4 16:42:55 2020

@author: karaouli
"""

from vtkclass import VtkClass
import numpy as np


# Intialize the class
int1=VtkClass()


# Let's make a 2D line, with some coordinates. We will make a circle as a demo, by drawing every 1 degree.
# Notice that Paraview is a 3d viewer, so even if we want a 2D line, we have to set z coordinates. In this case we set it to zero
resolution=360
radius=1
t=np.linspace(0,360,resolution+1)

x=np.zeros(t.shape)
y=np.zeros(t.shape)
z=np.zeros(t.shape)
for i in range (0,t.shape[0]):
    x[i]=(0.5*np.abs(radius)*np.cos(t[i]*np.pi/180)+0)
    y[i]=(0.5*np.abs(radius)*np.sin(t[i]*np.pi/180)+0)

# we provide to the function the x,y,z as a matrix. We need also a filename
int1.make_1d_line('../data/vtk/circle.vtk',np.c_[x,y,z])



# Now, we will ame a 2D plane. The difference is that paraview will see this a close geometrical shape
# for simplicity, we will use the same points
int1.make_2d_plane('../data/vtk/circle_plane.vtk',np.c_[x,y,z])
# If you wante a simple orthogonal, we provide the coordinates of the 4 corners. Still notice that we need the z-coordiante, even if it's 2D
x=np.array([0,1,1,0])
y=np.array([5,5,0,0])
z=np.zeros(x.shape)
int1.make_2d_plane('../data/vtk/square_plane.vtk',np.c_[x,y,z])

# Hint, those planes are useful if you want to georeference any image.
# First you make the plane, and then in Paraview, 
# 1. Right click on the new object
# 2. Add filter
# 3. Alphabetical
# 4. Texture to plane
# 5. On the Miscellaneous tab, select texture and load an image. Perhards a tutorial later on
# Currently, you can not do that from the code.



# Notice, if you want to plot several shapes, and each to have a different color, 
# then you have to pass the property_colot value. This is a number (float) that coresponds to
# a colormap you will choose in Paraview. If two objects have the same property value, then the color
# will be the same. If they have differetn property_value then color is assignb based on the colormap
int1.make_2d_plane('../data/vtk/square_plane_box1.vtk',np.c_[x,y,z],1.)
int1.make_2d_plane('../data/vtk/square_plane_box2.vtk',np.c_[x+1,y,z],2.) # move box2 by 1 unit



# Now, let's make a 3D voxel. We have two ways, either by providing the coordinates of the 8 nodes.
# coordiantes on the x,y,z shoud be consinst clockwise, starting with top layer
x=np.array([0,1,1,0,0,1,1,0])
y=np.array([0,0,1,1,0,0,1,1])
z=np.array([0,0,0,0,1,1,1,1])
int1.make_3d_voxels('../data/vtk/cube.vtk',np.c_[x,y,z],1)

# of course you can make any other shape (trapezoidal)
x=np.array([0,1,1,0,-1,2,2,-1])
y=np.array([0,0,1,1,0,0,1,1])
z=np.array([0,0,0,0,1,1,1,1])
int1.make_3d_voxels('../data/vtk/trapez.vtk',np.c_[x,y,z],1)



# another way to make a cube, is to provide the center ofthe shaoe, and edges alnong x,y,z
x=0
y=0
z=0
edge=[1,2,1]
int1.make_simple_3d_voxel('../data/vtk/simple_orthgonal.vtk',np.c_[x,y,z],edge,1)

# shapes with more nodes, are not supported (yet)



# Let's make now a cone. We need the coordinates of the base, and the coordinate of the top
x=np.array([0,1,1,0])
y=np.array([0,0,1,1])
z=np.array([0,0,0,0])

top=np.array([0.5,0.5,1])
int1.make_pyramid('../data/vtk/pyramid.vtk',np.c_[x,y,z],top,1)



#Finally, let's make a cylinder.
# We need to define the center of the cylinder
x=0
y=0
z=0

# the radius of the cylinder
radius=1
# the height of the cylider. Since we are talking for subsurafe, z will alwys be "depth".
height=3
int1.make_cylinder('../data/vtk/cylinder.vtk',np.c_[x,y,z],radius,height,1)


