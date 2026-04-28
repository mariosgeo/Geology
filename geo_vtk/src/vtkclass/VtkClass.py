# -*- coding: utf-8 -*-
"""
VTK Geological Data Visualization Module

Comprehensive toolkit for converting geological and geophysical data to VTK format
for 3D visualization and analysis. Supports borehole logs, 3D geological models,
raster data (GeoTIFF), and various geological structures.

This module provides the VtkClass which serves as the main interface for creating
VTK files from geological data. VTK (Visualization Toolkit) format enables
interactive 3D visualization in software like ParaView, VisIt, and Mayavi.

Key Features
------------
- Borehole data visualization (cylinders, cubes, point clouds)
- 3D geological grid conversion with multiple properties
- GeoTIFF and raster data surface generation  
- Complex geological structure creation (pyramids, cylinders)
- Multi-property geological models with uncertainty visualization
- Surface interpolation from scattered geological data

Typical Workflow
----------------
1. Create VtkClass instance
2. Prepare geological data (coordinates, properties, classifications)
3. Use appropriate make_* method to convert to VTK format
4. Visualize in VTK-compatible software

Created on Thu Jan 23 09:29:48 2020
@author: karaouli

Examples
--------
>>> # Basic borehole visualization
>>> vtk = VtkClass()
>>> borehole_data = np.loadtxt('borehole.txt')
>>> vtk.make_borehole_as_cube_multi('borehole.vtk', borehole_data, radius=1.0)

>>> # 3D geological model
>>> grid_3d = np.random.rand(50, 40, 10)
>>> x_coords = np.linspace(0, 1000, 40)
>>> y_coords = np.linspace(0, 800, 50)  
>>> z_coords = np.linspace(100, 0, 11)
>>> vtk.make_3d_grid_to_vtk('model.vtk', grid_3d, x_coords, y_coords, z_coords)

>>> # GeoTIFF surface
>>> vtk.tiff_to_vtk('elevation.tif')
"""
import numpy as np
import pandas as pd
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

from osgeo import gdalnumeric
from osgeo import gdal
from scipy.interpolate import interp1d,interp2d,griddata,CloughTocher2DInterpolator


class VtkClass:
    """
    Comprehensive VTK (Visualization Toolkit) interface for geological data visualization.
    
    This class provides methods to convert various geological data formats into VTK files
    for 3D visualization and analysis. Supports point clouds, borehole data, 2D/3D grids,
    surfaces, TIFF/GeoTIFF files, and complex geological structures.
    
    The VTK format is widely used in scientific visualization and is compatible with
    software like ParaView, VisIt, and Mayavi for interactive 3D exploration of
    geological models.
    
    Key Capabilities
    ----------------
    - Borehole visualization as cylinders, cubes, or point sets
    - 2D/3D geological grid conversion with multiple properties
    - Surface and topographic data from TIFF/ASCII files
    - Scatter plot to surface interpolation
    - Multi-property geological models with uncertainty
    - Complex geological structures (pyramids, cylinders, custom shapes)
    
    Supported Input Formats
    -----------------------
    - NumPy arrays (points, grids, properties)
    - GeoTIFF raster files
    - ASCII grid files (.asc)
    - Pandas DataFrames (borehole logs)
    - XYZ coordinate files
    
    Examples
    --------
    >>> vtk = VtkClass()
    >>> # Convert borehole data to VTK
    >>> vtk.make_borehole_as_cube('borehole.vtk', data, radius=0.5)
    >>> # Convert grid to VTK
    >>> vtk.make_3d_grid_to_vtk('model.vtk', grid_data, x_coords, y_coords, z_coords)
    >>> # Convert GeoTIFF to VTK surface
    >>> vtk.tiff_to_vtk('surface.vtk', 'elevation.tif')
    """
    
    def __init__(self):
        """
        Initialize VTK class instance for geological data visualization.
        
        Creates a new VtkClass object ready for converting geological data
        into VTK format files. No parameters required - all functionality
        is provided through the various make_* and conversion methods.
        
        Notes
        -----
        This initialization sets up the class but does not load any data.
        Use the appropriate make_* methods to convert your geological data
        to VTK format.
        
        Examples
        --------
        >>> vtk = VtkClass()
        >>> # Now ready to convert geological data to VTK format
        """
        pass  # VtkClass is a utility class with static-like methods
        
    def make_xyz_points(self,filename,data):
        """
        Generate VTK point cloud from 3D coordinate data.
        
        Creates a VTK unstructured grid file containing individual points
        from geological survey data, sampling locations, or measurement points.
        Each point is rendered as a vertex in 3D space.

        Parameters
        ----------
        filename : str
            Output VTK filename (including .vtk extension).
        data : numpy.ndarray
            Array of 3D coordinates with shape (N, 3) where:
            - Column 0: X coordinates (easting)
            - Column 1: Y coordinates (northing) 
            - Column 2: Z coordinates (elevation/depth)

        Raises
        ------
        SystemExit
            If data array doesn't have exactly 3 columns.

        Notes
        -----
        - Creates VTK UNSTRUCTURED_GRID format with VERTEX cells
        - All points are rendered as individual vertices
        - Coordinate system should be consistent (e.g., UTM, local grid)
        - No properties or attributes are assigned to points

        Examples
        --------
        >>> # Create point cloud from borehole locations
        >>> locations = np.array([[100, 200, 50], [150, 250, 45], [200, 300, 40]])
        >>> vtk.make_xyz_points('survey_points.vtk', locations)
        
        >>> # Create point cloud from GPS coordinates
        >>> gps_data = np.loadtxt('survey.xyz')  # X, Y, Z columns
        >>> vtk.make_xyz_points('gps_points.vtk', gps_data)
        """
        if data.shape[1]!=3:
            sys.exit("ERROR: Currently, only nx3 size matrix are allowed")   
            
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('vtk output\n')
        file.write('ASCII\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d double\n'%data.shape[0])
        
        for k in range(0,data.shape[0]):
            file.write('%.3f %.3f %.3f\n'%(data[k,0],data[k,1],data[k,2]))
            
        file.write('CELLS  %d %d\n'%(data.shape[0],2*data.shape[0]))        
        
        for k in range(0,data.shape[0]):
            file.write('1 %d\n'%k)
        
        file.write('CELL_TYPES %d\n'%data.shape[0])        
        for k in range(0,data.shape[0]):
            file.write('1\n')        
        file.close()        
        return
        
        
        
    def make_1d_line(self,filename,data):
        """
        Provide an array to generate a 1D line

        :data: an Nx3 matrix with xyz coordinates
        """
        if data.shape[1]!=3:
            sys.exit("ERROR: Currently, only nx3 size matrix are allowed")   
            
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('vtk output\n')
        file.write('ASCII\n')
        file.write('DATASET POLYDATA\n')
        file.write('POINTS %d double\n'%data.shape[0])
        
        for k in range(0,data.shape[0]):
            file.write('%.3f %.3f %.3f\n'%(data[k,0],data[k,1],data[k,2]))
            
        file.write('LINES 1 %d\n'%(data.shape[0]+1))        
        file.write('%d '%data.shape[0])
        for k in range(0,data.shape[0]):
            file.write('%d '%k)
        
        file.close()
        return
        
    def make_2d_plane(self,filename,data,property_color=1):
        """
        Provide an array to generate a 2D pane, in 3D space
        If you are generating a plane in 2D space, assign zeros to the x3 dim

        :data: an Nx3 matrix with xyz coordinates
        : OPTIONAL property_color is an value (float) in case you want to plot sevesarl objects, and each to have a different color. 
        """
        if data.shape[1]!=3:
            sys.exit("ERROR: Currently, only nx3 size matrix are allowed")    
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('vtk output\n')
        file.write('ASCII\n')
        file.write('DATASET POLYDATA\n')
        file.write('POINTS %d double\n'%data.shape[0])
        
        for k in range(0,data.shape[0]):
            file.write('%.3f %.3f %.3f\n'%(data[k,0],data[k,1],data[k,2]))
            
        file.write('POLYGONS 1 %d\n'%(data.shape[0]+1))        
        file.write('%d '%data.shape[0])
        for k in range(0,data.shape[0]):
            file.write('%d '%k)
        file.write('\n')
        file.write('CELL_DATA %d\n'%(1))    
        file.write('SCALARS %s float\n'%'Property')    
        file.write('LOOKUP_TABLE custom_table\n')
        file.write('%.1f'%property_color)
        file.close() 
        
        return

    def make_3d_voxels(self,filename,data,property_color=1):
        """
        Provide an array to generate a 3D cube/orthogonal, in 3D space
        

        :data: an 8x3 matrix with xyz coordinates of the 8 nodes
        :property_color (OPTIONAL). If you want to draw multitple voxes, and each have different color, assign here a numer-->map
        """
        
        
        
        if data.shape[0]!=8:
            sys.exit("ERROR: Currently, only 8 nodes for voxels are supported")
            
            
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('A voxel\n')               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        
        file.write('POINTS %d float\n'%(8))
        
        x1=data[0,0]
        y1=data[0,1]
        z1=data[0,2]
            
        x2=data[1,0]
        y2=data[1,1]
        z2=data[1,2]
        
        x3=data[2,0]
        y3=data[2,1]
        z3=data[2,2]

        x4=data[3,0]
        y4=data[3,1]
        z4=data[3,2]        
        
        x5=data[4,0]
        y5=data[4,1]
        z5=data[4,2]

        x6=data[5,0]
        y6=data[5,1]
        z6=data[5,2]
        
        x7=data[6,0]
        y7=data[6,1]
        z7=data[6,2]        
        
        x8=data[7,0]
        y8=data[7,1]
        z8=data[7,2]


            
        file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                       (x1,y1,z1,
                        x2,y2,z2,
                        x3,y3,z3,
                        x4,y4,z4,
                        x5,y5,z5,
                        x6,y6,z6,
                        x7,y7,z7,
                        x8,y8,z8))
        file.write('\n')
        
        
        
        file.write('CELLS %d %d\n'%(1,9))
        base=np.array([4,7,6,5,0,3,2,1])
        for i in range(0,1):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+i*8,base[1]+i*8,base[2]+i*8,base[3]+i*8,base[4]+i*8,base[5]+i*8,base[6]+i*8,base[7]+i*8))
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(1))    
        for i in range(0,1):
            file.write('12\n')
                     
        file.write('\n')
        file.write('CELL_DATA %d\n'%(1))    
        file.write('SCALARS %s float\n'%'PUMP_ON_OFF')    
        file.write('LOOKUP_TABLE custom_table\n')
        file.write('%.1f'%property_color)
        file.close()
    

        return
    
    def make_simple_3d_voxel(self,filename,data,radius,property_color=1):
        """
        Provide an array to generate a 3D/orthogonal, in 3D space
        

        :data: an 1x3 matrix with xyz coordinates of the center
        :radius: if float, then we have a cube of size radius.
                if (3,) then we have orthogonal of size (x,y,z)
        :property_color (OPTIONAL). If you want to draw multitple voxes, and each have different color, assign here a numer-->map
        """
        
        if (np.size(radius)!=1) & (np.size(radius)!=3):
            sys.exit("ERROR: Wrong radius input. Is has to be either float or array of (3,)")
        
        if (data.shape[0]!=3) & (data.shape[1]!=3):
            sys.exit("ERROR: Currently, only 1x3 size matrix are allowed")   
        
        if np.size(radius)==1:
            edge_x=radius/2
            edge_y=radius/2
            edge_z=radius/2
        else:
            edge_x=radius[0]/2
            edge_y=radius[1]/2
            edge_z=radius[2]/2
            
        xc=data[0,0]
        yc=data[0,1]
        zc=data[0,2]
        
                
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('A voxel\n')               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        
        file.write('POINTS %d float\n'%(8))
        
        x1=xc-edge_x
        y1=yc-edge_y
        z1=zc-edge_z
            
        x2=xc+edge_x
        y2=yc+edge_y
        z2=zc+edge_z
            
        file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                       (x1,y1,z1,
                        x1,y2,z1,
                        x2,y2,z1,
                        x2,y1,z1,
                        x1,y1,z2,
                        x1,y2,z2,
                        x2,y2,z2,
                        x2,y1,z2))
        file.write('\n')
        
        
        
        file.write('CELLS %d %d\n'%(1,9))
        base=np.array([4,7,6,5,0,3,2,1])
        for i in range(0,1):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+i*8,base[1]+i*8,base[2]+i*8,base[3]+i*8,base[4]+i*8,base[5]+i*8,base[6]+i*8,base[7]+i*8))
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(1))    
        for i in range(0,1):
            file.write('12\n')
                     
        file.write('\n')
        file.write('CELL_DATA %d\n'%(1))    
        file.write('SCALARS %s float\n'%'PUMP_ON_OFF')    
        file.write('LOOKUP_TABLE custom_table\n')
        file.write('%.1f'%property_color)
        file.close()
    

        return



    def make_pyramid(self,filename,data,top,property_color=1):
        """
        Provide an array to generate a 3D/orthogonal, in 3D space
        

        :data: an 4x3 matrix with xyz coordinates of the corner of the base
        :top: xyz coordinate of the top of pyramid
        :property_color (OPTIONAL). If you want to draw multitple voxes, and each have different color, assign here a numer-->map
        """
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('vtk output\n')
        file.write('ASCII\n')
        file.write('DATASET POLYDATA\n')
        file.write('POINTS %d double\n'%(5))
        file.write('%.5f %.5f %.5f\n'%(top[0],top[1],top[2]))
        for i in range(0,4):
            file.write('%.5f %.5f %.5f\n'%(data[i,0],data[i,1],data[i,2]))
        
        
        file.write('POLYGONS 5 21\n')
        file.write('4 4 3 2 1\n') 
        file.write(' 3 0 1 2\n') 
        file.write(' 3 0 2 3\n') 
        file.write('3 0 3 4\n') 
        file.write('3 0 4 1\n') 
        
        file.write('\n')
        file.write('CELL_DATA %d\n'%5) 
        file.write('SCALARS %s float\n'%'Property')    
        file.write('LOOKUP_TABLE custom_table\n')
        for i in range(0,5):
            file.write('%.3f\n'%property_color)
        
        
        return


    def make_cylinder(self,filename,center,radius=1,height=1,property_color=1,resolution=256,offset=np.array(([0,0]))):
        """
        Provide center, radius, height, to make a cylinder in in 3D space
        

        :center 1x3 (XYZ) center of cylinder
        :radius radius of cylinder default(1)
        :height height of cylinder default(1)
        :property_color (OPTIONAL). If you want to draw multitple voxes, and each have different color, assign here a numer-->map
        :resolution (OPTIONAL): defalut 256. Condirer smaller number in case rendering is slow
        :offset (OPTIONAL): titled cylinder, 1x2 (XY) offest of bottom surface, with respect top surface
        """        
        
        
        if (center.shape[1]!=3):
            sys.exit("ERROR: Currently, only 1x3 size matrix are allowed")   
            
        xc=center[0,0]
        yc=center[0,1]
        zc=center[0,2]
        t=np.linspace(0,360,resolution+1)
        
        x=np.zeros(t.shape)
        y=np.zeros(t.shape)
        for i in range (0,t.shape[0]):
            x[i]=(0.5*np.abs(radius)*np.cos(t[i]*np.pi/180)+0)
            y[i]=(0.5*np.abs(radius)*np.sin(t[i]*np.pi/180)+0)
        
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('vtk output\n')
        file.write('ASCII\n')
        file.write('DATASET POLYDATA\n')
        file.write('POINTS %d double\n'%(4*resolution))
        
        for i in range(0,resolution):
            file.write('%.5f %.5f %.5f\n'%(x[i]+xc,yc+y[i],zc))
            file.write('%.5f %.5f %.5f\n'%(x[i]+xc+offset[0],yc+y[i]+offset[1],zc-np.abs(height)))
        
        for i in range(0,resolution):
            file.write('%.5f %.5f %.5f\n'%(x[i]+xc,yc+y[i],zc))

        for i in range(0,resolution):
            file.write('%.5f %.5f %.5f\n'%(x[i]+offset[0]+xc,yc+offset[1]+y[i],zc-height))            
        
        file.write('POLYGONS %d %d\n'%(resolution+2,5*resolution+2*resolution+2 ))  
        base=np.array([0,1,3,2])
        for i in range(0,resolution-1):
            file.write('4 %d %d %d %d\n'%(base[0],base[1],base[2],base[3]))
            base=base+2
        file.write('4 %d %d %d %d\n'%(base[0],base[1],1,0))
        file.write('%d '%resolution)
        for i in range(0,resolution):
            file.write('%d '%(2*resolution+i))
        file.write('\n')
        file.write('%d '%resolution)
        for i in range(0,resolution):
            file.write('%d '%(3*resolution+i))
        
        file.write('\n')
        file.write('CELL_DATA %d\n'%((resolution+2)))    
        file.write('SCALARS %s float\n'%'Property')    
        file.write('LOOKUP_TABLE custom_table\n')
        for i in range(0,resolution+2):
            file.write('%.3f\n'%property_color)
        
        
        file.close()
        return
 
            
        
    def make_borehole_as_cube(self,filename,data,center,radius,elev,name=['property_0']):
        """
        Provide center (on 2d plane), radius, height, to make a cylinder in in 3D space
        
        :data an nXM matrix, with depth,  and class number (s). 
        :center 1x3 (XYZ) center of cylinder
        :radius (int) radius of cylinder
        :height (int) height of cylinder
        :property_color (OPTIONAL). If you want to draw multitple voxes, and each have different color, assign here a numer-->map
        :resolution (OPTIONAL): defalut 256. Condirer smaller number in case rendering is slow
        :offset (OPTIONAL): titled cylinder, 1x2 (XY) offest of bottom surface, with respect top surface
        """
        
        if data.shape[1]<2:
            sys.exit("ERROR: data should be at least nx2 size")
            
        no_layers=data.shape[0]
        no_properties=data.shape[1]-1 # in case you have more properies to plot ber borehole
        
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(8*(no_layers)))
        
        for i in range(0,no_layers):
            
            if i==0:
                up=data[i,0]-(data[i+1,0]-data[i,0])
                down=data[i,0]
            elif (i>0) & (i<no_layers-1):
                up=data[i-1,0]
                down=data[i,0]
            elif i==no_layers-1:
                up=data[i-1,0]
                down=1.2*data[i-1,0]
            
            up=-(up-elev)
            down=-(down-elev)
            
            x1=center[0]-radius/2
            y1=center[1]-radius/2
            z1=up
            
            x2=center[0]+radius/2
            y2=center[1]+radius/2
            z2=down
            
            file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                       (x1,y1,z1,
                        x1,y2,z1,
                        x2,y2,z1,
                        x2,y1,z1,
                        x1,y1,z2,
                        x1,y2,z2,
                        x2,y2,z2,
                        x2,y1,z2))
        file.write('\n')
        file.write('CELLS %d %d\n'%(no_layers,9*(no_layers)))
        base=np.array([4,7,6,5,0,3,2,1])
        for i in range(0,data.shape[0]):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+i*8,base[1]+i*8,base[2]+i*8,base[3]+i*8,base[4]+i*8,base[5]+i*8,base[6]+i*8,base[7]+i*8))
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(no_layers))    
        for i in range(0,no_layers):
            file.write('12\n')
                     
        file.write('\n')
        file.write('CELL_DATA %d\n'%(no_layers))    
        
        for k in range(0,no_properties):
            file.write('SCALARS %s float\n'%name[k])    
            file.write('LOOKUP_TABLE custom_table\n')    
            for i in range(0,no_layers):
                file.write('%.2f\n'%data[i,k+1])
        # if data.shape[1]>4:
        #         file.write('SCALARS %s float\n'%name2)    
        #         file.write('LOOKUP_TABLE custom_table\n')    
        #         for i in range(0,data.shape[0]-1):
        #             file.write('%.2f\n'%data[i,4])
        # if data.shape[1]>5:
        #         file.write('SCALARS %s float\n'%name3)    
        #         file.write('LOOKUP_TABLE custom_table\n')    
        #         for i in range(0,data.shape[0]-1):
        #             file.write('%.2f\n'%data[i,5])        
           
    
    
        
        file.close()
    
        return

    def make_borehole_as_cube_multi(self,filename,data,radius,label=np.arange(0,100,1)):
        """
        Create VTK visualization of multiple boreholes as stacked cubes.
        
        Converts borehole geological logs into 3D cube visualization where each
        geological layer is represented as a colored cube. Suitable for displaying
        multiple boreholes with varying geological formations and depths.

        Parameters
        ----------
        filename : str
            Output VTK filename (including .vtk extension).
        data : numpy.ndarray
            Borehole data array with shape (N, M) where:
            - Column 0: Depth values (increasing downward)
            - Column 1: X coordinates of borehole
            - Column 2: Y coordinates of borehole  
            - Column 3: Z elevation of surface
            - Column 4: Geological unit classification
            - Columns 5+: Additional properties (optional)
            Minimum required shape is (N, 2) for depth and classification.
        radius : float
            Half-width of cube representation for each geological layer.
            Controls the visual size of borehole in 3D space.
        label : numpy.ndarray, optional
            Array of geological unit labels for property mapping.
            Default is np.arange(0, 100, 1).

        Raises
        ------
        SystemExit
            If data array has fewer than 2 columns.

        Notes
        -----
        - Each geological layer becomes a hexahedron (cube) cell in VTK
        - Layer thickness determined by depth differences between samples
        - Geological units assigned as cell properties for coloring
        - Supports multiple properties per borehole layer
        - Coordinates should be in consistent units (meters recommended)

        Examples
        --------
        >>> # Create multiple borehole visualization
        >>> borehole_data = np.array([
        ...     [0.0, 100, 200, 50, 1],    # surface, sandy clay
        ...     [2.0, 100, 200, 50, 2],    # clay layer
        ...     [5.0, 100, 200, 50, 3],    # sand layer
        ... ])
        >>> vtk.make_borehole_as_cube_multi('boreholes.vtk', borehole_data, radius=1.0)
        
        >>> # With custom geological unit labels
        >>> units = np.array([1, 2, 3, 4, 5])  # Formation codes
        >>> vtk.make_borehole_as_cube_multi('geology.vtk', data, 0.5, label=units)
        """
        
        if data.shape[1]<2:
            sys.exit("ERROR: data should be at least nx2 size")
            
        no_layers=data.shape[0]
        no_properties=data.shape[1]-5 # in case you have more properies to plot ber borehole
        
        # radius=data[:,5]
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(8*(no_layers)))
        
        for i in range(0,no_layers):
            
            elev=data[i,4]
            
            up=data[i,0]
            down=data[i,1]
            
            
            up=-(up-elev)
            down=-(down-elev)
            
            x1=data[i,2]-radius/2
            y1=data[i,3]-radius/2
            z1=up
            
            x2=data[i,2]+radius/2
            y2=data[i,3]+radius/2
            z2=down
            if np.isnan(z1):
                print(i)
            if np.isnan(z2):
                print(i)            
            file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                       (x1,y1,z1,
                        x1,y2,z1,
                        x2,y2,z1,
                        x2,y1,z1,
                        x1,y1,z2,
                        x1,y2,z2,
                        x2,y2,z2,
                        x2,y1,z2))
        file.write('\n')
        file.write('CELLS %d %d\n'%(no_layers,9*(no_layers)))
        base=np.array([4,7,6,5,0,3,2,1])
        for i in range(0,data.shape[0]):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+i*8,base[1]+i*8,base[2]+i*8,base[3]+i*8,base[4]+i*8,base[5]+i*8,base[6]+i*8,base[7]+i*8))
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(no_layers))    
        for i in range(0,no_layers):
            file.write('12\n')
                     
        file.write('\n')
        file.write('CELL_DATA %d\n'%(no_layers))    
        
        for k in range(0,no_properties):
            file.write('SCALARS %s float\n'%(label[k]))    
            file.write('LOOKUP_TABLE custom_table\n')    
            for i in range(0,no_layers):
                file.write('%.2f\n'%data[i,k+5])

           
    
    
        
        file.close()
    
        return



    def make_borehole_as_cube_multi_perc(self,filename,data,radius,label=np.arange(0,100,1)):
        """
        Provide center (on 2d plane), radius, height, to make a cylinder in in 3D space
        
        :data an nXM matrix, with depth,  and class number (s). 
        :center 1x3 (XYZ) center of cylinder
        :radius (int) radius of cylinder
        :height (int) height of cylinder
        :property_color (OPTIONAL). If you want to draw multitple voxes, and each have different color, assign here a numer-->map
        :resolution (OPTIONAL): defalut 256. Condirer smaller number in case rendering is slow
        :offset (OPTIONAL): titled cylinder, 1x2 (XY) offest of bottom surface, with respect top surface
        """
        
        if data.shape[1]<2:
            sys.exit("ERROR: data should be at least nx2 size")
            
        no_layers=data.shape[0]
        no_properties=1 # in case you have more properies to plot ber borehole
        
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(6*8*(no_layers)))
        
        for i in range(0,no_layers):
            
            elev=data[i,4]
            
            up=data[i,0]
            down=data[i,1]
            
            
            up=-(up-elev)
            down=-(down-elev)
            
            
            #here scale on geology
            # find which non zeror element exist g,s,c,l,p
            # 'gravel_component','sand_component', 'clay_component', 'loam_component', 'peat_component','silt_component'
            #clay silt sand gravel organic nothing
        
            x1=data[i,2]-radius/2
            y1=data[i,3]-radius/2
            
            x2=data[i,2]+radius/2
            y2=data[i,3]+radius/2
            
            for k in range(0,6):
                
                x2=x1+data[i,k+5]*radius/2
                y2=y1+data[i,k+5]*radius/2
                z1=up
                

                z2=down
                
                
          
                file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                           (x1,y1,z1,
                            x1,y2,z1,
                            x2,y2,z1,
                            x2,y1,z1,
                            x1,y1,z2,
                            x1,y2,z2,
                            x2,y2,z2,
                            x2,y1,z2))
                
                x1=x2
                y1=y2
                
                
                
        file.write('\n')
        file.write('CELLS %d %d\n'%(6*no_layers,6*9*(no_layers)))
        base=np.array([4,7,6,5,0,3,2,1])
        for i in range(0,6*data.shape[0]):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+i*8,base[1]+i*8,base[2]+i*8,base[3]+i*8,base[4]+i*8,base[5]+i*8,base[6]+i*8,base[7]+i*8))
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(6*no_layers))    
        for i in range(0,6*no_layers):
            file.write('12\n')
                     
        file.write('\n')
        file.write('CELL_DATA %d\n'%(6*no_layers))    
        
        for k in range(0,no_properties):
            file.write('SCALARS %s float\n'%(label[k]))    
            file.write('LOOKUP_TABLE custom_table\n')    
            for i in range(0,no_layers):
                for k in range(0,6):
                    file.write('%.2f\n'%k)

           
    
    
        
        file.close()
    
        return

    def make_dts_from_xyz(self,filename,data,label=np.arange(0,100,1)):
        """
        Provide center (on 2d plane), radius, height, to make a cylinder in in 3D space
        
        :data an nXM matrix, with depth,  and class number (s). 
        :center 1x3 (XYZ) center of cylinder
        :radius (int) radius of cylinder
        :height (int) height of cylinder
        :property_color (OPTIONAL). If you want to draw multitple voxes, and each have different color, assign here a numer-->map
        :resolution (OPTIONAL): defalut 256. Condirer smaller number in case rendering is slow
        :offset (OPTIONAL): titled cylinder, 1x2 (XY) offest of bottom surface, with respect top surface
        """
        
        if data.shape[1]<2:
            sys.exit("ERROR: data should be at least nx2 size")
            
        no_layers=data.shape[0]
        no_properties=data.shape[1]-3 # in case you have more properies to plot ber borehole
        
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(8*(no_layers)))
        
        nx=np.abs(np.median(np.diff(data[:,0])))
        ny=np.abs(np.median(np.diff(data[:,1])))
        nz=np.abs(np.median(np.diff(data[:,2])))
        if nz==0:
            nz=5*np.max(np.c_[nx,ny])
        if nx==0:
            nx=1*np.max(np.c_[nz,ny])                     
        if ny==0:
            ny=1*np.max(np.c_[nx,nz])
        # nx=0.5
        # ny=0.5
        
            
        
        for i in range(0,no_layers):
            
            
            
            
            z1=(data[i,2]-nz/2)
            z2=(data[i,2]+nz/2           )


            # if i==0:
            #     x1=data[i,0]-nx/2                
            #     x2=0.5*(data[i+1,0]+data[i,0])
            #     y1=data[i,1]-ny/2                
            #     y2=0.5*(data[i+1,1]+data[i,1])                
            # elif i==no_layers-1:
            #     x1=0.5*(data[i-1,0]+data[i,0]) 
            #     x2=data[i,0]+ny/2
            #     y1=0.5*(data[i-1,1]+data[i,1]) 
            #     y2=data[i,1]+ny/2                
            # else:
            #     x1=0.5*(data[i-1,0]+data[i,0]) 
            #     x2=0.5*(data[i+1,0]+data[i,0])
            
            x1=data[i,0]-nx/2            
            x2=data[i,0]+nx/2            
            y1=data[i,1]-ny/2            
            y2=data[i,1]+ny/2                        
            
            
           
            file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                       (x1,y1,z1,
                        x1,y2,z1,
                        x2,y2,z1,
                        x2,y1,z1,
                        x1,y1,z2,
                        x1,y2,z2,
                        x2,y2,z2,
                        x2,y1,z2))
        file.write('\n')
        file.write('CELLS %d %d\n'%(no_layers,9*(no_layers)))
        base=np.array([4,7,6,5,0,3,2,1])
        for i in range(0,data.shape[0]):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+i*8,base[1]+i*8,base[2]+i*8,base[3]+i*8,base[4]+i*8,base[5]+i*8,base[6]+i*8,base[7]+i*8))
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(no_layers))    
        for i in range(0,no_layers):
            file.write('12\n')
                     
        file.write('\n')
        file.write('CELL_DATA %d\n'%(no_layers))    
        
        for k in range(0,no_properties):
            file.write('SCALARS Property_%d float\n'%k)    
            file.write('LOOKUP_TABLE custom_table\n')    
            for i in range(0,no_layers):
                file.write('%.2f\n'%data[i,k+3])

           
    
    
        
        file.close()
    
        return


    def make_cylinder_borehole(self,filename,data,center,radius,elev=0,resolution=256,offset=np.array(([0,0]))):
        """
        Provide center (on 2d plane), radius, height, to make a cylinder in in 3D space
        
        :data an nXM matrix, with depth,  and class number (s). 
        :center 1x3 (XYZ) center of cylinder
        :radius (int) radius of cylinder
        :height (int) height of cylinder
        :property_color (OPTIONAL). If you want to draw multitple voxes, and each have different color, assign here a numer-->map
        :resolution (OPTIONAL): defalut 256. Condirer smaller number in case rendering is slow
        :offset (OPTIONAL): titled cylinder, 1x2 (XY) offest of bottom surface, with respect top surface
        """
        
        if data.shape[1]<2:
            sys.exit("ERROR: data should be at least nx2 size")
            
        no_layers=data.shape[0]
        no_properties=data.shape[1]-1 # in case you have more properies to plot ber borehole
        # no_layers=2
        xc=center[0]
        yc=center[1]
        # zc=center[2]
        t=np.linspace(0,360,resolution+1)
        
        x=np.zeros(t.shape)
        y=np.zeros(t.shape)
        for i in range (0,t.shape[0]):
            x[i]=(0.5*np.abs(radius)*np.cos(t[i]*np.pi/180)+0)
            y[i]=(0.5*np.abs(radius)*np.sin(t[i]*np.pi/180)+0)
        
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('vtk output\n')
        file.write('ASCII\n')
        file.write('DATASET POLYDATA\n')
        file.write('POINTS %d double\n'%(4*resolution*(no_layers)))
        
        
        for k in range(0,no_layers):
            
            
            if k==0:
                up=0
                down=data[k,0]
            elif (k>0) & (k<no_layers):
                up=data[k-1,0]
                down=data[k,0]
            elif k==no_layers:
                up=data[k,0]
                down=1.2*data.iloc[k,0]
            
            up=-(up-elev)
            down=-(down-elev)
            for i in range(0,resolution):
                file.write('%.5f %.5f %.5f\n'%(x[i]+xc,yc+y[i],up))
                file.write('%.5f %.5f %.5f\n'%(x[i]+xc+offset[0],yc+y[i]+offset[1],down))
            
            for i in range(0,resolution):
                file.write('%.5f %.5f %.5f\n'%(x[i]+xc,yc+y[i],up))
    
            for i in range(0,resolution):
                file.write('%.5f %.5f %.5f\n'%(x[i]+offset[0]+xc,yc+offset[1]+y[i],down))            
        
        
        
        
        # no_layers=1
        file.write('POLYGONS %d %d\n'%((no_layers)*(resolution)+2*(no_layers),(no_layers)*(5*resolution)+2*(no_layers)*resolution+2*(no_layers)) )  
        base=np.array([0,1,3,2])
        for k in range(0,no_layers):
            for i in range(0,resolution-1):
                file.write('4 %d %d %d %d\n'%(base[0],base[1],base[2],base[3]))
                base=base+2
            file.write('4 %d %d %d %d\n'%(base[0],base[1],(k)*(4*resolution)+1,(k)*(4*resolution)))
            file.write('%d '%(resolution))
            for i in range(0,resolution):
                    file.write('%d '%((2+4*k)*resolution+i))
            file.write('\n')
            file.write('%d '%(resolution))
            for i in range(0,resolution):
                file.write('%d '%(((3+4*k)*resolution+i)))
            file.write('\n')

            base=base+2*resolution+2
        
       
            
        file.write('\n')
        file.write('CELL_DATA %d\n'%(no_layers*(resolution+2))) 
        for ii in range(0,no_properties):
            file.write('SCALARS IC_%d float\n'%ii)    
            file.write('LOOKUP_TABLE custom_table\n')
            for k in range(0,no_layers):
                for i in range(0,resolution+2):
                    file.write('%.3f\n'%data[k,ii+1])    

        
        
        file.close()
        return



    def make_cylinder_borehole_deviation(self,filename,data,center,radius,dev,elev=0,resolution=256):
        """
        Provide center (on 2d plane), radius, height, to make a cylinder in in 3D space
        
        :data an nXM matrix, with depth,  and class number (s). 
        :center 1x3 (XYZ) center of cylinder
        :radius (int) radius of cylinder
        :height (int) height of cylinder
        :dev Nx2 matrix with deviation x,y
        :property_color (OPTIONAL). If you want to draw multitple voxes, and each have different color, assign here a numer-->map
        :resolution (OPTIONAL): defalut 256. Condirer smaller number in case rendering is slow
        :offset (OPTIONAL): titled cylinder, 1x2 (XY) offest of bottom surface, with respect top surface
        """
        
        if data.shape[1]<2:
            sys.exit("ERROR: data should be at least nx2 size")
            
        no_layers=data.shape[0]
        no_properties=data.shape[1]-1 # in case you have more properies to plot ber borehole
        # no_layers=2
        xc=center[0]
        yc=center[1]
        # zc=center[2]
        t=np.linspace(0,360,resolution+1)
        
        x=np.zeros(t.shape)
        y=np.zeros(t.shape)
        for i in range (0,t.shape[0]):
            x[i]=(0.5*np.abs(radius)*np.cos(t[i]*np.pi/180)+0)
            y[i]=(0.5*np.abs(radius)*np.sin(t[i]*np.pi/180)+0)
        
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('vtk output\n')
        file.write('ASCII\n')
        file.write('DATASET POLYDATA\n')
        file.write('POINTS %d double\n'%(4*resolution*(no_layers)))
        
        # Add an extra layer
        # dev=np.r_[dev,[dev[-1,:]]]
        
        for k in range(0,no_layers):

            
            
            if k==0:
                up=0
                down=data[k,0]
            elif (k>0) & (k<no_layers):
                up=data[k-1,0]
                down=data[k,0]
            elif k==no_layers:
                up=data[k,0]
                down=1.2*data.iloc[k,0]
            
            up=-(up-elev)
            down=-(down-elev)
            for i in range(0,resolution):
                file.write('%.5f %.5f %.5f\n'%(x[i]+xc+dev[k,0],yc+y[i]+dev[k,1],up))
                file.write('%.5f %.5f %.5f\n'%(x[i]+xc+dev[k+1,0],yc+y[i]+dev[k+1,1],down))
            
            for i in range(0,resolution):
                file.write('%.5f %.5f %.5f\n'%(x[i]+xc+dev[k,0],yc+y[i]+dev[k,1],up))
    
            for i in range(0,resolution):
                file.write('%.5f %.5f %.5f\n'%(x[i]+dev[k+1,0]+xc,yc+dev[k+1,1]+y[i],down))            
        
        
        
        
        # no_layers=1
        file.write('POLYGONS %d %d\n'%((no_layers)*(resolution)+2*(no_layers),(no_layers)*(5*resolution)+2*(no_layers)*resolution+2*(no_layers)) )  
        base=np.array([0,1,3,2])
        for k in range(0,no_layers):
            for i in range(0,resolution-1):
                file.write('4 %d %d %d %d\n'%(base[0],base[1],base[2],base[3]))
                base=base+2
            file.write('4 %d %d %d %d\n'%(base[0],base[1],(k)*(4*resolution)+1,(k)*(4*resolution)))
            file.write('%d '%(resolution))
            for i in range(0,resolution):
                    file.write('%d '%((2+4*k)*resolution+i))
            file.write('\n')
            file.write('%d '%(resolution))
            for i in range(0,resolution):
                file.write('%d '%(((3+4*k)*resolution+i)))
            file.write('\n')

            base=base+2*resolution+2
        
       
            
        file.write('\n')
        file.write('CELL_DATA %d\n'%(no_layers*(resolution+2))) 
        for ii in range(0,no_properties):
            file.write('SCALARS Property_%d float\n'%ii)    
            file.write('LOOKUP_TABLE custom_table\n')
            for k in range(0,no_layers):
                for i in range(0,resolution+2):
                    file.write('%.3f\n'%data[k,ii+1])    

        
        
        file.close()
        return
        
    
    def tiff_to_vtk_3d(self,filename):
        """
        Provide a geotiff and get a dem model with the same name as vtk file. 
        Notice, in this version we do no support cooridante conversion. Future versions will

        """        
        
        
        raster = gdal.Open(filename)
        ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
        x_coords = ulx + np.arange(0,raster.RasterXSize) * xres +  (xres / 2) #add half the cell size
        y_coords = uly + np.arange(0,raster.RasterYSize) * yres +  (yres / 2) #add half the cell size
        
        y_coords=np.flipud(y_coords)
        
        Xc, Yc = np.meshgrid(x_coords,y_coords);
        band1 = raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
        band1[band1>1e6]=np.nan
        band1[band1<-1e6]=np.nan
        
        # band1=pd.DataFrame(band1)
        # band1=band1.interpolate().values
        # band1[np.isnan(band1)]=np.nanmin(band1)

            
        
        
        if np.abs(xres)!=np.abs(yres):
            sys.exit('Error. dx and dy are different. Adjust the code\\')
        else:    
            dx=xres
            # plt.imshow(band1)
            band1=band1.T
            number_elements=np.count_nonzero(np.isfinite(band1.ravel()))
            
            filename=filename[:-4]+'_3d.vtk'
            file=open(filename,'w')
            file.write('# vtk DataFile Version 1.0\n')
            file.write('%s\n'%filename)               
            file.write('ASCII\n')
            file.write('\n')
            file.write('DATASET UNSTRUCTURED_GRID\n')
            file.write('POINTS %d float\n'%(8*(number_elements)))
            
            
            for i in range(0,x_coords.shape[0]):
                for j in range(0,y_coords.shape[0]):
                    if np.isfinite(band1[i,j]):
                        x1=x_coords[i]-dx/2
                        y1=y_coords[j]-dx/2
                        z1=band1[i,j]
                        
                        x2=x_coords[i]+dx/2
                        y2=y_coords[j]+dx/2
                        z2=band1[i,j]-1
                        # z2=np.nanmin(band1.ravel())-1
                        
                        file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                                   (x1,y1,z1,
                                    x1,y2,z1,
                                    x2,y2,z1,
                                    x2,y1,z1,
                                    x1,y1,z2,
                                    x1,y2,z2,
                                    x2,y2,z2,
                                    x2,y1,z2))
                    
            file.write('\n')
            file.write('CELLS %d %d\n'%(number_elements,9*(number_elements)))
            base=np.array([4,7,6,5,0,3,2,1])
            k=0
            for i in range(0,x_coords.shape[0]):
                for j in range(0,y_coords.shape[0]):
                    if np.isfinite(band1[i,j]):
                        file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+k*8,base[1]+k*8,base[2]+k*8,base[3]+k*8,base[4]+k*8,base[5]+k*8,base[6]+k*8,base[7]+k*8))
                        k=k+1
            file.write('\n')
            file.write('CELL_TYPES %d\n'%(number_elements))    
            for i in range(0,number_elements):
                file.write('12\n')
                    
            file.write('\n')
            file.write('CELL_DATA %d\n'%(number_elements))    
            file.write('SCALARS Bathymetry float\n') 
            file.write('LOOKUP_TABLE custom_table\n')    
            for i in range(0,x_coords.shape[0]):
                for j in range(0,y_coords.shape[0]):
                    if np.isfinite(band1[i,j]):
                        file.write('%.2f\n'%band1[i,j])
            
            
            
            file.close()          
            return


    def tiff_to_vtk(self,filename,xrange=None,yrange=None):
        """
        Convert GeoTIFF raster data to VTK structured grid surface.
        
        Reads geospatially referenced TIFF files (elevation models, geological maps,
        geophysical data) and creates a VTK surface for 3D visualization. Automatically
        handles coordinate transformations and missing data interpolation.

        Parameters
        ----------
        filename : str
            Path to input GeoTIFF file. Should contain geospatial metadata
            for proper coordinate extraction.
        xrange : list or tuple, optional
            Range of X-coordinates to extract as [xmin, xmax].
            If None, uses full extent of raster. Default is None.
        yrange : list or tuple, optional
            Range of Y-coordinates to extract as [ymin, ymax]. 
            If None, uses full extent of raster. Default is None.

        Notes
        -----
        - Automatically extracts coordinates from GeoTIFF geotransform
        - Handles coordinate systems with negative Y-resolution (standard for GIS)
        - Interpolates missing data (NaN values) using pandas interpolation
        - Values > 1e6 are treated as no-data and converted to NaN
        - Creates VTK STRUCTURED_GRID format suitable for surface visualization
        - Output filename derived from input filename with .vtk extension

        File Format Support
        -------------------
        - GeoTIFF (.tif, .tiff) with embedded geospatial metadata
        - Single-band raster data (elevation, properties, classifications)
        - Standard GIS coordinate reference systems

        Examples
        --------
        >>> # Convert full DEM to VTK
        >>> vtk.tiff_to_vtk('elevation.tif')
        
        >>> # Convert subset of geological map
        >>> vtk.tiff_to_vtk('geology.tif', xrange=[100000, 200000], 
        ...                yrange=[4000000, 4100000])
        
        >>> # Convert geophysical survey data
        >>> vtk.tiff_to_vtk('magnetic_data.tif')

        See Also
        --------
        tiff_to_vtk_3d : For 3D volumetric TIFF data
        tiff_to_vtk_as_data : For multi-band TIFF data
        """        
        
        
        raster = gdal.Open(filename)
        ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
        x_coords = ulx + np.arange(0,raster.RasterXSize) * xres +  (xres / 2) #add half the cell size
        y_coords = uly + np.arange(0,raster.RasterYSize) * yres +  (yres / 2) #add half the cell size
        
        if yres<0:
            y_coords=np.flipud(y_coords)
        
        
        rgb = raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
        rgb[rgb>1e6]=np.nan
        rgb[rgb==0]=np.nan
        
        rgb=pd.DataFrame(rgb)
        rgb=rgb.interpolate().ffill().bfill().values
        Xc, Yc = np.meshgrid(x_coords,y_coords);

        # # remove finall nan (typically in the cor
        # test=np.where(np.isnan(rgb))
        # for i in range(0,len(test[0])):
        #     if (test[1][i]<rgb.shape[1])& (test[1][i]>0):
        #         rgb[test[0][i],test[1][i]]=0.5*(rgb[test[0][i],test[1][i]-1]+ rgb[test[0][i],test[1][i]+1])
        #     elif test[1][i]==0:
        #         rgb[test[0][i],test[1][i]]=(rgb[test[0][i],test[1][i]+1])
        #     elif test[1][i]==rgb.shape[1]:
        #         rgb[test[0][i],test[1][i]]=(rgb[test[0][i],test[1][i]-1])
        # test=np.where(np.isnan(rgb))
        # for i in range(0,len(test[0])):
        #     if (test[0][i]<rgb.shape[0] & (test[0][i]>0)):
        #         rgb[test[0][i],test[1][i]]=0.5*(rgb[test[0][i]-1,test[1][i]]+ rgb[test[0][i]+1,test[1][i]])
        #     elif test[0][i]==0:
        #         rgb[test[0][i],test[1][i]]=(rgb[test[0][i]+1,test[1][i]])
        #     elif test[0][i]==rgb.shape[0]:
        #         rgb[test[0][i],test[1][i]]=(rgb[test[0][i]-1,test[1][i]])
        
        rgb[np.isnan(rgb)]=np.nanmin(rgb)
        rgb=np.flipud(rgb)
        
        if xrange is not None:
            ix=np.where((x_coords>=xrange[0]) &
                        (x_coords<=xrange[1]) 
                        )[0]
        else:
            ix=np.arange(x_coords.shape[0])
        
        if yrange is not None:
            iy=np.where((y_coords>=yrange[0]) &
                        (y_coords<=yrange[1]) 
                        )[0]
        else:
            iy=np.arange(y_coords.shape[0])

        
        rgb=rgb[iy[:,None],ix[None,:]]
        x_coords=x_coords[ix]
        y_coords=y_coords[iy]
        Xc, Yc = np.meshgrid(x_coords,y_coords);
        
        
        
        
        
        
        
        if np.abs(xres)!=np.abs(yres):
            sys.exit('Error. dx and dy are different. Adjust the code\\')
        else:    
            dx=xres
            # plt.imshow(band1)
            
            number_elements=np.count_nonzero(np.isfinite(rgb.ravel()))


            base=np.zeros((rgb.shape[1]-1,4))
            base[:,0]=np.arange(0,rgb.shape[1]-1)
            base[:,1]=np.arange(1,rgb.shape[1])
            base[:,2]=np.arange(1,rgb.shape[1])+rgb.shape[1]
            base[:,3]=np.arange(0,rgb.shape[1]-1)+rgb.shape[1]
            base_all=np.zeros(((rgb.shape[0]-1)*(rgb.shape[1]-1),4))
            
            for i in range(0,rgb.shape[0]-1):
                base_all[i*(rgb.shape[1]-1):(i+1)*(rgb.shape[1]-1),:]=base+i*(rgb.shape[1])

            XX=Xc.ravel()
            YY=Yc.ravel()
            ZZ=rgb.ravel()
            
            filename=filename[:-4]+'_2d.vtk'
            file=open(filename,'w')
            file.write('# vtk DataFile Version 4.2\n')
            file.write('%s\n'%filename)               
            file.write('ASCII\n')
            file.write('DATASET POLYDATA\n')
            file.write('POINTS %d float\n'%((rgb.shape[0]*rgb.shape[1])))
            for i in range(0,rgb.shape[0]*rgb.shape[1]):
                file.write('%.2f %.2f %.2f\n'%(XX[i],YY[i],ZZ[i]))
            
            file.write('POLYGONS %d %d\n'%(base_all.shape[0],5*(base_all.shape[0])))
            for i in range(0,base_all.shape[0]):
                file.write('4 %d %d %d %d\n'%(base_all[i,0],base_all[i,1],base_all[i,2],base_all[i,3]))
            
            file.write('POINT_DATA %d\n'%((rgb.shape[0]*rgb.shape[1])))
            file.write('SCALARS elevation double\n')
            file.write('LOOKUP_TABLE default\n')
            for i in range(0,((rgb.shape[0]*rgb.shape[1]))):
                file.write('%.2f\n'%ZZ[i])
            
            file.close()
            
            
            self.Xc=Xc
            self.Yc=Yc
            self.rgb=rgb
            
            self.xrange=[np.min(x_coords),np.max(x_coords)]
            self.yrange=[np.min(y_coords),np.max(y_coords)]
            return rgb,x_coords,y_coords


    def tiff_to_vtk_as_data(self,filename,rgb,x_coords,y_coords,xres,yres,xrange=None,yrange=None):
        """
        Provide a geotiff and get a dem model with the same name as vtk file. 
        Notice, in this version we do no support cooridante conversion. Future versions will

        """        
        
        

        
        if xrange is not None:
            ix=np.where((x_coords>=xrange[0]) &
                        (x_coords<=xrange[1]) 
                        )[0]
        else:
            ix=np.arange(x_coords.shape[0])
        
        if yrange is not None:
            iy=np.where((y_coords>=yrange[0]) &
                        (y_coords<=yrange[1]) 
                        )[0]
        else:
            iy=np.arange(y_coords.shape[0])

        
        rgb=rgb[iy[:,None],ix[None,:]]
        x_coords=x_coords[ix]
        y_coords=y_coords[iy]
        Xc, Yc = np.meshgrid(x_coords,y_coords);
        
        
        
        
        
        
        
        if np.abs(xres)!=np.abs(yres):
            sys.exit('Error. dx and dy are different. Adjust the code\\')
        else:    
            dx=xres
            # plt.imshow(band1)
            
            number_elements=np.count_nonzero(np.isfinite(rgb.ravel()))


            base=np.zeros((rgb.shape[1]-1,4))
            base[:,0]=np.arange(0,rgb.shape[1]-1)
            base[:,1]=np.arange(1,rgb.shape[1])
            base[:,2]=np.arange(1,rgb.shape[1])+rgb.shape[1]
            base[:,3]=np.arange(0,rgb.shape[1]-1)+rgb.shape[1]
            base_all=np.zeros(((rgb.shape[0]-1)*(rgb.shape[1]-1),4))
            
            for i in range(0,rgb.shape[0]-1):
                base_all[i*(rgb.shape[1]-1):(i+1)*(rgb.shape[1]-1),:]=base+i*(rgb.shape[1])

            XX=Xc.ravel()
            YY=Yc.ravel()
            ZZ=rgb.ravel()
            
            
            file=open(filename,'w')
            file.write('# vtk DataFile Version 4.2\n')
            file.write('%s\n'%filename)               
            file.write('ASCII\n')
            file.write('DATASET POLYDATA\n')
            file.write('POINTS %d float\n'%((rgb.shape[0]*rgb.shape[1])))
            for i in range(0,rgb.shape[0]*rgb.shape[1]):
                file.write('%.2f %.2f %.2f\n'%(XX[i],YY[i],ZZ[i]))
            
            file.write('POLYGONS %d %d\n'%(base_all.shape[0],5*(base_all.shape[0])))
            for i in range(0,base_all.shape[0]):
                file.write('4 %d %d %d %d\n'%(base_all[i,0],base_all[i,1],base_all[i,2],base_all[i,3]))
            
            file.write('POINT_DATA %d\n'%((rgb.shape[0]*rgb.shape[1])))
            file.write('SCALARS elevation double\n')
            file.write('LOOKUP_TABLE default\n')
            for i in range(0,((rgb.shape[0]*rgb.shape[1]))):
                file.write('%.2f\n'%ZZ[i])
            
            file.close()
            
            
            self.Xc=Xc
            self.Yc=Yc
            self.rgb=rgb
            
            self.xrange=[np.min(x_coords),np.max(x_coords)]
            self.yrange=[np.min(y_coords),np.max(y_coords)]
            return
    def tiff_to_vtk_3d_data(self,filename,band1,x_coords,y_coords,xres,yres,band2=np.array([0]),band3=np.array([0]),band4=np.array([0]),label='Property'):
        """
        Provide a geotiff and get a dem model with the same name as vtk file. 
        Notice, in this version we do no support cooridante conversion. Future versions will
        YOu can pass additinal arguments

        """        
        
        
        # raster = gdal.Open(filename)
        # ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
        # x_coords = ulx + np.arange(0,raster.RasterXSize) * xres +  (xres / 2) #add half the cell size
        # y_coords = uly + np.arange(0,raster.RasterYSize) * yres +  (yres / 2) #add half the cell size
        
        if yres<0:
            y_coords=np.flipud(y_coords)
        
        # Xc, Yc = np.meshgrid(x_coords,y_coords);
        # band1 = raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
        band1[band1>1e6]=np.nan
        
        # band1=pd.DataFrame(band1)
        # band1=band1.interpolate().values
        # band1[np.isnan(band1)]=np.nanmin(band1)

                
        bot_z2=np.nanmin(band1.ravel())
        
        if np.abs(xres)!=np.abs(yres):
            sys.exit('Error. dx and dy are different. Adjust the code\\')
        else:    
            dx=xres
            # plt.imshow(band1)
            band1=band1.T
            band2=band2.T
            band3=band3.T
            band4=band4.T


            number_elements=np.count_nonzero(np.isfinite(band1.ravel()))
            
            
            file=open(filename,'w')
            file.write('# vtk DataFile Version 1.0\n')
            file.write('%s\n'%filename)               
            file.write('ASCII\n')
            file.write('\n')
            file.write('DATASET UNSTRUCTURED_GRID\n')
            file.write('POINTS %d float\n'%(8*(number_elements)))
            
            
            for i in range(0,x_coords.shape[0]):
                for j in range(0,y_coords.shape[0]):
                    if np.isfinite(band1[i,j]):
                        x1=x_coords[i]-dx/2
                        y1=y_coords[j]-dx/2
                        z1=band1[i,j]
                        
                        x2=x_coords[i]+dx/2
                        y2=y_coords[j]+dx/2
                        # z2=band1[i,j]+bot_z2-0.1
                        z2=bot_z2-0.1
                        # z2=band1[i,j]-0.1

                        
                        # z2=np.nanmin(band[i-1:i+1,j-1:j+1])
                        
                        # z2=np.nanmin(band1.ravel())-0.1
                        
                        
                        file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                                   (x1,y1,z1,
                                    x1,y2,z1,
                                    x2,y2,z1,
                                    x2,y1,z1,
                                    x1,y1,z2,
                                    x1,y2,z2,
                                    x2,y2,z2,
                                    x2,y1,z2))
                    
            file.write('\n')
            file.write('CELLS %d %d\n'%(number_elements,9*(number_elements)))
            base=np.array([4,7,6,5,0,3,2,1])
            k=0
            for i in range(0,x_coords.shape[0]):
                for j in range(0,y_coords.shape[0]):
                    if np.isfinite(band1[i,j]):
                        file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+k*8,base[1]+k*8,base[2]+k*8,base[3]+k*8,base[4]+k*8,base[5]+k*8,base[6]+k*8,base[7]+k*8))
                        k=k+1
            file.write('\n')
            file.write('CELL_TYPES %d\n'%(number_elements))    
            for i in range(0,number_elements):
                file.write('12\n')
                    
            file.write('\n')
            file.write('CELL_DATA %d\n'%(number_elements))    
            file.write('SCALARS %s float\n'%label) 
            file.write('LOOKUP_TABLE custom_table\n')    
            for i in range(0,x_coords.shape[0]):
                for j in range(0,y_coords.shape[0]):
                    if np.isfinite(band1[i,j]):
                        file.write('%.2f\n'%band1[i,j])
            
            if len(band2)>1:
                file.write('SCALARS Ratio float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('%.2f\n'%band2[i,j])
            if len(band3)>1:
                file.write('SCALARS Diff_seq float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('%.2f\n'%band3[i,j])
            if len(band4)>1:
                file.write('SCALARS Pipe float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('%.2f\n'%band4[i,j])                        
            
            file.close()          
            return
    
    def tiff_to_vtk_3d_data_no_z(self,filename,band1,x_coords,y_coords,xres,yres,band2=np.array([0]),band3=np.array([0]),band4=np.array([0]),label='Property',z=0):
        """
        Provide a geotiff and get a dem model with the same name as vtk file. 
        Notice, in this version we do no support cooridante conversion. Future versions will
        YOu can pass additinal arguments

        """        
        
        
        # raster = gdal.Open(filename)
        # ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
        # x_coords = ulx + np.arange(0,raster.RasterXSize) * xres +  (xres / 2) #add half the cell size
        # y_coords = uly + np.arange(0,raster.RasterYSize) * yres +  (yres / 2) #add half the cell size
        
        if yres<0:
            y_coords=np.flipud(y_coords)
        
        # Xc, Yc = np.meshgrid(x_coords,y_coords);
        # band1 = raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
        band1[band1>1e6]=np.nan
        
        # band1=pd.DataFrame(band1)
        # band1=band1.interpolate().values
        # band1[np.isnan(band1)]=np.nanmin(band1)

                
        bot_z2=np.nanmin(band1.ravel())
        
        if np.abs(xres)!=np.abs(yres):
            sys.exit('Error. dx and dy are different. Adjust the code\\')
        else:    
            dx=xres
            # plt.imshow(band1)
            band1=band1.T
            band2=band2.T
            band3=band3.T
            band4=band4.T


            number_elements=np.count_nonzero(np.isfinite(band1.ravel()))
            
            
            file=open(filename,'w')
            file.write('# vtk DataFile Version 1.0\n')
            file.write('%s\n'%filename)               
            file.write('ASCII\n')
            file.write('\n')
            file.write('DATASET UNSTRUCTURED_GRID\n')
            file.write('POINTS %d float\n'%(8*(number_elements)))
            
            
            for i in range(0,x_coords.shape[0]):
                for j in range(0,y_coords.shape[0]):
                    if np.isfinite(band1[i,j]):
                        x1=x_coords[i]-dx/2
                        y1=y_coords[j]-dx/2
                        z1=z-dx/2
                        
                        x2=x_coords[i]+dx/2
                        y2=y_coords[j]+dx/2
                        # z2=band1[i,j]+bot_z2-0.1
                        # z2=bot_z2-0.1
                        z2=z+dx/2

                        
                        # z2=np.nanmin(band[i-1:i+1,j-1:j+1])
                        
                        # z2=np.nanmin(band1.ravel())-0.1
                        
                        
                        file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                                   (x1,y1,z1,
                                    x1,y2,z1,
                                    x2,y2,z1,
                                    x2,y1,z1,
                                    x1,y1,z2,
                                    x1,y2,z2,
                                    x2,y2,z2,
                                    x2,y1,z2))
                    
            file.write('\n')
            file.write('CELLS %d %d\n'%(number_elements,9*(number_elements)))
            base=np.array([4,7,6,5,0,3,2,1])
            k=0
            for i in range(0,x_coords.shape[0]):
                for j in range(0,y_coords.shape[0]):
                    if np.isfinite(band1[i,j]):
                        file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+k*8,base[1]+k*8,base[2]+k*8,base[3]+k*8,base[4]+k*8,base[5]+k*8,base[6]+k*8,base[7]+k*8))
                        k=k+1
            file.write('\n')
            file.write('CELL_TYPES %d\n'%(number_elements))    
            for i in range(0,number_elements):
                file.write('12\n')
                    
            file.write('\n')
            file.write('CELL_DATA %d\n'%(number_elements))    
            file.write('SCALARS %s float\n'%label) 
            file.write('LOOKUP_TABLE custom_table\n')    
            for i in range(0,x_coords.shape[0]):
                for j in range(0,y_coords.shape[0]):
                    if np.isfinite(band1[i,j]):
                        file.write('%.2f\n'%band1[i,j])
            
            if len(band2)>1:
                file.write('SCALARS Ratio float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('%.2f\n'%band2[i,j])
            if len(band3)>1:
                file.write('SCALARS Diff_seq float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('%.2f\n'%band3[i,j])
            if len(band4)>1:
                file.write('SCALARS Pipe float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('%.2f\n'%band4[i,j])                        
            
            file.close()          
            return


    def scatter_to_surface(self,filename,band1,xres,yres,zres,label='Surface'):
        """
        Provide a geotiff and get a dem model with the same name as vtk file. 
        Notice, in this version we do no support cooridante conversion. Future versions will
        YOu can pass additinal arguments

        """        
        
        
        
        
        x_coords=band1[:,0]
        y_coords=band1[:,1]
        band1=band1[:,2]
        dx=xres
        dy=yres
        dz=zres
 
                
        bot_z2=np.nanmin(band1.ravel())
        top_z2=np.nanmax(band1.ravel())



        number_elements=np.count_nonzero(np.isfinite(band1.ravel()))
        
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(8*(number_elements)))
        
        for i in range(0,len(x_coords)):

            x1=x_coords[i]-dx/2
            y1=y_coords[i]-dy/2
            z1=band1[i]-dz/2
            
            x2=x_coords[i]+dx/2
            y2=y_coords[i]+dy/2
            # z2=band1[i,j]+bot_z2-0.1
            # z2=bot_z2-0.1
            z2=band1[i]+dz/2

            
            # z2=np.nanmin(band[i-1:i+1,j-1:j+1])
            
            # z2=np.nanmin(band1.ravel())-0.1
            
            
            file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                       (x1,y1,z1,
                        x1,y2,z1,
                        x2,y2,z1,
                        x2,y1,z1,
                        x1,y1,z2,
                        x1,y2,z2,
                        x2,y2,z2,
                        x2,y1,z2))
                
        file.write('\n')
        file.write('CELLS %d %d\n'%(number_elements,9*(number_elements)))
        base=np.array([4,7,6,5,0,3,2,1])
        k=0
        for i in range(0,len(x_coords)):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+k*8,base[1]+k*8,base[2]+k*8,base[3]+k*8,base[4]+k*8,base[5]+k*8,base[6]+k*8,base[7]+k*8))
            k=k+1
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(number_elements))    
        for i in range(0,number_elements):
            file.write('12\n')
                
        file.write('\n')
        file.write('CELL_DATA %d\n'%(number_elements))    
        file.write('SCALARS %s float\n'%label) 
        file.write('LOOKUP_TABLE custom_table\n')    
        for i in range(0,len(x_coords)):
            file.write('%.2f\n'%band1[i])
        

            file.close()          
            return

    def scatter_to_surface_no_z(self,filename,band1,xres,yres,zres,label='Surface'):
        """
        Provide a geotiff and get a dem model with the same name as vtk file. 
        Notice, in this version we do no support cooridante conversion. Future versions will
        YOu can pass additinal arguments

        """        
        
        
        
        
        x_coords=band1[:,0]
        y_coords=band1[:,1]
        band1=band1[:,2]
        dx=xres
        dy=yres
        dz=zres
 
                
        bot_z2=np.nanmin(band1.ravel())
        top_z2=np.nanmax(band1.ravel())



        number_elements=np.count_nonzero(np.isfinite(band1.ravel()))
        
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(8*(number_elements)))
        
        for i in range(0,len(x_coords)):

            x1=x_coords[i]-dx/2
            y1=y_coords[i]-dy/2
            # z1=band1[i]-dz/2
            z1=dz/2

            
            x2=x_coords[i]+dx/2
            y2=y_coords[i]+dy/2
            # z2=band1[i,j]+bot_z2-0.1
            # z2=bot_z2-0.1
            z2=dz/2

            
            # z2=np.nanmin(band[i-1:i+1,j-1:j+1])
            
            # z2=np.nanmin(band1.ravel())-0.1
            
            
            file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                       (x1,y1,z1,
                        x1,y2,z1,
                        x2,y2,z1,
                        x2,y1,z1,
                        x1,y1,z2,
                        x1,y2,z2,
                        x2,y2,z2,
                        x2,y1,z2))
                
        file.write('\n')
        file.write('CELLS %d %d\n'%(number_elements,9*(number_elements)))
        base=np.array([4,7,6,5,0,3,2,1])
        k=0
        for i in range(0,len(x_coords)):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+k*8,base[1]+k*8,base[2]+k*8,base[3]+k*8,base[4]+k*8,base[5]+k*8,base[6]+k*8,base[7]+k*8))
            k=k+1
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(number_elements))    
        for i in range(0,number_elements):
            file.write('12\n')
                
        file.write('\n')
        file.write('CELL_DATA %d\n'%(number_elements))    
        file.write('SCALARS %s float\n'%label) 
        file.write('LOOKUP_TABLE custom_table\n')    
        for i in range(0,len(x_coords)):
            file.write('%.2f\n'%band1[i])
        

            file.close()          
            return

    def tiff_to_vtk_data(self,filename,rgb,x_coords,y_coords,xres,yres,rgb2=np.array([0])):
        """
        Provide a geotiff and get a dem model with the same name as vtk file. 
        Notice, in this version we do no support cooridante conversion. Future versions will

        """        
        
        
        # raster = gdal.Open(filename)
        # ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
        # x_coords = ulx + np.arange(0,raster.RasterXSize) * xres +  (xres / 2) #add half the cell size
        # y_coords = uly + np.arange(0,raster.RasterYSize) * yres +  (yres / 2) #add half the cell size
        
        if yres<0:
            y_coords=np.flipud(y_coords)
        
        Xc, Yc = np.meshgrid(x_coords,y_coords);
        # rgb = raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
        rgb[rgb>1e6]=np.nan
        
        # rgb=pd.DataFrame(rgb)
        # rgb=rgb.interpolate().values
        # rgb[np.isnan(rgb)]=np.nanmin(rgb)
        #rgb=np.flipud(rgb)
        
        if np.abs(xres)!=np.abs(yres):
            sys.exit('Error. dx and dy are different. Adjust the code\\')
        else:    
            dx=xres
            # plt.imshow(band1)
            
            number_elements=np.count_nonzero(np.isfinite(rgb.ravel()))


            base=np.zeros((rgb.shape[1]-1,4))
            base[:,0]=np.arange(0,rgb.shape[1]-1)
            base[:,1]=np.arange(1,rgb.shape[1])
            base[:,2]=np.arange(1,rgb.shape[1])+rgb.shape[1]
            base[:,3]=np.arange(0,rgb.shape[1]-1)+rgb.shape[1]
            base_all=np.zeros(((rgb.shape[0]-1)*(rgb.shape[1]-1),4))
            
            for i in range(0,rgb.shape[0]-1):
                base_all[i*(rgb.shape[1]-1):(i+1)*(rgb.shape[1]-1),:]=base+i*(rgb.shape[1])

            XX=Xc.ravel()
            YY=Yc.ravel()
            ZZ=rgb.ravel()
            
            
            
            file=open(filename,'w')
            file.write('# vtk DataFile Version 4.2\n')
            file.write('%s\n'%filename)               
            file.write('ASCII\n')
            file.write('DATASET POLYDATA\n')
            file.write('POINTS %d float\n'%((number_elements)))
            for i in range(0,rgb.shape[0]*rgb.shape[1]):
                if np.isfinite(ZZ[i]):
                    file.write('%.2f %.2f %.2f\n'%(XX[i],YY[i],ZZ[i]))
            
            file.write('POLYGONS %d %d\n'%(number_elements,5*(number_elements)))
            for i in range(0,base_all.shape[0]):
                if np.isfinite(ZZ[i]):
                    file.write('4 %d %d %d %d\n'%(base_all[i,0],base_all[i,1],base_all[i,2],base_all[i,3]))
            
            file.write('POINT_DATA %d\n'%((number_elements)))
            file.write('SCALARS elevation double\n')
            file.write('LOOKUP_TABLE default\n')
            for i in range(0,((rgb.shape[0]*rgb.shape[1]))):
                if np.isfinite(ZZ[i]):
                    file.write('%.2f\n'%ZZ[i])

            if len(rgb2)>1:
                file.write('SCALARS property double\n')
                file.write('LOOKUP_TABLE default\n')
                ZZ2=rgb2.ravel()
                for i in range(0,((rgb.shape[0]*rgb.shape[1]))):
                    if np.isfinite(ZZ[i]):
                        file.write('%.2f\n'%ZZ2[i])

            
            file.close()
            return


    def make_2d_plane_for_texture(self,filename,x,y,z,property_color=1):
        """
        Provide an array to generate a 2D pane, in 3D space
        If you are generating a plane in 2D space, assign zeros to the x3 dim

        :data: an Nx3 matrix with xyz coordinates
        : OPTIONAL property_color is an value (float) in case you want to plot sevesarl objects, and each to have a different color. 
        """
        
        no_planes=x.shape[0]-1
        no_points=x.shape[0]
        if z.shape[0]!=2:
            sys.exit("ERROR: Currently, only 2 size vector are allowed")    
            
        if x.shape[0]!=y.shape[0]:
            sys.exit("ERROR: x,y have different size")                
            
        z=np.r_[z[0]*np.ones(x.shape[0]),z[1]*np.ones(x.shape[0])]
        x=np.r_[x,np.flipud(x)]          
        y=np.r_[y,np.flipud(y)]          
        
        data=np.c_[x,y,z]
                    
        
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('vtk output\n')
        file.write('ASCII\n')
        file.write('DATASET POLYDATA\n')
        file.write('POINTS %d double\n'%data.shape[0])
        
        for k in range(0,data.shape[0]):
            file.write('%.3f %.3f %.3f\n'%(data[k,0],data[k,1],data[k,2]))
        
        basis=np.array([0,1,2*no_planes,2*no_planes+1,0])
        file.write('POLYGONS %d %d\n'%(no_planes,6*(no_planes)))        
               
        for k in range(0,no_planes):
            file.write('5 ')
            file.write('%d %d %d %d %d\n'%(basis[0],basis[1],basis[2],basis[3],basis[4]))
            basis[0]=basis[0]+1
            basis[1]=basis[1]+1
            basis[2]=basis[2]-1
            basis[3]=basis[3]-1
            basis[4]=basis[4]+1

        file.write('CELL_DATA %d\n'%(no_planes))    
        file.write('SCALARS %s float\n'%'Property')    
        file.write('LOOKUP_TABLE custom_table\n')
        for k in range(0,no_planes):
            file.write('%.1f\n'%property_color)
        file.close() 
        
        return

    def make_2d_profiles(self,filename,data,x,y,z):
        """
        Provide a 2D grided data, that represent a a 2d profile.
        dim[1] is the depth 
        Provide the coordate of this profile, on a map. There are two options
        if you know the the cooridantes (x,y) for each of the point of the 
        input matrix, then x,y.shape[0]==data.shape[1].
        Spacing is then based on those points
        If you know only few points (or begging and end of the line), then 
        we interpolate in equlivend spacing.
        
        :data mxn matrix
        :x nx1 vector with cooridantes along the x-axis (or less to interpolate)
        :y nx1 vector with cooridantes along the y-axis (or less to interpolate)
        :z mx1 vector with elevation (it must be data.shape[0 dimensions] or two points. Then it divides equal size layers)        
        """      

        
        if (x.shape[0]!=data.shape[1] ) & (y.shape[0]!=data.shape[1] ):        
            # find starting point and ending poits and we interpolater
            t=np.sqrt(np.power(x[1:]-x[:-1],2) + np.power(y[1:]-y[:-1],2)     )
            dis=np.r_[0,np.cumsum(t)]  
            # interpoltate to find starting point
            fx=interp1d(dis,x,kind='linear',fill_value='extrapolate')
            fy=interp1d(dis,y,kind='linear',fill_value='extrapolate')
            # split the section in equal spacings
            dis_new=np.linspace(np.min(dis),np.max(dis),data.shape[1])
            #find coordiantes
            x=fx(dis_new)
            y=fy(dis_new)            

        if (z.shape[0]!=data.shape[0] ):
            z=np.linspace(np.min(z),np.max(z),data.shape[0])
            
        file=open(filename,'w')
        file.write('# vtk DataFile Version 3.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        
        file.write('DATASET STRUCTURED_GRID\n')
        file.write('DIMENSIONS 1 %d %d\n'%(data.shape[1],data.shape[0]))
        file.write('POINTS %d float\n'%((data.shape[0]*data.shape[1])))
        
        for i in range(0,(data.shape[0])):
            for k in range(0,data.shape[1]):        
                file.write('%.2f %.2f %.2f\n'%
                       (x[k],y[k],z[i],
    
                                ))
        file.write('\n')
       
        
        file.write('POINT_DATA %d\n'%(data.shape[0]*data.shape[1]))    
        file.write('SCALARS Property_1 float\n')    
        file.write('LOOKUP_TABLE default\n')    
        for i in range(0,(data.shape[0])):
            for k in range(0,data.shape[1]):
                file.write('%.2f\n'%data[i,k])
    
    
        
        file.close()
        return
    
    def make_2d_profiles_with_elevation(self,filename,data,x,y,dep_top,dep_bot=0,elev=0,data2=[0]):
        """
        Provide a 2D grided data, that represent a a 2d profile.
        dim[1] is the depth 
        Provide the coordate of this profile, on a map. There are two options
        if you know the the cooridantes (x,y) for each of the point of the 
        input matrix, then x,y.shape[0]==data.shape[1].
        Spacing is then based on those points
        If you know only few points (or begging and end of the line), then 
        we interpolate in equlivend spacing.
        
        :data mxn matrix
        :x nx1 vector with cooridantes along the x-axis (or less to interpolate)
        :y nx1 vector with cooridantes along the y-axis (or less to interpolate)
        :dep_top mxn  vector with elevation (it must be data.shape[0 dimensions] or two points. Then it divides equal size layers)  
        :dep_bot (OPTIONAL) if provided, then 
        :elev nx1 
        """    
        
        
        # First we check how the depths are provided
        if (dep_top.shape[0]!=data.shape[0] ):
            print('Warning: The depth values are not equal to the size of the data. Equal spacing')
            dep_top=np.matrix(np.linspace(np.min(dep_top.ravel()),np.max(dep_top.ravel()),data.shape[0])          )
            # force bottom layer indepenet what it is
            dep_bot=0
        
        
        if (dep_top.shape[0]==1):
            print('Only top Suraface is provided. Bottom will be calculated')
            dep_bot=np.r_[dep_top[1:],1.1*dep_top[-1]]
        
        
        # now we check if depth are different per every z or not
        # make sure is a vector
        if  dep_top.shape==((data.shape[0],)):
            dep_top=np.matrix(dep_top).T
            dep_bot=0*dep_top
            
            
        if dep_top.shape[1]==1:
            dep_top=np.reshape(np.repeat(dep_top,data.shape[1],axis=0),(dep_top.shape[0],data.shape[1]))      
            dep_bot=np.reshape(np.repeat(dep_bot,data.shape[1],axis=0),(dep_bot.shape[0],data.shape[1]))      


            

  
        
        
        if (x.shape[0]!=y.shape[0]) & (x.shape[0]!=dep_top.shape[0]) & (x.shape[0]!=dep_bot.shape[0]) & (data.shape[0]!=dep_bot.shape[0]):
            sys.exit('Error: x,y,z shuould have the same dimensions')
        
        
        if (x.shape[0]!=data.shape[1] ) & (y.shape[0]!=data.shape[1] ) :        
            # find starting point and ending poits and we interpolater
            t=np.sqrt(np.power(x[1:]-x[:-1],2) + np.power(y[1:]-y[:-1],2)     )
            dis=np.r_[0,np.cumsum(t)]  
            # interpoltate to find starting point
            fx=interp1d(dis,x,kind='linear',fill_value='extrapolate')
            fy=interp1d(dis,y,kind='linear',fill_value='extrapolate')
            # fz=interp1d(dis,z,kind='linear',fill_value='extrapolate')
            # split the section in equal spacings
            dis_new=np.linspace(np.min(dis),np.max(dis),data.shape[1])
            #find coordiantes
            x=fx(dis_new)
            y=fy(dis_new)
            # z=fz(dis_new)           

        

            
        # if elev==0:
        #     elev=np.zeros((data.shape[1],1))
        
        
        
        
        no_polygons=np.count_nonzero(~np.isnan(data))        

        # we need x,y,z to have +1 size
        # we assume that x,y,z are the cneters of the cells. It can be as irrelgural as user wants
        # later versions will have an automatic split, is the distance is significant bigger between nearby cels.
        mid_points_x=0.5*(np.float64(x[:-1])+ np.float64(x[1:]))
        mid_points_y=0.5*(np.float64(y[:-1])+ np.float64(y[1:]))
        
        mid_points_x=np.r_[x[0],mid_points_x,x[-1]]
        mid_points_y=np.r_[y[0],mid_points_y,y[-1]]
        
        # dz=np.abs(z[-2]-z[-1])
        # dep_top=z
        # dep_bottom=np.r_[z[1:],z[1]-dz]
        
        
        base=np.array([0,1,2,3])
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('DATASET POLYDATA\n')
        file.write('POINTS %d float\n'%(4*data.shape[1]*data.shape[0]))
        
        # find topography
        
        
        for k in range(0,data.shape[1]):
            x1=mid_points_x[k]
            x2=mid_points_x[k+1]
            y1=mid_points_y[k]
            y2=mid_points_y[k+1]
            
            
            for n in range(0,data.shape[0]):
                z1=-dep_top[n,k]+elev[k]
                z2=-dep_bot[n,k]+elev[k]
                file.write('%.6f %.6f %.6f\n'%(x1,y1,z2))
                file.write('%.6f %.6f %.6f\n'%(x2,y2,z2))
                file.write('%.6f %.6f %.6f\n'%(x2,y2,z1))
                file.write('%.6f %.6f %.6f\n'%(x1,y1,z1))
        
        
        file.write('POLYGONS %d %d\n'%(no_polygons,5*no_polygons))   
        for k in range(0,data.shape[1]):
            for n in range(0,data.shape[0]):
                if np.isfinite(data[n,k]):
                    file.write('4 %d %d %d %d\n'%(base[0],base[1],base[2],base[3]))
                base=base+4
        file.write('CELL_DATA %d\n'%((no_polygons)))
        file.write('SCALARS Resistivity(Ohm.m) double\n')
        file.write('LOOKUP_TABLE default\n')        
        for k in range(0,data.shape[1]):
            for n in range(0,data.shape[0]):
                if np.isfinite(data[n,k]):
                    file.write('%.2f\n'%data[n,k])

        if len(data2)>1:
            file.write('SCALARS Geology_Class double\n')
            file.write('LOOKUP_TABLE default\n')        
            for k in range(0,data.shape[1]):
                for n in range(0,data.shape[0]):
                    if np.isfinite(data[n,k]):
                        file.write('%.2f\n'%data2[n,k])
                
                
        file.close()
        return


    def make_2d_profiles_with_elevation_scatter_data(self,filename,data,x=np.array([0]),y=np.array([0]),elev=np.array([0])):
        """
        Provide a 2D xy-grided data, that represent a a 2d profile.
        
        Provide the coordate of this profile, on a map. There are two options
        if you know the the cooridantes (x,y) for each of the point of the 
        input matrix, then x,y.shape[0]==data.shape[1].
        Spacing is then based on those points
        If you know only few points (or begging and end of the line), then 
        we interpolate in equlivend spacing.
        
        :data mx3 matrix
        :x nx1 vector with cooridantes along the x-axis (or less to interpolate)
        :y nx1 vector with cooridantes along the y-axis (or less to interpolate)
        :elev nx1 vector of elevations
        """    
        
        if (len(x)==len(y))  & (len(x)==len(elev)) :
            #all is good
            i=1
        else:
            sys.exit('Error: x,y,z shuould have the same dimensions')
        
        
        # Check id dpths given are negative
        if np.min(data[:,1]<0):
            d_flag=1
            data[:,1]=np.abs(data[:,1])
        else:
            d_flag=0
        
        # make data into grid
        x_unique=np.unique(data[:,0])
        z_unique=np.unique(data[:,1])          
        
        xc,yc=np.meshgrid(x_unique,z_unique)
        # average dx,dz
        dx=np.abs(x_unique[1:]-x_unique[:-1])
        dy=dx
        dz=np.abs(z_unique[1:]-z_unique[:-1])

        # find regular grid now
        x_unique=np.arange(np.min(x_unique),np.max(x_unique)+np.average(dx),np.average(dx))
        i1=np.int32(np.floor_divide(data[:,0]-np.min(x_unique),np.average(dx)))
        z_unique=np.arange(np.min(z_unique),np.max(z_unique)+np.average(dz),np.average(dz))
        i2=np.int32(np.floor_divide(data[:,1]-np.min(z_unique),np.average(dz)))
 
        
        ix=np.where((i1>=0) & (i1<x_unique.shape[0]) & (i2>=0) & (i2<z_unique.shape[0])   )[0]
        lin_index=(i1[ix])*(z_unique.shape[0])  +(i2[ix])   
        
        
        
        bs=np.nan*np.zeros((x_unique.shape[0]*z_unique.shape[0],1))        
        df=pd.DataFrame({'values':data[:,2],'ii':lin_index})

        bi=df.groupby('ii').mean()
        bs[bi.index.values]=np.reshape(bi['values'].values,(bi.shape[0],1))
        bs=np.reshape(bs,((x_unique.shape[0],z_unique.shape[0])))
        bs=bs.T
        
        
        
        
        # average data
        # for i in range(0,x_unique.shape[0]):
        #     for j in range(0,z_unique.shape[0]):
        #         dis=np.sqrt( np.power(data[:,1]-z_unique[j],2) + np.power(data[:,0]-x_unique[i],2)  )
        #         i1=np.argmin(dis)
        #         plot_data[j,i]=data[i1,2]
        
        
        
        
        if len(x)>1:# We have coordinates
            # find actual coordantes to rotate
            t=np.sqrt(np.power(x[1:]-x[:-1],2) + np.power(y[1:]-y[:-1],2)     )
            dis=np.r_[0,np.cumsum(t)]  
            # interpoltate to find starting point
            fx=interp1d(dis,x,kind='linear',fill_value='extrapolate')
            fy=interp1d(dis,y,kind='linear',fill_value='extrapolate')
            fz=interp1d(dis,elev,kind='linear',fill_value='extrapolate')
            # split the section in equal spacings
            # t=np.sqrt(np.power(x[1:]-x[:-1],2) + np.power(y[1:]-y[:-1],2)     )
            # dis_mew=np.r_[0,np.cumsum(t)]  
            #find coordiantes
            xx=fx(xc)
            yy=fy(xc)
            elev=fz(xc)    
        else:
            xx=xc
            yy=0*xx
            # find actual coordantes to rotate
            t=np.sqrt(np.power(x_unique[1:]-x_unique[:-1],2)     )
            dis=np.r_[0,np.cumsum(t)]  
            elev=0*xx

    

            
        dep_top=0.5*(yc[1:,:]+ yc[:-1,:])
        dep_top=np.r_[np.matrix(0*dep_top[0,:]),dep_top]
        dep_bot=np.r_[dep_top[1:,:],np.matrix(yc[-1,:])]
        
        
        
        # we need x,y,z to have +1 size
        # we assume that x,y,z are the cneters of the cells. It can be as irrelgural as user wants
        # later versions will have an automatic split, is the distance is significant bigger between nearby cels.
        # mid_points_x=0.5*(np.float32(x[:-1])+ np.float32(x[1:]))
        # mid_points_y=0.5*(np.float32(y[:-1])+ np.float32(y[1:]))
        
        # mid_points_x=np.r_[x[0],mid_points_x,x[-1]]
        # mid_points_y=np.r_[y[0],mid_points_y,y[-1]]
        mid_points_x=np.c_[xx[:,0]-np.average(dx/2),xx,xx[:,-1]+np.average(dx/2)]
        mid_points_y=np.c_[yy[:,0]-np.average(dy/2),yy,yy[:,-1]+np.average(dy/2)]

        # dz=np.abs(z[-2]-z[-1])
        # dep_top=z
        # dep_bottom=np.r_[z[1:],z[1]-dz]
        
        
        base=np.array([0,1,2,3])
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('DATASET POLYDATA\n')
        file.write('POINTS %d float\n'%(bs.shape[1]*4*bs.shape[0]))
        
        # find topography
        
        
        for k in range(0,bs.shape[1]):

            
            
            for n in range(0,bs.shape[0]):
                
                
                x1=mid_points_x[n,k]
                x2=mid_points_x[n,k+1]
                y1=mid_points_y[n,k]
                y2=mid_points_y[n,k+1]
                
                z1=-dep_top[n,k]+elev[n,k]
                z2=-dep_bot[n,k]+elev[n,k]
                file.write('%.2f %.2f %.2f\n'%(x1,y1,z2))
                file.write('%.2f %.2f %.2f\n'%(x2,y2,z2))
                file.write('%.2f %.2f %.2f\n'%(x2,y2,z1))
                file.write('%.2f %.2f %.2f\n'%(x1,y1,z1))
                
        # basically, only write where we have number 
        no_polygons=np.count_nonzero(~np.isnan(bs))        
        
        file.write('POLYGONS %d %d\n'%(no_polygons,5*no_polygons))   
        for k in range(0,bs.shape[1]):
            for n in range(0,bs.shape[0]):
                if np.isfinite(bs[n,k]):
                    file.write('4 %d %d %d %d\n'%(base[0],base[1],base[2],base[3]))
                base=base+4
        
        file.write('CELL_DATA %d\n'%((no_polygons)))
        file.write('SCALARS Resistivity(Ohm.m) double\n')
        file.write('LOOKUP_TABLE default\n')        
        for k in range(0,bs.shape[1]):
            for n in range(0,bs.shape[0]):
                if np.isfinite(bs[n,k]):
                    file.write('%.2f\n'%bs[n,k])
    
        file.close()
        return



    def make_2d_profiles_from_xyz(self,filename,data,trim=0.01,x=np.array([0]),y=np.array([0]),elev=np.array([0])):
        """
        Provide a 2D xy-grided data, that represent a a 2d profile.
        
        Currently under development. It will create a mesh based on the xy data
        
        :data mx3 matrix
        :x nx1 vector with cooridantes along the x-axis (or less to interpolate)
        :y nx1 vector with cooridantes along the y-axis (or less to interpolate)
        :elev nx1 vector of elevations
        """    
        
        if (len(x)==len(y))  & (len(x)==len(elev)) :
            #all is good
            i=1
        else:
            sys.exit('Error: x,y,z shuould have the same dimensions')
        
        
        # Check id dpths given are negative
        if np.min(data[:,1]<0):
            d_flag=1
            data[:,1]=np.abs(data[:,1])
        else:
            d_flag=0
        
        
        triang = mpl.tri.Triangulation(data[:,0], data[:,1])
        mask = mpl.tri.TriAnalyzer(triang).get_flat_tri_mask(trim)
        triang.set_mask(mask)

        # plt.triplot(triang)
        nodes_x=triang.x 
        nodes_y=triang.y
        
        tr=triang.triangles
        
        # find center of triangles for interpolation
        x_tri=1/3*(nodes_x[tr[:,0]]+nodes_x[tr[:,1]]+ nodes_x[tr[:,2]])
        z_tri=1/3*(nodes_y[tr[:,0]]+nodes_y[tr[:,1]]+ nodes_y[tr[:,2]])

        
        # f = interp2d(data[:,0], data[:,1], data[:,2], kind='linear')
        f=CloughTocher2DInterpolator(np.c_[data[:,0], data[:,1]], data[:,2])
        new_val=f(x_tri,z_tri)
        
        
        
        no_polygons=np.count_nonzero(~(mask))  
        
        # find center of triangles
      
        
        
        
        
        if len(x)>1:# We have coordinates
            # find actual coordantes to rotate
            t=np.sqrt(np.power(x[1:]-x[:-1],2) + np.power(y[1:]-y[:-1],2)     )
            dis=np.r_[0,np.cumsum(t)]  
            # interpoltate to find starting point
            fx=interp1d(dis,x,kind='linear',fill_value='extrapolate')
            fy=interp1d(dis,y,kind='linear',fill_value='extrapolate')
            fz=interp1d(dis,elev,kind='linear',fill_value='extrapolate')
            # split the section in equal spacings
            # t=np.sqrt(np.power(x[1:]-x[:-1],2) + np.power(y[1:]-y[:-1],2)     )
            # dis_mew=np.r_[0,np.cumsum(t)]  
            #find coordiantes
            xx=fx(nodes_x)
            yy=fy(nodes_x)
            elev=fz(nodes_x)    
        else:
            xx=nodes_x
            yy=0*xx
            elev=0*xx
      
        
        # base=np.array([0,1,2,3])
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(nodes_x.shape[0]))
        
        for k in range(0,nodes_x.shape[0]):
            file.write('%.2f %.2f %.2f\n'%(xx[k],yy[k],-(nodes_y[k]-elev[k])))

        
        file.write('CELLS %d %d\n'%(no_polygons,4*no_polygons))   
        for k in range(0,tr.shape[0]):
            if mask[k]==False:
                file.write('3 %d %d %d\n'%(tr[k,0],tr[k,1],tr[k,2]))
     
        file.write('CELL_TYPES %d\n'%((no_polygons)))
        for k in range(0,tr.shape[0]):
            if mask[k]==False:
                file.write('5\n')                    
        file.write('CELL_DATA %d\n'%((no_polygons)))
        file.write('SCALARS Property_1 double\n')
        file.write('LOOKUP_TABLE default\n')        
        for k in range(0,tr.shape[0]):
            if mask[k]==False:
                file.write('%.3f\n'%new_val[k])     
    
        file.close()
        return

        
    def make_3d_grid_to_vtk(self,filename,data,xc,yc,zc,data2=[0],mosaic=[0],name='Parameter'):
        """
        Convert 3D geological grid data to VTK structured grid format.
        
        Creates VTK representation of 3D geological models with multiple properties.
        Supports geological formations, geophysical data, or any 3D spatial data
        organized on a regular grid structure.

        Parameters
        ----------
        filename : str
            Output VTK filename (including .vtk extension).
        data : numpy.ndarray
            3D grid data with shape (m, n, k) where:
            - m: number of points along Y-axis (rows)
            - n: number of points along X-axis (columns)  
            - k: number of depth/elevation layers
        xc : numpy.ndarray
            1D array of X-coordinates with length n.
            Defines the grid spacing along X-axis.
        yc : numpy.ndarray
            1D array of Y-coordinates with length m.
            Defines the grid spacing along Y-axis.
        zc : numpy.ndarray
            1D array of Z-coordinates with length (k+1).
            Defines layer boundaries - must have one more element than data depth.
        data2 : numpy.ndarray, optional
            Secondary data array for additional properties. 
            Default is [0] (no secondary data).
        mosaic : numpy.ndarray, optional
            2D masking array with shape (m, n) to exclude certain areas.
            Default is [0] (no masking applied).
        name : str, optional
            Name for the data property in VTK file. Default is 'Parameter'.

        Raises
        ------
        SystemExit
            If mosaic dimensions don't match data dimensions.

        Notes
        -----
        - Creates VTK STRUCTURED_GRID format for regular 3D grids
        - Coordinate arrays define the actual spatial positioning
        - Z-coordinates typically represent elevation (positive up) or depth (positive down)
        - Multiple data properties can be included for multi-attribute visualization
        - Mosaic masking useful for complex geological boundaries

        Examples
        --------
        >>> # Create 3D geological model VTK
        >>> grid_3d = np.random.rand(50, 40, 10)  # Geological properties
        >>> x_coords = np.linspace(0, 1000, 40)   # 1km extent
        >>> y_coords = np.linspace(0, 800, 50)    # 800m extent  
        >>> z_coords = np.linspace(100, 0, 11)    # 100m depth, 11 layer boundaries
        >>> vtk.make_3d_grid_to_vtk('model_3d.vtk', grid_3d, x_coords, y_coords, z_coords)
        
        >>> # With multiple properties and masking
        >>> resistivity = np.random.rand(30, 25, 8)
        >>> density = np.random.rand(30, 25, 8) 
        >>> land_mask = np.ones((30, 25))  # 1 for land, 0 for water
        >>> vtk.make_3d_grid_to_vtk('geophysics.vtk', resistivity, x_coords, y_coords, 
        ...                        z_coords, data2=density, mosaic=land_mask, 
        ...                        name='Resistivity')
        """            
        ind=0
        if len(mosaic)==1:
            mosaic=np.zeros((data.shape[0],data.shape[1]))
            ind=1
        if (mosaic.shape[0]!=data.shape[0]):
            sys.exit('Error: mosaic have the same dimensions')
        if (mosaic.shape[1]!=data.shape[1]):
            sys.exit('Error: mosaic have the same dimensions')

        if (xc.shape[0]!=data.shape[1]):
            sys.exit('Error: x should have the same dimensions')            
        if (yc.shape[0]!=data.shape[0]):
            sys.exit('Error: y should have the same dimensions')            
        if ((zc.shape[0])!=data.shape[2]+1):
            sys.exit('Error: z should have the same dimensions')                    
        
        no_elem=np.count_nonzero(~np.isnan(data))
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(8*(no_elem)))    
        
        dx=(xc[1]-xc[0])
        dy=(yc[1]-yc[0])
        dz=(zc[1]-zc[0])
        for k in range(0,data.shape[0]):
            for n in range(0,data.shape[1]):
                for i in range(0,data.shape[2]):
                    # write only if we have data:
                    if np.isnan(data[k,n,i])==False:
                        x1=xc[n]-dx/2
                        x2=xc[n]+dx/2
                        
                        y1=yc[k]-dy/2
                        y2=yc[k]+dy/2
                                             
                        
                        if ind==1:
                            z1=zc[i]
                            z2=zc[i+1]
                        else:
                            z1=mosaic[k,n]-zc[i]
                            z2=mosaic[k,n]-zc[i+1]
                        
                        # z1=mosaic[k,n]+(zc[i]-dz/2)
                        # z2=mosaic[k,n]+(zc[i]+dz/2)
                
                        file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                                   (x1,y1,z1,
                                    x1,y2,z1,
                                    x2,y2,z1,
                                    x2,y1,z1,
                                    x1,y1,z2,
                                    x1,y2,z2,
                                    x2,y2,z2,
                                    x2,y1,z2))
        file.write('\n')
                        
    #    no_elem=38                
        file.write('CELLS %d %d\n'%(no_elem,9*(no_elem)))
        base=np.array([4,7,6,5,0,3,2,1])
        for i in range(0,no_elem):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+i*8,base[1]+i*8,base[2]+i*8,base[3]+i*8,base[4]+i*8,base[5]+i*8,base[6]+i*8,base[7]+i*8))
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(no_elem))    
        for i in range(0,no_elem):
            file.write('12\n')
                     
        file.write('\n')
        file.write('CELL_DATA %d\n'%(no_elem))    
        file.write('SCALARS %s float\n'%name)    
        file.write('LOOKUP_TABLE custom_table\n')    
        for k in range(0,data.shape[0]):
            for n in range(0,data.shape[1]):
                for i in range(0,data.shape[2]):
                    # write only if we have data:
                    if np.isnan(data[k,n,i])==False:
                        file.write('%.6f\n'%data[k,n,i])
        if len(data2)>1:
            file.write('SCALARS %s float\n'%'Probability')    
            file.write('LOOKUP_TABLE custom_table\n')    
            for k in range(0,data2.shape[0]):
                for n in range(0,data2.shape[1]):
                    for i in range(0,data2.shape[2]):
                        # write only if we have data:
                        if np.isnan(data[k,n,i])==False:
                            file.write('%.6f\n'%data2[k,n,i])


    #    for i in range(0,no_elem):
    #        file.write('11\n')
           
    
    
        
        file.close()
        return




    def make_3d_grid_to_vtk_cube(self,filename,data,data2=[0],name='Parameter'):
        """
        Provide a 3D grided data, that represent a a 3d volume.
        dim[2] is the depth 
        Provide the coordate of this profile, on a map. 
        xc,yc,zc are the coordantes
        
        :data mxnxk matrix
        :xc nx1 vector with cooridantes along the x-axis
        :yc mx1 vector with cooridantes along the y-axis
        :zc (k+1)x1 vector with elevation (it must be data.shape[0 dimensions] +1)
        :elev mxn (OPTIONAL if you have topograpy in grid) 
        """            
        

        xc=np.arange(0,data.shape[0])+0.5
        yc=np.arange(0,data.shape[1])+0.5
        zc=np.arange(0,data.shape[2])+0.5
        
        no_elem=np.count_nonzero(~np.isnan(data))
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(8*(no_elem)))    
        
        dx=(xc[1]-xc[0])
        dy=(yc[1]-yc[0])
        dz=(zc[1]-zc[0])
        for k in range(0,data.shape[0]):
            for n in range(0,data.shape[1]):
                for i in range(0,data.shape[2]):
                    # write only if we have data:
                    if np.isnan(data[k,n,i])==False:
                        x1=xc[n]-dx/2
                        x2=xc[n]+dx/2
                        
                        y1=yc[k]-dy/2
                        y2=yc[k]+dy/2
                                             
                        
                        z1=zc[i]-dz/2
                        z2=zc[i]+dz/2
                
                        file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                                   (x1,y1,z1,
                                    x1,y2,z1,
                                    x2,y2,z1,
                                    x2,y1,z1,
                                    x1,y1,z2,
                                    x1,y2,z2,
                                    x2,y2,z2,
                                    x2,y1,z2))
        file.write('\n')
                        
    #    no_elem=38                
        file.write('CELLS %d %d\n'%(no_elem,9*(no_elem)))
        base=np.array([4,7,6,5,0,3,2,1])
        for i in range(0,no_elem):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+i*8,base[1]+i*8,base[2]+i*8,base[3]+i*8,base[4]+i*8,base[5]+i*8,base[6]+i*8,base[7]+i*8))
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(no_elem))    
        for i in range(0,no_elem):
            file.write('12\n')
                     
        file.write('\n')
        file.write('CELL_DATA %d\n'%(no_elem))    
        file.write('SCALARS %s float\n'%name)    
        file.write('LOOKUP_TABLE custom_table\n')    
        for k in range(0,data.shape[0]):
            for n in range(0,data.shape[1]):
                for i in range(0,data.shape[2]):
                    # write only if we have data:
                    if np.isnan(data[k,n,i])==False:
                        file.write('%.6f\n'%data[k,n,i])
        if len(data2)>1:
            file.write('SCALARS %s float\n'%'Probability')    
            file.write('LOOKUP_TABLE custom_table\n')    
            for k in range(0,data2.shape[0]):
                for n in range(0,data2.shape[1]):
                    for i in range(0,data2.shape[2]):
                        # write only if we have data:
                        if np.isnan(data[k,n,i])==False:
                            file.write('%.6f\n'%data2[k,n,i])


    #    for i in range(0,no_elem):
    #        file.write('11\n')
           
    
    
        
        file.close()
        return


    def make_3d_grid_to_vtk_iregular(self,filename,data,xc,yc,zc,dx,dy,dz,data2=[0],mosaic=[0],name='Parameter'):
        """
        Provide a 3D grided data, that represent a a 3d volume.
        dim[2] is the depth 
        Provide the coordate of this profile, on a map. 
        xc,yc,zc are the coordantes
        
        :data mxnxk matrix
        :xc nx1 vector with cooridantes along the x-axis
        :yc mx1 vector with cooridantes along the y-axis
        :zc (k+1)x1 vector with elevation (it must be data.shape[0 dimensions] +1)
        :elev mxn (OPTIONAL if you have topograpy in grid) 
        """            
        ind=0
        if len(mosaic)==1:
            mosaic=np.zeros((data.shape[0],data.shape[1]))
            ind=1
        if (mosaic.shape[0]!=data.shape[0]):
            sys.exit('Error: mosaic have the same dimensions')
        if (mosaic.shape[1]!=data.shape[1]):
            sys.exit('Error: mosaic have the same dimensions')

        if (xc.shape[0]!=data.shape[1]):
            sys.exit('Error: x should have the same dimensions')            
        if (yc.shape[0]!=data.shape[0]):
            sys.exit('Error: y should have the same dimensions')            
        if ((zc.shape[0])!=data.shape[2]):
            sys.exit('Error: z should have the same dimensions')                    
        
        no_elem=np.count_nonzero(~np.isnan(data))
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(8*(no_elem)))    
        

        for k in range(0,data.shape[0]):
            for n in range(0,data.shape[1]):
                for i in range(0,data.shape[2]):
                    # write only if we have data:
                    if np.isnan(data[k,n,i])==False:
                        x1=xc[n]-dx[n]/2
                        x2=xc[n]+dx[n]/2
                        
                        y1=yc[k]-dy[k]/2
                        y2=yc[k]+dy[k]/2
                                             
                        
                        if ind==1:
                            z1=zc[i]-dz[i]/2
                            z2=zc[i]+dz[i]/2
                        else:
                            z1=mosaic[k,n]-zc[i]-dz[i]/2
                            z2=mosaic[k,n]-zc[i]+dz[i]/2
                        
                        # z1=mosaic[k,n]+(zc[i]-dz/2)
                        # z2=mosaic[k,n]+(zc[i]+dz/2)
                
                        file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                                   (x1,y1,z1,
                                    x1,y2,z1,
                                    x2,y2,z1,
                                    x2,y1,z1,
                                    x1,y1,z2,
                                    x1,y2,z2,
                                    x2,y2,z2,
                                    x2,y1,z2))
        file.write('\n')
                        
    #    no_elem=38                
        file.write('CELLS %d %d\n'%(no_elem,9*(no_elem)))
        base=np.array([4,7,6,5,0,3,2,1])
        for i in range(0,no_elem):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+i*8,base[1]+i*8,base[2]+i*8,base[3]+i*8,base[4]+i*8,base[5]+i*8,base[6]+i*8,base[7]+i*8))
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(no_elem))    
        for i in range(0,no_elem):
            file.write('12\n')
                     
        file.write('\n')
        file.write('CELL_DATA %d\n'%(no_elem))    
        file.write('SCALARS %s float\n'%name)    
        file.write('LOOKUP_TABLE custom_table\n')    
        for k in range(0,data.shape[0]):
            for n in range(0,data.shape[1]):
                for i in range(0,data.shape[2]):
                    # write only if we have data:
                    if np.isnan(data[k,n,i])==False:
                        file.write('%.6f\n'%data[k,n,i])
        if len(data2)>1:
            file.write('SCALARS %s float\n'%'Probability')    
            file.write('LOOKUP_TABLE custom_table\n')    
            for k in range(0,data2.shape[0]):
                for n in range(0,data2.shape[1]):
                    for i in range(0,data2.shape[2]):
                        # write only if we have data:
                        if np.isnan(data[k,n,i])==False:
                            file.write('%.6f\n'%data2[k,n,i])


    #    for i in range(0,no_elem):
    #        file.write('11\n')
           
    
    
        
        file.close()
        return



    def make_asc_to_vtk(self,filename,filename2,v=0):
        """
        Parameters
        ----------
        filename : TYPE
            input filename with asc.
        filename2 : TYPE
            output filename.
        v : TYPE, optional
            Choose to plot as 3d voces or as surface. The default is 0 (3d).

        Returns
        -------
        None.

        """
        
        
        header=np.genfromtxt(filename,max_rows=6)
        header=header[:,1]
        
        nx = header[0]
        ny = header[1]
        ulx = header[2]
        uly = header[3]
        xres = header[4]
        yres=xres
        nan_value=header[5]
        
        
        x_coords = ulx + np.arange(0,nx) * xres +  (xres / 2) #add half the cell size
        y_coords = uly + np.arange(0,ny) * yres +  (yres / 2) #add half the cell size
        
        raster=np.genfromtxt(filename,skip_header=6)
        raster[raster==nan_value]=np.nan
        y_coords=np.flipud(y_coords)
        raster=np.reshape(raster,(np.int32(ny),np.int32(nx)))
        if v==0:
            self.tiff_to_vtk_3d_data(filename2,raster,x_coords,y_coords,xres,yres)
        else:
            # Alternativly, you can plot is as a surface (usefull when you want to embades another image)
            self.tiff_to_vtk_data(filename2,raster,x_coords,y_coords,xres,yres)
        
        
        
        return
    
    def tiff_to_vtk_3d_data_up_down(self,filename,band1,x_coords,y_coords,xres,yres,band2,val,kv,kh,c,kd):
        """
        Provide a geotiff and get a dem model with the same name as vtk file. 
        Notice, in this version we do no support cooridante conversion. Future versions will
        YOu can pass additinal arguments

        """        
        
        
        # raster = gdal.Open(filename)
        # ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
        # x_coords = ulx + np.arange(0,raster.RasterXSize) * xres +  (xres / 2) #add half the cell size
        # y_coords = uly + np.arange(0,raster.RasterYSize) * yres +  (yres / 2) #add half the cell size
        
        if yres<0:
            y_coords=np.flipud(y_coords)
        
        # Xc, Yc = np.meshgrid(x_coords,y_coords);
        # band1 = raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
        band1[band1>1e6]=np.nan
        
        # band1=pd.DataFrame(band1)
        # band1=band1.interpolate().values
        # band1[np.isnan(band1)]=np.nanmin(band1)

                    
        
        
        if np.abs(xres)!=np.abs(yres):
            sys.exit('Error. dx and dy are different. Adjust the code\\')
        else:    
            dx=xres
            # plt.imshow(band1)
            band1=band1.T
            band2=band2.T
            
            kv=kv.T
            kh=kh.T
            c=c.T
            kd=kd.T
            
            
            number_elements=np.count_nonzero(np.isfinite(band1.ravel()))
            number_elements2=np.count_nonzero(np.isfinite(band2.ravel()))
            if number_elements!=number_elements2:
                sys.exit("ERROR: TOP and BOT have different elements")   
            if number_elements==0:
                print("No data in region found") 
            else:
           
                file=open(filename,'w')
                file.write('# vtk DataFile Version 1.0\n')
                file.write('%s\n'%filename)               
                file.write('ASCII\n')
                file.write('\n')
                file.write('DATASET UNSTRUCTURED_GRID\n')
                file.write('POINTS %d float\n'%(8*(number_elements)))
                
                
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            x1=x_coords[i]-dx/2
                            y1=y_coords[j]-dx/2
                            z1=band1[i,j]
                            
                            x2=x_coords[i]+dx/2
                            y2=y_coords[j]+dx/2
                            z2=band2[i,j]
                            
                            # z2=np.nanmin(band[i-1:i+1,j-1:j+1])
                            
                            # z2=np.nanmin(band1.ravel())-0.1
                            
                            
                            file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                                       (x1,y1,z1,
                                        x1,y2,z1,
                                        x2,y2,z1,
                                        x2,y1,z1,
                                        x1,y1,z2,
                                        x1,y2,z2,
                                        x2,y2,z2,
                                        x2,y1,z2))
                        
                file.write('\n')
                file.write('CELLS %d %d\n'%(number_elements,9*(number_elements)))
                base=np.array([4,7,6,5,0,3,2,1])
                k=0
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+k*8,base[1]+k*8,base[2]+k*8,base[3]+k*8,base[4]+k*8,base[5]+k*8,base[6]+k*8,base[7]+k*8))
                            k=k+1
                file.write('\n')
                file.write('CELL_TYPES %d\n'%(number_elements))    
                for i in range(0,number_elements):
                    file.write('12\n')
                        
                file.write('\n')
                file.write('CELL_DATA %d\n'%(number_elements))    
                file.write('SCALARS UNIT float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('%.2f\n'%val)          
                
                file.write('SCALARS verticale_doorlatendheid float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            if np.isfinite(kv[i,j]):
                                file.write('%.2f\n'%kv[i,j])                                                   
                            else:
                                file.write('%.2f\n'%0)                                                   
                
                file.write('SCALARS Horizontale_doorlatendheid float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            if np.isfinite(kh[i,j]):
                                file.write('%.2f\n'%kh[i,j])                                                   
                            else:
                                file.write('%.2f\n'%0)        

                file.write('SCALARS Weerstand float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            if np.isfinite(c[i,j]):
                                file.write('%.2f\n'%c[i,j])                                                   
                            else:
                                file.write('%.2f\n'%0)        

                file.write('SCALARS transmissiviteit float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            if np.isfinite(kd[i,j]):
                                file.write('%.2f\n'%kd[i,j])                                                   
                            else:
                                file.write('%.2f\n'%0)                   


                
                file.close()          
        return
    def tiff_to_vtk_3d_data_up_down_small(self,filename,band1,x_coords,y_coords,xres,yres,band2,val,val2):
        """
        Provide a geotiff and get a dem model with the same name as vtk file. 
        Notice, in this version we do no support cooridante conversion. Future versions will
        YOu can pass additinal arguments

        """        
        
        
        # raster = gdal.Open(filename)
        # ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
        # x_coords = ulx + np.arange(0,raster.RasterXSize) * xres +  (xres / 2) #add half the cell size
        # y_coords = uly + np.arange(0,raster.RasterYSize) * yres +  (yres / 2) #add half the cell size
        
        if yres<0:
            y_coords=np.flipud(y_coords)
        
        # Xc, Yc = np.meshgrid(x_coords,y_coords);
        # band1 = raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
        band1[band1>1e6]=np.nan
        
        # band1=pd.DataFrame(band1)
        # band1=band1.interpolate().values
        # band1[np.isnan(band1)]=np.nanmin(band1)

                    
        
        
        if np.abs(xres)!=np.abs(yres):
            sys.exit('Error. dx and dy are different. Adjust the code\\')
        else:    
            dx=xres
            # plt.imshow(band1)
            band1=band1.T
            band2=band2.T
            val=val.T
            val2=val2.T

            
            
            number_elements=np.count_nonzero(np.isfinite(band1.ravel()))
            number_elements2=np.count_nonzero(np.isfinite(band2.ravel()))
            if number_elements!=number_elements2:
                sys.exit("ERROR: TOP and BOT have different elements")   
            if number_elements==0:
                print("No data in region found") 
            else:
           
                file=open(filename,'w')
                file.write('# vtk DataFile Version 1.0\n')
                file.write('%s\n'%filename)               
                file.write('ASCII\n')
                file.write('\n')
                file.write('DATASET UNSTRUCTURED_GRID\n')
                file.write('POINTS %d float\n'%(8*(number_elements)))
                
                
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            x1=x_coords[i]-dx/2
                            y1=y_coords[j]-dx/2
                            z1=band1[i,j]
                            
                            x2=x_coords[i]+dx/2
                            y2=y_coords[j]+dx/2
                            z2=band2[i,j]
                            
                            # z2=np.nanmin(band[i-1:i+1,j-1:j+1])
                            
                            # z2=np.nanmin(band1.ravel())-0.1
                            
                            
                            file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                                       (x1,y1,z1,
                                        x1,y2,z1,
                                        x2,y2,z1,
                                        x2,y1,z1,
                                        x1,y1,z2,
                                        x1,y2,z2,
                                        x2,y2,z2,
                                        x2,y1,z2))
                        
                file.write('\n')
                file.write('CELLS %d %d\n'%(number_elements,9*(number_elements)))
                base=np.array([4,7,6,5,0,3,2,1])
                k=0
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+k*8,base[1]+k*8,base[2]+k*8,base[3]+k*8,base[4]+k*8,base[5]+k*8,base[6]+k*8,base[7]+k*8))
                            k=k+1
                file.write('\n')
                file.write('CELL_TYPES %d\n'%(number_elements))    
                for i in range(0,number_elements):
                    file.write('12\n')
                        
                file.write('\n')
                file.write('CELL_DATA %d\n'%(number_elements))    
                file.write('SCALARS UNIT float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('%.2f\n'%val[i,j])          
                file.write('SCALARS Probability float\n') 
                file.write('LOOKUP_TABLE custom_table\n')    
                for i in range(0,x_coords.shape[0]):
                    for j in range(0,y_coords.shape[0]):
                        if np.isfinite(band1[i,j]):
                            file.write('%.2f\n'%val2[i,j])                     
              


                
                file.close()          
        return
        
    def make_asc_to_vtk_as_surface(self,filename,filename2,trim=0.01):
        """
        Parameters
        ----------
        filename : TYPE
            input filename with asc.
        filename2 : TYPE
            output filename.
        
            

        Returns
        -------
        None.

        """
        
        
        header=np.genfromtxt(filename,max_rows=6)
        header=header[:,1]
        
        nx = header[0]
        ny = header[1]
        ulx = header[2]
        uly = header[3]
        xres = header[4]
        yres=xres
        nan_value=header[5]
        
        
        x = ulx + np.arange(0,nx) * xres +  (xres / 2) #add half the cell size
        y = uly + np.arange(0,ny) * yres +  (yres / 2) #add half the cell size
        
        raster=np.genfromtxt(filename,skip_header=6)
        raster[raster==nan_value]=np.nan
        y=np.flipud(y)
        
        

        
        if (len(x)==len(y))  & (len(x)==len(elev)) :
            #all is good
            i=1
        else:
            sys.exit('Error: x,y,z shuould have the same dimensions')
        
        
        # Check id dpths given are negative
        if np.min(data[:,1]<0):
            d_flag=1
            data[:,1]=np.abs(data[:,1])
        else:
            d_flag=0
        
        
        
        # use only area we have data
        xc,yc=np.mehsgrid(x,y)
        raster=raster.ravel()
        ix=np.where(np.isfinite(raster))
        
        xc=xc.ravel()
        yc=yc.ravel()
        
        data=np.c_[xc[ix],yc[ix],raster[ix]]
        
        
        triang = mpl.tri.Triangulation(data[:,0], data[:,1])
        mask = mpl.tri.TriAnalyzer(triang).get_flat_tri_mask(trim)
        triang.set_mask(mask)

        # plt.triplot(triang)
        nodes_x=triang.x 
        nodes_y=triang.y
        
        tr=triang.triangles
        
        # find center of triangles for interpolation
        x_tri=1/3*(nodes_x[tr[:,0]]+nodes_x[tr[:,1]]+ nodes_x[tr[:,2]])
        z_tri=1/3*(nodes_y[tr[:,0]]+nodes_y[tr[:,1]]+ nodes_y[tr[:,2]])

        
        # f = interp2d(data[:,0], data[:,1], data[:,2], kind='linear')
        f=CloughTocher2DInterpolator(np.c_[data[:,0], data[:,1]], data[:,2])
        new_val=f(x_tri,z_tri)
        
        
        
        no_polygons=np.count_nonzero(~(mask))  
        
        # find center of triangles
      
        
        
        
        
        if len(x)>1:# We have coordinates
            # find actual coordantes to rotate
            t=np.sqrt(np.power(x[1:]-x[:-1],2) + np.power(y[1:]-y[:-1],2)     )
            dis=np.r_[0,np.cumsum(t)]  
            # interpoltate to find starting point
            fx=interp1d(dis,x,kind='linear',fill_value='extrapolate')
            fy=interp1d(dis,y,kind='linear',fill_value='extrapolate')
            fz=interp1d(dis,elev,kind='linear',fill_value='extrapolate')
            # split the section in equal spacings
            # t=np.sqrt(np.power(x[1:]-x[:-1],2) + np.power(y[1:]-y[:-1],2)     )
            # dis_mew=np.r_[0,np.cumsum(t)]  
            #find coordiantes
            xx=fx(nodes_x)
            yy=fy(nodes_x)
            elev=fz(nodes_x)    
        else:
            xx=nodes_x
            yy=0*xx
            elev=0*xx
      
        
        # base=np.array([0,1,2,3])
        file=open(filename,'w')
        file.write('# vtk DataFile Version 4.2\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(nodes_x.shape[0]))
        
        for k in range(0,nodes_x.shape[0]):
            file.write('%.2f %.2f %.2f\n'%(xx[k],yy[k],-(nodes_y[k]-elev[k])))

        
        file.write('CELLS %d %d\n'%(no_polygons,4*no_polygons))   
        for k in range(0,tr.shape[0]):
            if mask[k]==False:
                file.write('3 %d %d %d\n'%(tr[k,0],tr[k,1],tr[k,2]))
     
        file.write('CELL_TYPES %d\n'%((no_polygons)))
        for k in range(0,tr.shape[0]):
            if mask[k]==False:
                file.write('5\n')                    
        file.write('CELL_DATA %d\n'%((no_polygons)))
        file.write('SCALARS Property_1 double\n')
        file.write('LOOKUP_TABLE default\n')        
        for k in range(0,tr.shape[0]):
            if mask[k]==False:
                file.write('%.3f\n'%new_val[k])     
    
        file.close()
        return
        
        
        return        
        
    def make_3d_grid_from_xy(self,filename,xc,yc,zc,bot,dx,dy,name='Parameter'):
        """
        Provide a 3D grided data, that represent a a 3d volume.
        dim[2] is the depth 
        Provide the coordate of this profile, on a map. 
        xc,yc,zc are the coordantes
        
        :data mxnxk matrix
        :xc nx1 vector with cooridantes along the x-axis
        :yc mx1 vector with cooridantes along the y-axis
        :zc (k+1)x1 vector with elevation (it must be data.shape[0 dimensions] +1)
        :elev mxn (OPTIONAL if you have topograpy in grid) 
        """            
        # dx=np.abs(xc[1]-xc[0])           
        # dy=np.abs(yc[1]-yc[0])           
        
        no_elem=len(xc)
        file=open(filename,'w')
        file.write('# vtk DataFile Version 1.0\n')
        file.write('%s\n'%filename)               
        file.write('ASCII\n')
        file.write('\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')
        file.write('POINTS %d float\n'%(8*(no_elem)))    
        

        for k in range(0,len(xc)):
            x1=xc[k]-dx/2
            x2=xc[k]+dx/2
            
            y1=yc[k]-dy/2
            y2=yc[k]+dy/2
                                 
            
            z1=zc[k]
            z2=bot


    
            file.write('%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                       (x1,y1,z1,
                        x1,y2,z1,
                        x2,y2,z1,
                        x2,y1,z1,
                        x1,y1,z2,
                        x1,y2,z2,
                        x2,y2,z2,
                        x2,y1,z2))
        file.write('\n')
                        
    #    no_elem=38                
        file.write('CELLS %d %d\n'%(no_elem,9*(no_elem)))
        base=np.array([4,7,6,5,0,3,2,1])
        for i in range(0,no_elem):
            file.write('8 %d %d %d %d %d %d %d %d\n'%(base[0]+i*8,base[1]+i*8,base[2]+i*8,base[3]+i*8,base[4]+i*8,base[5]+i*8,base[6]+i*8,base[7]+i*8))
        file.write('\n')
        file.write('CELL_TYPES %d\n'%(no_elem))    
        for i in range(0,no_elem):
            file.write('12\n')
                     
        file.write('\n')
        file.write('CELL_DATA %d\n'%(no_elem))    
        file.write('SCALARS %s float\n'%name)    
        file.write('LOOKUP_TABLE custom_table\n')    
        for k in range(0,len(xc)):
            file.write('%.6f\n'%zc[k])

           
    
    
        
        file.close()
        return

          
        