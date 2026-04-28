# -*- coding: utf-8 -*-
"""
In this demo script, we will read several files that contain geological boreholes
Created on Thu Feb  6 13:37:43 2020

@author: karaouli

First things. Typically, each borehole is associated with a sinlge csv file.
That file has at least two columns, one that shows depth and one that shows the 
geological unit. See the data\Boreholes folder

I.e. borehole F1, is on F1.txt and it reads
Base below surface;lithology
2;Silty clay
3; silty clay loam
3.5; silty clay
5: Sand

As you can see, this is a simple csv file.
1st row is a header. This program expectes a header, seperated by semi-columns (;)
The order is importnant. 1st column based my depth from SURFACE.
2nd columnd is the desciption.

Now, notice that vtk files, cannot have strings as classes. They have to be numbers.
In other words, we have to convert the lithological type, to a seperate UNIQUE number.
Some of the lithological units are unique per borehole (i.e. F1 has silty clay loam , but F2 does not).
If you have multiple files (boreholes), befoire you assigne unique number, you
have to finb how many unique lithology classes are. 



Now, the coordaintes of each boreholes, as also the elevation, comes with a dirrent file.
In this case, the file ERT_boreholesxyz.xlsx is a file with 4 columns
1st row is a header
 
1st column is the name of the borehole. IT IS IMPORTANT that the name in this column, 
is the same with the filename (without extension) of the borehole file. If they are
different, it will not work.
i..e MH1 on the xlsx file, correspnts to a file MH1.txt

2nd column has the x-coordinate of the borehole
3rd column has the y-coordate of the borehole
4th column the elevation


This is what we do in this script.

"""


import glob
import pandas as pd
import numpy as np
import os
from vtkclass import VtkClass

# First with pandas, read the file with the coordinates per borehole
cpt_coords=pd.read_excel(r'..\data\Boreholes\ERT_boreholesxyz.xlsx')
# then, lets read all the files in the borehole directory
yy=glob.glob(r'..\data\Boreholes\*.txt')

# We will save all boreholes in a pandas dataframe
data=[]

# Our task is to sync the coordinate file, with the same order we read the txt files
# In other words, the order you have save the coordinates in the xls file,
# does not matter. The only import is the name on the xlsx file, be the same 
# with the txt. Since these are fileanmes, the filename is case sensitive.
#i.e. f1.txt and F1.txt are different files.

va=np.zeros((len(yy),3)) # this matrix has the correct order for the coordaintes
t=0
for i in yy:
    data.append(pd.read_csv(i,sep=';|:',engine='python'))
    a=(os.path.splitext(os.path.basename(i)))[0]
    ix=cpt_coords.loc[cpt_coords['Name']==a]
    va[t,:]=ix.iloc[:,1:].values
    t=t+1

# in our example, we synced the xlsx file and the txt files
# and now the matrix va has the correct order.
    

# Next task is to find how many different lithological files exists, so to 
# assign a unique class.
# This is not case sensitiye (i.e. sand and SaNd or SAND) is the same class.
# also it will eliminate spaces before and fter the name
# It will not though check for spelling.
# If you have type sand and san, then this is a diffrent class

tmp_lith=[]
for i in range (0,len(data)):
    tmp_lith.append(data[i]['lithology'])
    
tmp_lith=(pd.concat(tmp_lith).str.strip()).str.upper().unique()    
# now we have a temporary matrix with all unique class found from all fiies.
# This is the bases from the classes


# now that we know the unique class number, we will scan all boreholes
# and assign the correct number

# Convert each borehole to number
for i in range(0,len(data)):
    datain=data[i]['lithology'].str.strip().str.upper()
    i0=np.zeros(datain.shape)
    for k in range(0,datain.shape[0]):
        i0[k]=np.argwhere(datain[k]==tmp_lith)
    data[i]['Class']=i0

# now, per borehole, we have append a column, that has assgin a class number.
# check the data dataframe 
print(data[0].columns)    


# Now, we are ready to generate one vtk file per borehole
# initialize our class
int1=VtkClass()

# the VTK class, expects a two colums (or more if you have more properies)
# 1st column is the depth, 2nd column is the number of class as a numpy array
# we save this a nx2 matrix (depth, class)
# next we need to provide the x,y coordiantes of the borehole
# then the radius to plot. This is user based, depneds on the number of boreholes,
# depths, etc. Try to see what looks good
# Finally, we neeed the elevation of the boreholes. If unkown, set it to zero




# loop through all borehole files
for i in range(0,cpt_coords.shape[0]):        
    current_borehole=data[i].iloc[:,[0,2]].values # this is the array with depth vs class number
    x_center=va[i,0]
    y_center=va[i,1]
    elev=va[i,2]
    radius=2
    
    int1.make_cylinder_borehole('..\\data\\vtk\\'+cpt_coords['Name'][i]+'.vtk',current_borehole, [x_center,y_center],radius,elev)

# now we have generated a file per borehole, we have to setup the colorscale/legend
# since color and geology is rather compliated, we will only export the class and 
# lithology type and user can set the colors in Paraview
file=open('..\\data\\vtk\\borehole_legend.txt','w')
file.write('Class Name; Class Id\n')
for i in range(0,tmp_lith.shape[0]):
    file.write('%s;%d\n'%(tmp_lith[i],i))
file.close()

# This file is self expenatory :) 
# Follow the next part of the tutorial to assign the approriate colors




# Please note in case you have millions of boreholes to plot, consider using make_borehole_as_cube
for i in range(0,cpt_coords.shape[0]):        
    current_borehole=data[i].iloc[:,[0,2]].values # this is the array with depth vs class number
    x_center=va[i,0]
    y_center=va[i,1]
    elev=va[i,2]
    radius=2
    
    int1.make_borehole_as_cube('..\\data\\vtk\\'+cpt_coords['Name'][i]+'_squared.vtk',current_borehole, [x_center,y_center],radius,elev)
