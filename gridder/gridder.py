# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:59:51 2019

@author: karaouli
"""
import sys
import numpy as np
from matplotlib import path
import pandas as pd
from skimage.restoration import inpaint
import matplotlib.pyplot as plt
import os
#import 


class Geo_Gridder:
    def __init__(self,training_poins,training_data,method='mean'):
        """
        Initialize gridder instance

        Parameters
        ----------
        training_poins : Nx2 matrix
            DESCRIPTION. The x,y coordinates of the data.
        training_data : Nx! vector
            DESCRIPTION. The value.
        method : TYPE, optional
            DESCRIPTION. The default is 'mean'.You can choose for the griider
            "mean" or "average" ot "mode"

        Returns
        -------
        None.

        """

        
        if method not in ['mean','average','data','mode']:
            sys.exit("ERROR: Interpolation method: " + str(method) + " does not exist")
                    

            
        # define variables
        self.training_points = training_poins  # training points
        self.training_data = training_data  # data at the training points
#        self.int = []  # interpolation method
        self.prediction_points = []  # prediction points
        self.data_predict = []  # data at the prediction points
#        self.nb_points = []  # number of points in the prediction grid
        self.df=[] # make Dataframe to gtoub
        self.xs=[] #make the x for the grid
        self.ys=[] #make the y for the grid
        self.xg=[] #x-grid
        self.yg=[] #y-grid
        self.lin_index=[] # make the index
        self.method=method
        self.bs=[] # the gridded data ouput
        self.count=[]
        # self.bs2=[]
        
        return
        
    def mode(self,df, key_cols, value_col, count_col):
        '''                                                                                                                                                                                                                                                                                                                                                              
        Pandas does not provide a `mode` aggregation function                                                                                                                                                                                                                                                                                                            
        for its `GroupBy` objects. This function is meant to fill                                                                                                                                                                                                                                                                                                        
        that gap, though the semantics are not exactly the same.                                                                                                                                                                                                                                                                                                         
    
        The input is a DataFrame with the columns `key_cols`                                                                                                                                                                                                                                                                                                             
        that you would like to group on, and the column                                                                                                                                                                                                                                                                                                                  
        `value_col` for which you would like to obtain the mode.                                                                                                                                                                                                                                                                                                         
    
        The output is a DataFrame with a record per group that has at least one mode                                                                                                                                                                                                                                                                                     
        (null values are not counted). The `key_cols` are included as columns, `value_col`                                                                                                                                                                                                                                                                               
        contains a mode (ties are broken arbitrarily and deterministically) for each                                                                                                                                                                                                                                                                                     
        group, and `count_col` indicates how many times each mode appeared in its group.                                                                                                                                                                                                                                                                                 
        '''
        return df.groupby(key_cols + [value_col]).size() \
                 .to_frame(count_col).reset_index() \
                 .sort_values(count_col, ascending=False) \
                 .drop_duplicates(subset=key_cols)
    
    def modes(self,df, key_cols, value_col, count_col):
        '''                                                                                                                                                                                                                                                                                                                                                              
        Pandas does not provide a `mode` aggregation function                                                                                                                                                                                                                                                                                                            
        for its `GroupBy` objects. This function is meant to fill                                                                                                                                                                                                                                                                                                        
        that gap, though the semantics are not exactly the same.                                                                                                                                                                                                                                                                                                         
    
        The input is a DataFrame with the columns `key_cols`                                                                                                                                                                                                                                                                                                             
        that you would like to group on, and the column                                                                                                                                                                                                                                                                                                                  
        `value_col` for which you would like to obtain the modes.                                                                                                                                                                                                                                                                                                        
    
        The output is a DataFrame with a record per group that has at least                                                                                                                                                                                                                                                                                              
        one mode (null values are not counted). The `key_cols` are included as                                                                                                                                                                                                                                                                                           
        columns, `value_col` contains lists indicating the modes for each group,                                                                                                                                                                                                                                                                                         
        and `count_col` indicates how many times each mode appeared in its group.                                                                                                                                                                                                                                                                                        
        '''
        return df.groupby(key_cols + [value_col]).size() \
                 .to_frame(count_col).reset_index() \
                 .groupby(key_cols + [count_col])[value_col].unique() \
                 .to_frame().reset_index() \
                 .sort_values(count_col, ascending=False) \
                 .drop_duplicates(subset=key_cols)        
    
    
    
    
    def make_grid(self,xmin=None,xmax=None,
                  ymin=None,ymax=None,dx=None,dy=None,pts=None ):
        """
        Calls the make grid. Grid is based either on user input,
        by providing custom xmin,xmax,ymin,ymax.
        If they are not provided, then they will be based on the
        input data.

        Parameters
        ----------
        xmin : FLOAT, optional
            DESCRIPTION. Provide you own grid dimension. The default is None.
        xmax : FLOAT, optional
            DESCRIPTION. The default is None.
        ymin : FLOAT, optional
            DESCRIPTION. The default is None.
        ymax : FLOAT, optional
            DESCRIPTION. The default is None.
        dx : FLOAT, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        

        
        if xmin==None:
            xmin=np.nanmin(self.training_points[:,0])
        if xmax==None:
            xmax=np.nanmax(self.training_points[:,0])
        if ymin==None:
            ymin=np.nanmin(self.training_points[:,1])
        if ymax==None:
            ymax=np.nanmax(self.training_points[:,1])            
        if dx==None:
            dx=1
            dy=1

        # warning, I have x and y different oriented
        if pts is not None:
            self.xs = np.linspace(xmin, xmax, pts[1])
            self.ys = np.linspace(ymin, ymax, pts[0])
            dx=self.xs[1]-self.xs[0]
            dy=self.ys[1]-self.ys[0]
        else:
            self.xs=np.arange(xmin,xmax+dx,dx)
            self.ys=np.arange(ymin,ymax+dy,dy)
        
        self.xg,self.yg=np.meshgrid(self.xs,self.ys)
        i1=np.int32(np.floor_divide(self.training_points[:,0]-xmin,dx))
        i2=np.int32(np.floor_divide(self.training_points[:,1]-ymin,dy))
        # keep only what's in the boundary. Points excaclty on boundary, are removed
        ix=np.where((i1>=0) & (i1<self.xs.shape[0]) & (i2>=0) & (i2<self.ys.shape[0])   )[0]
        
        self.lin_index=(i1[ix])*(self.ys.shape[0])  +(i2[ix]) 
        # make Dataframe with akk 
        self.df=pd.DataFrame({'values':self.training_data[ix],'ii':self.lin_index})
        
        
        return

    

        
        
        

    def gridder(self):
        """
        Calls the gridder by method. Data are now gridded in the bs matrix.
        Matrix count has the count values of the average data

        Returns
        -------
        None.

        """




        
        self.bs=np.nan*np.zeros(((self.xs.shape[0])*(self.ys.shape[0]),1))
        self.count=np.zeros(((self.xs.shape[0])*(self.ys.shape[0]),1))
        if self.method=='mean_fast':
            bi=self.df.groupby('ii').mean()
            self.bs[bi.index.values]=np.reshape(bi['values'].values,(bi.shape[0],1))
        elif self.method=='mean':
            bi=self.df.groupby('ii').agg(['count','mean']).reset_index()
            
            self.bs[bi['ii'].values]=np.reshape(bi['values']['mean'].values,(bi.shape[0],1))            
            self.count[bi['ii'].values]=np.reshape(bi['values']['count'].values,(bi.shape[0],1))            
        elif self.method=='average_fast':
            bi=self.df.groupby('ii').median()
            self.bs[bi.index.values]=np.reshape(bi['values'].values,(bi.shape[0],1))
        elif self.method=='average':
            bi=self.df.groupby('ii').median()
            self.bs[bi.index.values]=np.reshape(bi['values'].values,(bi.shape[0],1))
        elif self.method=='mode':
            bi=self.mode(self.df, ['ii'], 'values', 'count')
            self.bs[bi['ii'].values]=np.reshape(bi['values'].values,(bi.shape[0],1))       
        
        

        self.bs=np.reshape(self.bs,((self.xs.shape[0],self.ys.shape[0])))
        self.count=np.reshape(self.count,((self.xs.shape[0],self.ys.shape[0])))
        self.bs=(self.bs.T)
        self.count=(self.count)
        # self.bs=bi
        # self.bs2=bi2          
        return
        
        
    def in_paint(self,external_polygon=None,no_x=1756,no_y=1027,buffer=25):
        """
        Inpaint the gridded data. 

        Parameters
        ----------
        external_polygon : MATRIX, optional
            DESCRIPTION. If you want to only inpaint in area defeined in the polygon.
            The default is None.
        no_x : TYPE, optional
            DESCRIPTION. The default is 200. This depeneds on the memory availabe.
            If it does not fit in memory, we split the data in batches
        no_y : TYPE, optional
            DESCRIPTION. The default is 200. This depeneds on the memory availabe.
            If it does not fit in memory, we split the data in batches
        buffer : TYPE, optional
            DESCRIPTION. The default is 25. This defines the overlay between
            two batches

        Returns
        -------
        None.

        """
        # This is the mask to be inpainted. If no polygon provided, inpaint everywhere
        mask=np.ones(self.xg.size)

        
        if external_polygon is not None:
            mask=np.zeros(self.xg.size)
            # make sure that it is a closed polygon
            gee=np.r_[external_polygon,np.c_[external_polygon[0,0],external_polygon[0,1]]]
            p2=path.Path(gee)
            # This makes a mask 
            flags = p2.contains_points(np.hstack((self.xg.flatten()[:,np.newaxis],self.yg.flatten()[:,np.newaxis])))
            mask[flags==True]=1 # if there is a polygon, do not inpaint out of it

            

        # makemask  it into matrix
        mask=np.reshape(mask,self.xg.shape)
        self.mask=np.copy(mask) # keep the original mask
        # do not inpaint where we have data
        mask[np.isfinite(self.bs)]=0

        
        
        
        
        # plt.hist(self.mask.ravel())
        # plt.title('Points to be inpainted')
        print("No of points:%d, data points:%d, inpainted data:%d\n"%(self.bs.size,self.training_data.shape[0],np.count_nonzero(mask)))
        x3=np.arange(0,mask.shape[0],no_x) #FUTURE automatic make estimation
        y3=np.arange(0,mask.shape[1],no_y)
    
        # # plt.imshow(mask)
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        # plt.imshow(mask)
        ee=np.copy(self.bs)
        ee[np.isnan(ee)]=0  # This requires thinking!!!! Why 0, perhaps a better choice is to use up and down boundaties
        lll=0
        for i in range(0,x3.shape[0]):
            for j in range(0,y3.shape[0]):
                print('Inpainting....')
                
                if i==0:
                    buff_l=0
                    buff_r=buffer
                elif i==x3.shape[0]-1:
                    buff_l=buffer
                    buff_r=x3.shape[0]        
                else:
                    buff_l=buffer
                    buff_r=buffer
                    
        
                            
                if j==0:
                    buff_u=0
                    buff_d=buffer
                elif j==y3.shape[0]-1:
                    buff_u=buffer
                    buff_d=y3.shape[0]        
                else:
                    buff_u=buffer
                    buff_d=buffer
        
        
        
        
        
        
                if i<x3.shape[0]-1:
                    l=x3[i]
                    r=x3[i+1]
                else:
                    l=x3[i]
                    r=self.bs.shape[0]
                if j<y3.shape[0]-1:
                    u=y3[j]
                    d=y3[j+1]
                else:
                    u=y3[j]
                    d=self.bs.shape[1]   
                
                
                
        #        plt.clf()
                # From the start mask, keep inpaiinting. Will fix in new version
                mask2=np.zeros(mask.shape)
                mask2[(l-buff_l):(r+buff_r),(u-buff_u):(d+buff_d)]=1
                mask2[mask==0]=0
                mask2[np.isnan(mask)]=0
                mask2[np.nonzero(ee)]=0
                mask[l:r,u:d]=0 # for the next iteration, do not inpaint again
        
       
        
                # plt.plot(np.r_[d,u,u,d,d],np.r_[l,l,r,r,l])
                ee = np.ascontiguousarray(ee) 

                del1=inpaint.inpaint_biharmonic((ee), np.uint8(mask2), channel_axis=None) 
                
        #        plt.imshow(del1,vmin=0,vmax=255,cmap='rainbow')
                ee[l:r,u:d]=del1[l:r,u:d]
    
            
        ee[self.mask==0]=np.nan
        self.prediction_data=ee    
        
        return



    def in_paint2(self,external_polygon=None):
        """
        Inpaint the gridded data. 

        Parameters
        ----------
        external_polygon : MATRIX, optional
            DESCRIPTION. If you want to only inpaint in area defeined in the polygon.
            The default is None.
        no_x : TYPE, optional
            DESCRIPTION. The default is 200. This depeneds on the memory availabe.
            If it does not fit in memory, we split the data in batches
        no_y : TYPE, optional
            DESCRIPTION. The default is 200. This depeneds on the memory availabe.
            If it does not fit in memory, we split the data in batches
        buffer : TYPE, optional
            DESCRIPTION. The default is 25. This defines the overlay between
            two batches

        Returns
        -------
        None.

        """
        # This is the mask to be inpainted. If no polygon provided, inpaint everywhere
        mask=np.ones(self.xg.size)

        
        if external_polygon is not None:
            # mask=np.zeros(self.xg.size)
            # make sure that it is a closed polygon
            gee=np.r_[external_polygon,np.c_[external_polygon[0,0],external_polygon[0,1]]]
            p2=path.Path(gee)
            # This makes a mask 
            flags = p2.contains_points(np.hstack((self.xg.flatten()[:,np.newaxis],self.yg.flatten()[:,np.newaxis])))
            # mask[flags==True]=1 # if there is a polygon, do not inpaint out of it
            flags=np.reshape(flags,self.xg.shape)

            

        # makemask  it into matrix
        mask=np.reshape(mask,self.xg.shape)
        self.mask=np.copy(mask) # keep the original mask
        # do not inpaint where we have data
        mask[np.isfinite(self.bs)]=0

        
        
        
        
        # plt.hist(self.mask.ravel())
        # plt.title('Points to be inpainted')
        print("No of points:%d, data points:%d, inpainted data:%d\n"%(self.bs.size,self.training_data.shape[0],np.count_nonzero(mask)))

    
        ee=np.copy(self.bs)
        ee[np.isnan(ee)]=0  # This requires thinking!!!! Why 0, perhaps a better choice is to use up and down boundaties

        ee = np.ascontiguousarray(ee) 

        ee=inpaint.inpaint_biharmonic(ee, mask, channel_axis=None) 
                


    
            
        ee[self.mask==0]=np.nan
        if external_polygon is not None:
            ee[flags==False]=np.nan
        self.prediction_data=ee    
        
        return



    def plot_2D(self, output_name, validation=None, show=True):
        """
        Plots the results for a 2D field

        :param output_name: file name of the plot
        :param validation: (optional) validation data. if not False the validation data and error are plotted.
                           Default is False. The size of the validation dataset must be the same as the interpolated
        :param show: bool to show the figure (default True). if set to false saves the figure
        :return:
        """

        # check if the folder for the results exist. if not creates it
        if not os.path.isdir(os.path.dirname(output_name)):
            os.makedirs(os.path.dirname(output_name))

        # create the grid
        grid_xn = self.xg
        grid_yn = self.yg
        # reshape the data
        data = self.prediction_data

        # create fig
        fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        # plot the training points
        im1 = ax[0, 0].scatter(self.training_points[:, 0], self.training_points[:, 1], c=self.training_data,
                               vmin=np.min(self.training_data), vmax=np.max(self.training_data),
                               cmap="jet", edgecolor='k', marker="o")
        cbar = plt.colorbar(im1, ax=ax[0, 0])
        cbar.ax.set_ylabel('Samples')
        ax[0, 0].grid()
        ax[0, 0].set_ylabel("Y coordinate")
        # plot the interpolated data
        ax[1, 0].pcolor(grid_xn, grid_yn, data, vmin=np.min(self.training_data), vmax=np.max(data), cmap="jet")
        im2 = ax[1, 0].scatter(self.training_points[:, 0], self.training_points[:, 1], c=self.training_data,
                               vmin=np.min(self.training_data), vmax=np.max(self.training_data),
                               cmap="jet", edgecolor='k', marker="o")
        cbar = plt.colorbar(im2, ax=ax[1, 0])
        cbar.ax.set_ylabel('Interpolation')

        ax[1, 0].set_xlabel("X coordinate")
        ax[1, 0].set_ylabel("Y coordinate")
        ax[1, 0].grid()

        # if validation dataset is available
        if validation is not None:
            # check size of validation dataset
            if len(validation.ravel()) != len(self.prediction_data.ravel()):
                sys.exit("ERROR: length of the validation dataset is different from the length of the prediction dataset")

            # compute the relative error
            abs_error = np.abs(self.prediction_data - validation)
            # reshape error for grid
            # abs_error = abs_error.reshape(self.nb_points)
            # plot the exact function
            im3 = ax[0, 1].pcolor(grid_xn, grid_yn, validation,
                                  vmin=np.min(validation), vmax=np.max(validation), cmap="jet")
            cbar = plt.colorbar(im3, ax=ax[0, 1])
            cbar.ax.set_ylabel("Validation")
            ax[0, 0].grid()
            # plot the error
            im4 = ax[1, 1].pcolor(grid_xn, grid_yn, abs_error,
                                  vmin=0, vmax=np.max(abs_error), cmap="jet")
            cbar = plt.colorbar(im4, ax=ax[1, 1])
            cbar.ax.set_ylabel("|Error|")

            ax[1, 1].set_xlabel("X coordinate")
            ax[1, 1].grid()
        else:
            fig.delaxes(ax[0, 1])
            fig.delaxes(ax[1, 1])

        # if show is true -> show figure
        if show:
            plt.show()
        # else -> save figure
        else:
            plt.savefig(output_name)
            plt.close()
        return 
        