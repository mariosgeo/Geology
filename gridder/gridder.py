# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:59:51 2019

@author: karaouli
"""
from email.mime import image
import sys
import numpy as np
from matplotlib import path
import pandas as pd
from skimage.restoration import inpaint
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom
from matplotlib import rcParams
import matplotlib.patches as mpatches
from sklearn import metrics
from matplotlib.path import Path
from matplotlib import ticker
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
        self.uncertainty_data=[] # uncertainty model at the prediction points
        
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
    
    def make_mask(self,basis):
        mask = np.ones_like(basis)
        mask[np.isfinite(basis)] = 0
        #plt.imshow(mask)
        return mask 


    def weighted_inpaint_biharmonic(self,image, mask, x_weight=1.0, y_weight=1.0):
        image[np.isnan(image)] = 0  # Replace NaNs with zeros for inpainting
        #mask = make_mask(image)  # Create a mask where the image is NaN
        # Compute zoom factors (inverse of weights)
        zoom_factors = [y_weight, x_weight]
        # Rescale image and mask
        image_scaled = zoom(image, zoom_factors, order=1)
        mask_scaled = zoom(mask, zoom_factors, order=0)
        # Inpaint on the scaled image
        inpainted_scaled = inpaint.inpaint_biharmonic(image_scaled, mask_scaled, channel_axis=None)
        # Rescale back to original shape
        inpainted = zoom(inpainted_scaled, [1/y_weight, 1/x_weight], order=1)
        # Crop to original shape (in case of rounding)
        inpainted = inpainted[:image.shape[0], :image.shape[1]]
        return inpainted
    
    def weighted_inpaint_biharmonic_3d(self,image, mask, x_weight=1.0, y_weight=1.0, z_weight=1.0):
        image[np.isnan(image)] = 0  # Replace NaNs with zeros for inpainting
        
        # Compute zoom factors (inverse of weights)
        zoom_factors = [z_weight, y_weight, x_weight]
        
        # Rescale image and mask
        image_scaled = zoom(image, zoom_factors, order=1)
        mask_scaled = zoom(mask, zoom_factors, order=0)
        
        # Inpaint on the scaled image
        inpainted_scaled = inpaint.inpaint_biharmonic(image_scaled, mask_scaled, channel_axis=None)
        
        # Rescale back to original shape
        inpainted = zoom(inpainted_scaled, [1/z_weight, 1/y_weight, 1/x_weight], order=1)
        
        # Crop to original shape (in case of rounding)
        inpainted = inpainted[:image.shape[0], :image.shape[1], :image.shape[2]]

        return inpainted

    def one_vs_all(self,image=None, x_weight=1.0, y_weight=3.0,external_polygon=None):
        
        if image is None:
            image=self.bs
        else:
            self.bs=image
        
        # find here unique labels in the image
        uni_class=np.unique(image.ravel()[~np.isnan(image.ravel())])
        # for every unique label, create a mask and inpaint
        del2=np.zeros((image.shape[0],image.shape[1],len(uni_class)))
        for i, label in enumerate(uni_class):
            #make binary        
            ee = np.float64(np.where(image.copy() == label, 1, 0))

            del2[:,:,i] = self.weighted_inpaint_biharmonic(ee.copy(), self.make_mask(image), x_weight=x_weight, y_weight=y_weight)

        #normalize values
        del3=np.zeros_like(del2)
        tmp=np.sum(del2,axis=2)
        for i in range(0,del3.shape[2]):
            del3[:,:,i]=del2[:,:,i]/tmp
    
        
        del4=np.argmax(del3,axis=2)  

        #map back to actual classes
        inverse_value_map = {}
        for i, label in enumerate(uni_class):
            inverse_value_map[i] = label        
        vectorized_inverse_map = np.vectorize(inverse_value_map.get)
        del4 = vectorized_inverse_map(del4)
        # calculate percentage of uncertainty
        # del3 contains the probabilities for each class    
        del_perc=100-100*np.max(del3,axis=2) 
        self.prediction_data = del4
        self.uncertainty_data = del_perc

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

    def one_vs_all_3d(self,image=None, x_weight=1.0, y_weight=1.0, z_weight=1.0,external_polygon=None):
        
        if image is None:
            image=self.bs
        else:
            self.bs=image
        # find here unique labels in the image
        uni_class=np.unique(image.ravel()[~np.isnan(image.ravel())])
        # for every unique label, create a mask and inpaint
        del2=np.zeros((image.shape[0],image.shape[1],image.shape[2],len(uni_class)))
        for i, label in enumerate(uni_class):
            #make binary        
            ee = np.float64(np.where(image.copy() == label, 1, 0))

            del2[:,:,:,i] = self.weighted_inpaint_biharmonic_3d(ee.copy(), self.make_mask(image), x_weight=x_weight, y_weight=y_weight,z_weight=z_weight)

        #normalize values
        del3=np.zeros_like(del2)
        tmp=np.sum(del2,axis=3)
        for i in range(0,del3.shape[3]):
            del3[:,:,:,i]=del2[:,:,:,i]/tmp
    
        
        del4=np.argmax(del3,axis=3)  

        #map back to actual classes
        inverse_value_map = {}
        for i, label in enumerate(uni_class):
            inverse_value_map[i] = label        
        vectorized_inverse_map = np.vectorize(inverse_value_map.get)
        del4 = vectorized_inverse_map(del4)
        # calculate percentage of uncertainty
        # del3 contains the probabilities for each class    
        del_perc=100-100*np.max(del3,axis=3) 
        self.prediction_data = del4
        self.uncertainty_data = del_perc
        return         
    
    

    def plot_model(self, cmap='viridis',filename='model.pdf',labels=np.arange(0,100,1)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=False)

        # First subplot
        im1 = axes[0].imshow(self.bs, cmap=cmap, vmin=0, vmax=9)
        axes[0].set_ylabel('Depth (m)')
        axes[0].set_title('Given boreholes')
        axes[0].set_xlabel('Distance (m)')
        axes[0].set_xlim(-0.5-2, self.bs.shape[1] - 0.5 + 2)

        # Second subplot
        if self.uncertainty_data is None:
            uncertainty = 1 - np.abs(self.prediction_data - 0.5) * 2
            uncertainty_percent = uncertainty * 100
        else:
            uncertainty_percent = self.uncertainty_data

        im2 = axes[1].imshow(uncertainty_percent, cmap='gist_gray', vmin=0, vmax=100)
        axes[1].set_title('% Uncertainty')
        axes[1].set_xlabel('Distance (m)')
        cbar = plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.2, fraction=0.046)
        cbar.set_label('% Error')

        # Third subplot
        im3 = axes[2].imshow(np.round(self.prediction_data), cmap=cmap, vmin=0, vmax=9)
        axes[2].set_title('Reconstructed model')
        axes[2].set_xlabel('Distance (m)')
        unique_values = np.unique(self.bs.ravel()[~np.isnan(self.bs.ravel())])
        cmap_obj = im1.get_cmap()
        norm = im1.norm
        legend_handles = []
        for idx, value in enumerate(unique_values):
            color = cmap_obj(norm(value))
            if int(value) < len(labels):
                label = labels[int(value)]
            else:
                label = f'Unit {value}'
            patch = mpatches.Patch(color=color, label=label)
            legend_handles.append(patch)

        # Place legend below the axis label
        axes[0].legend(
            handles=legend_handles,
            title="Legend",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.55),  # Move further down
            ncol=len(unique_values)
        )

        # Align subplots on the top side
        for ax in axes:
            ax.set_anchor('N')

        plt.subplots_adjust(wspace=0.15)
        rcParams['pdf.fonttype'] = 42  # Ensures fonts are embedded as TrueType
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.show()    

    def plot_model_with_validation(self,real, cmap='viridis', random_indices=None,labels=np.arange(0,100,1),filename='model.pdf'):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        
        
        del1 = self.prediction_data.copy()
        del1[np.isnan(real)] = np.nan

        # First subplot
        im1 = axes[0][0].imshow(self.bs, cmap=cmap, vmin=0, vmax=9)
        axes[0][0].set_ylabel('Depth (cm)')
        axes[0][0].set_title('All boreholes')
        axes[0][0].set_xlim(-0.5-2, self.bs.shape[1] - 0.5 + 2)

        # Second subplot
        if self.uncertainty_data is None:
            uncertainty = 1 - np.abs(del1 - 0.5) * 2
            uncertainty_percent = uncertainty * 100
        else:
            uncertainty_percent = self.uncertainty_data

        im2 = axes[0][1].imshow(uncertainty_percent, cmap='gist_gray', vmin=0, vmax=100)
        axes[0][1].set_title('% Uncertainty')
        axes[0][1].set_xlabel('Distance (m)')
        axes[0][1].set_xlim(-0.5-2, self.bs.shape[1] - 0.5 + 2)
        cbar = fig.colorbar(im2, ax=axes[0][1], orientation='vertical', pad=0.02, fraction=0.046)
        cbar.set_label('% Error')

        # Third subplot
        axes[0][2].imshow(np.round(del1), cmap=cmap, vmin=0, vmax=9)
        axes[0][2].set_title('Reconstructed model')
        unique_values = np.unique(self.bs.ravel()[~np.isnan(self.bs.ravel())])

        cmap_obj = im1.get_cmap()
        norm = im1.norm
        legend_handles = []
        for idx, value in enumerate(unique_values):
            color = cmap_obj(norm(value))
            if int(value) < len(labels):
                label = labels[int(value)]
            else:
                label = f'Unit {value}'
            patch = mpatches.Patch(color=color, label=label)
            legend_handles.append(patch)

        axes[0][0].legend(handles=legend_handles, title="Legend",
                        loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, frameon=True)

        # Fourth subplot
        axes[1][0].imshow(np.round(real), cmap=cmap, vmin=0, vmax=9)
        axes[1][0].set_title('Real model')

        # Validation/confusion matrix
        if random_indices is None:
            random_indices = np.arange(real.shape[1])
        test_data = real[:, random_indices].ravel()
        pred_data = del1[:, random_indices].ravel()
        i_keep = np.where(np.isfinite(test_data))[0]
        confusion_matrix = metrics.confusion_matrix(test_data[i_keep], pred_data[i_keep])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        cm_display.plot(ax=axes[1][1], cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
        axes[1][1].set_title('Confusion Matrix')

        tick_labels = [h.get_label() for h in legend_handles]
        axes[1][1].set_xticks(np.arange(len(tick_labels)))
        axes[1][1].set_yticks(np.arange(len(tick_labels)))
        axes[1][1].set_xticklabels(tick_labels, rotation=45, ha='right')
        axes[1][1].set_yticklabels(tick_labels)

        # Hide unused subplot
        axes[1][2].axis('off')
        diff_binary = (np.round(real) != np.round(del1)).astype(int)
        axes[1][2].imshow(diff_binary, cmap='gray', vmin=0, vmax=1)
        axes[1][2].set_title('Binary Difference Mask')

        #plt.tight_layout()
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.show()


    def plot_model_with_validation_real(self,real, basis, del1, X, Y, tt, del_perc=None, cmap='viridis', random_indices=None,labels=np.arange(0,100,1),filename='model_with_validation.pdf'):

        # Prepare polygon mask for topography
        polygon = tt.T
        points = np.c_[X.ravel(), Y.ravel()]
        poly_path = Path(polygon)
        inside = poly_path.contains_points(points)
        inside_mask = inside.reshape(X.shape)
        del1_plot = del1.copy()
        del1_plot[~inside_mask] = np.nan
        if del_perc is not None:
            del_perc = del_perc.copy()
            del_perc[~inside_mask] = np.nan

        unique_values = np.unique(basis.ravel()[~np.isnan(basis.ravel())])

        # Validation/confusion matrix
        if random_indices is None:
            random_indices = np.arange(real.shape[1])
        test_data = real[:, random_indices].ravel()
        test_data_plot = np.nan * np.ones_like(real)
        pred_data_plot = np.nan * np.ones_like(real)
        test_data_plot[:, random_indices] = real[:, random_indices]
        pred_data_plot[:, random_indices] = del1_plot[:, random_indices]
        pred_data = del1[:, random_indices].ravel()
        i_keep = np.where(np.isfinite(test_data))[0]

        # Set global style
        plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16})

        fig, axes = plt.subplots(4, 1, figsize=(18, 24), constrained_layout=True)

        # Calculate common x-limits (min/max) for all subplots
        x_min = np.nanmin(X)
        x_max = np.nanmax(X)

        # Plot 1: All boreholes and validation
        im1 = axes[0].pcolor(X + 25, Y - 0.125, np.round(real), cmap=cmap, vmin=0, vmax=9, edgecolors='w', linewidths=1.0, shading='auto')
        axes[0].pcolor(X + 25, Y - 0.125, np.round(test_data_plot), cmap=cmap, vmin=0, vmax=9, edgecolors='k', linewidths=1.0, shading='auto')
        axes[0].set_ylabel('Elevation (m)')
        axes[0].set_title('Boreholes used for inpainting - Boreholes used for validation')
        axes[0].set_aspect('auto')
        axes[0].set_xlim(x_min, x_max)
        axes[0].grid(True, linestyle='--', alpha=0.5)
        cmap_obj = im1.get_cmap()
        norm = im1.norm
        legend_handles = []
        for idx, value in enumerate(unique_values):
            color = cmap_obj(norm(value))
            label = labels[int(value)] if int(value) < len(labels) else f'Unit {value}'
            patch = mpatches.Patch(color=color, label=label)
            legend_handles.append(patch)
        axes[0].legend(handles=legend_handles, title="Legend", loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1, frameon=True)

        # Plot 2: Validation vs Prediction
        axes[1].pcolor(X + 50, Y - 0.125, np.round(test_data_plot), cmap=cmap, vmin=0, vmax=9, edgecolors='k', linewidths=1.0, shading='auto')
        axes[1].pcolor(X + 25, Y - 0.125, np.round(pred_data_plot), cmap=cmap, vmin=0, vmax=9, edgecolors='r', linewidths=1.0, shading='auto')
        axes[1].set_ylabel('Elevation (m)')
        axes[1].set_title('Validation boreholes - Predicted boreholes')
        axes[1].set_aspect('auto')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        # Add legend for validation and prediction
        red_patch = mpatches.Patch(edgecolor='r', facecolor='none', linewidth=2, label='Prediction')
        black_patch = mpatches.Patch(edgecolor='k', facecolor='none', linewidth=2, label='Validation')
        axes[1].legend(handles=[red_patch, black_patch], loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True)

        # Plot 3: Reconstructed model
        axes[2].pcolor(X, Y, np.round(del1_plot), cmap=cmap, vmin=0, vmax=9, edgecolors='w', linewidths=0.5, shading='auto')
        axes[2].set_title('Reconstructed model')
        axes[2].pcolor(X + 25, Y - 0.125, np.round(real), cmap=cmap, vmin=0, vmax=9, edgecolors='b', linewidths=0.5, shading='auto')
        axes[2].pcolor(X + 25, Y - 0.125, np.round(pred_data_plot), cmap=cmap, vmin=0, vmax=9, edgecolors='r', linewidths=1.0, shading='auto')

        axes[2].set_ylabel('Elevation (m)')
        axes[2].set_aspect('auto')
        axes[2].grid(True, linestyle='--', alpha=0.5)
        # Add legend for validation and prediction
        red_patch = mpatches.Patch(edgecolor='b', facecolor='none', linewidth=2, label='Data Used')
        black_patch = mpatches.Patch(edgecolor='r', facecolor='none', linewidth=2, label='Prediction')
        axes[2].legend(handles=[red_patch, black_patch], loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True)


        # Plot 4: Uncertainty
        im2 = axes[3].pcolor(X, Y, del_perc, cmap='gist_gray', vmin=0, vmax=100, shading='auto')
        axes[3].set_title('% Uncertainty')
        axes[3].set_ylabel('Elevation (m)')
        axes[3].set_xlabel('Distance (m)')
        axes[3].set_aspect('auto')
        axes[3].grid(True, linestyle='--', alpha=0.5)
        cbar = fig.colorbar(im2, ax=axes[3], orientation='vertical', pad=0.02, fraction=0.04)
        cbar.set_label('% Error')
        cbar.ax.tick_params(labelsize=12)
        rcParams['pdf.fonttype'] = 42  # Ensures fonts are embedded as TrueType
        fig.savefig(filename, format="pdf", bbox_inches="tight")
        # Confusion Matrix (only classes present in true or predicted labels)
        present_true = np.unique(test_data[i_keep])
        present_pred = np.unique(pred_data[i_keep])
        present_classes = np.unique(np.concatenate([present_true, present_pred]))
        # Remove nan and zero if not present
        present_classes = present_classes[np.isfinite(present_classes)]
        # Only keep classes with at least one true or predicted sample
        mask_present = np.isin(test_data[i_keep], present_classes) | np.isin(pred_data[i_keep], present_classes)
        # Get corresponding labels
        present_labels = [labels[int(c)] if int(c) < len(labels) else f'Unit {c}' for c in present_classes]
        fig_cm, ax_cm = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
        confusion_matrix = metrics.confusion_matrix(
            test_data[i_keep][mask_present], 
            pred_data[i_keep][mask_present], 
            labels=present_classes
        )
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        cm_display.plot(ax=ax_cm, cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
        ax_cm.set_title('Confusion Matrix (Present Classes)', fontsize=18)
        ax_cm.set_xticks(np.arange(len(present_labels)))
        ax_cm.set_yticks(np.arange(len(present_labels)))
        ax_cm.set_xticklabels(present_labels, rotation=45, ha='right', fontsize=14)
        ax_cm.set_yticklabels(present_labels, fontsize=14)
        ax_cm.grid(False)
        fig_cm.savefig(filename[:-4]+'_validation.pdf', format="pdf", bbox_inches="tight")
        plt.show()
