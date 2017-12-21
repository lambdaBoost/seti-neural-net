# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:18:36 2016

@author: Alex
"""

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np




hdulist = fits.open('ucb-amp193.fits')


#hdulist.info() #gives info for fits file

scidata=hdulist[0].data  ##extract data segment from primary HDU as numpy array
scidata=np.transpose(np.log(scidata))#orient properly and take logarithm
#scidata.shape ##gets shape of the data
#scidata is a 2 dimensional array of 2080, 4608 element arrays
#each 4608 element array is a line of the original image




sample_array=np.zeros((20,20))#initiate a 20X20 array to sweep across the main image

#start and end of sample array - these coordinates provide a good example of the main spectrum
ystart=424
yend=444

#start end end of x axis of smaple array
xstart=450
xend=470

#populate the sample with data from scidata
for j in range(ystart,yend):
    for i in range(xstart,xend):
        sample_array[j-ystart][i-xstart]=scidata[j][i]
 
#initiate 1D array to be used as callibration for samples - sum across the array to get a curve       
callibration_array=np.zeros(20)
for j in range(20):
    callibration_array[j]=sum(sample_array[j])
    
#normalise array so brightness is irellevant
callibration_array=callibration_array-min(callibration_array)
callibration_array=callibration_array/max(callibration_array)




#plt.imshow(scidata,cmap='Greys_r',interpolation='none')##Plot the whole spectrum