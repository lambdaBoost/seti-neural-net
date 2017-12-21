# -*- coding: utf-8 -*-
"""
Created on Thu May  5 19:26:01 2016

@author: Alex
"""

#BEFORE RUNNING:
#1. CHANGE LINE 119 TO 2040
#2. RUN THE 'REJECT' SAMPLES THROUGH THE NN TO TRAIN IT. USE THE TRAIN_NN FUNCTION WITH NO INPUT TO DO THIS
#3. RUN A LIST OF SIMULATED POSITIVES THROUGH THE NN TO COMPLETE THE TRAINING. THE TRAIN_NN FUNCTION HANDLES THIS AUTOMATICALLY TOO.
#4. GO AHEAD AND LOAD IN NEW IMAGE. THIS IMAGE WILL BE HDULIST IN LINE 30 - SIMPLY REPLACE THE FILENAME
#5. RUN THE 'RUN' FUNCTION. THE DETECT_THRESHOLD MAY NEED CHANGING EMPIRICALLY. TEST SAMPLES CAN BE TRIALLED BY RUNNING NN.FEEDFORWARD() WITH A SIMULATED DETECTION.
#6. IF A DETECTION OCCURS, DO GET_SAMPLE ON THE RETURNED COORDINATES. THEN DO 'PLOT_PROCESS_SAMPLE' ON THE RESULT. THIS CAN THEN BE PLOTTED USING THE LINE AT THE BOTTOM OF THE MODULE

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from nn import nn
import random
import itertools
import sample_maker

#detection threshold. Will change by trial and error
detect_threshold=0.8

#establish a network using the nn class
APF_nn=nn(400,200,1)

#open the FITS file. Note dimensions are 4608X2080 pixels
hdulist = fits.open('ucb-amp193.fits')


#hdulist.info() #gives info for fits file

scidata=hdulist[0].data  ##extract data segment from primary HDU as numpy array
scidata=np.transpose(np.log(scidata))#orient properly and take logarithm
#scidata.shape ##gets shape of the data
#scidata is a 2 dimensional array of 2080, 4608 element arrays
#each 4608 element array is a line of the original image

#array to callibrate sample array to. Obtained from the get_callibration script
callibration_array=np.array([ 0.0247247 ,  0.01263583,  0.02264937,  0.02295071,  0.02917792,
        0.06842698,  0.1516559 ,  0.32294593,  0.67731768,  1.        ,
        0.95166509,  0.54608702,  0.22979681,  0.09049339,  0.02370086,
        0.01764515,  0.        ,  0.01924952,  0.01238659,  0.00925774])
        
xaxis=[0]*20
for i in range(20):
    xaxis[i]=i
#plt.scatter(xaxis,callibration_array)##plots the sample array



#start coordinates for the sample array
sample_x_start=0
sample_y_start=170


#creates the sample array based on the given coordinates
def get_sample(xstart,ystart):
    sample_array=np.zeros((20,20))#initiate a 20X20 array to sweep across the main image

    xend=xstart+20
    yend=ystart+20
#loop to initiate the sample position for the row. Moves the sample down by 20 and compares it to the callibration array
#output is least_squares_array which gives the sum of least squares for the sample, compared to the callibration array
#pick the lowest value from this array. The position of this value is the starting y coordinate for the next row (relative to the last row+the default interval)

#populate the sample with data from scidata
    for j in range(ystart,yend):
        for i in range(xstart,xend):
            sample_array[j-ystart][i-xstart]=scidata[j][i]
    return sample_array

#function to sweep sample in y direction to find best match
def sample_sweep(sweep,xstart,ystart):

#initialise list of least squares for different y values for sample (move down in increments of 1 pixel for 20 pixels)
    least_squares_array=[0]*sweep

#loop to initiate the sample position for the row. Moves the sample down by 20 and compares it to the callibration array
#output is least_squares_array which gives the sum of least squares for the sample, compared to the callibration array
#pick the lowest value from this array. The position of this value is the starting y coordinate for the next row (relative to the last row+the default interval)
    for l in range(sweep):
#populate the sample with data from scidata
        sample_array=get_sample(xstart,ystart)
 
#sum across rows for sample array 
        rowsum=[0]*20       
        for k in range(20):
            rowsum[k]=sum(sample_array[k])

        rowsum=rowsum-min(rowsum)
        rowsum=rowsum/max(rowsum)

    #list of squared differences between callibration and the sample
        difference=[0]*20
        for n in range(20):
            difference[n]=(rowsum[n]-callibration_array[n])**2
        sum_squares=sum(difference)
    #add sum of squares to the overall list
        least_squares_array[l]=sum_squares
    
    #increase y coordinate by 1
        ystart+=1
        #yend=ystart+20
    
#y index of new sample (add this to the y index of the last sample)
    new_index=least_squares_array.index(min(least_squares_array))
    return new_index

#processes the sample to scale it with brightest pixel =1, dimmest=0
#ie. normalises the array for the NN to operate on
#flattens to a list of pixels
def process_sample(array):     
    processed_array=array-np.amin(array)#(amin is 'array minimum')
    processed_array=processed_array/np.amax(processed_array)
    processed_list=processed_array.flatten()
    return(processed_list)
    
#returns sample in array form for plotting (plot script at bottom of script)
def plot_process_sample(array):     
    processed_array=array-np.amin(array)#(amin is 'array minimum')
    processed_array=processed_array/np.amax(processed_array)
    return(processed_array)


#load in the example image and train using samples from it. 
def train_nn(sample_x_start=0,sample_y_start=170):

    #list of coordinates of samples
    #sample_list=[]

    while(sample_y_start<=2040):#2040 for default image size
    

    
        #find the first sample of the row
        sample_y=sample_y_start+sample_sweep(20,sample_x_start,sample_y_start)#sweep the sample by 20 to find next row
        sample_x=sample_x_start#start the sample x coordinate at the default start value (zero)
        
        #THIS LOOP GOES OVER THE SAMPLE IMAGE AND BACKPROPOGATES THE SAMPLES THROUGH THE NN TO TRAIN 'REJECT' EXAMPLES IN        
        while(sample_x<=4560):#inner while loop to go across row (while less than size of image)
            print(sample_x,sample_y)#prints out coordinates of each sample. (comment this ot by default)
  ####### #will do get_sample, then process_sample for each (sample_x,sample_y) here. Will then apply NN to processed sample
            #sample_list.append([sample_x,sample_y])
            sample_x+=20#increment by 20 - moving sample across.
            sample_y=sample_y-3+sample_sweep(6,sample_x,(sample_y-3))#Sweeps 3 pixels each direction for each step across
            
            #train_input is the input sample at the coordinates returned. It is flattened to a list using the process sample function
            train_input=get_sample(sample_x,sample_y)
            train_input=process_sample(train_input)
            
            APF_nn.feedforward(train_input)#feed the sample through the nn
            APF_nn.backpropogate([0],train_input,0.5)#backpropogate sample. Output target is 0. learning rate assumed to be 0.5
         
        sample_y_start+=20#move sample down by 20 to start new row

   
    ####GENERATE AND BACKPROPOGATE SIMULATED DETECTIONS THROUGH THE NN
    counter=0
    while(counter<=200):#DECIDE NUMBER OF SAMPLES TO FEED THROUGH. WILL BE 4X THIS NUMBER
        #GENERATE SIMULATED SAMPLES
        a=flatten(sample_maker.generate_narrowband_1())
        b=flatten(sample_maker.generate_narrowband_2())
        c=flatten(sample_maker.generate_narrowband_1CR())
        d=flatten(sample_maker.generate_narrowband_2CR())
        
        APF_nn.feedforward(a)#feed the sample through the nn
        APF_nn.backpropogate([1],a,0.5)#backpropogate sample. Output target is 1. learning rate assumed to be 0.5

        APF_nn.feedforward(b)#feed the sample through the nn
        APF_nn.backpropogate([1],b,0.5)#backpropogate sample.
        
        APF_nn.feedforward(c)#feed the sample through the nn
        APF_nn.backpropogate([1],c,0.5)#backpropogate sample.
        
        APF_nn.feedforward(d)#feed the sample through the nn
        APF_nn.backpropogate([1],d,0.5)#backpropogate sample.
        
        counter+=1
        
def run(sample_x_start=0,sample_y_start=170):
    #list of coordinates of samples
    #sample_list=[]

    while(sample_y_start<=2040):#2040 for default image size
    

    
        #find the first sample of the row
        sample_y=sample_y_start+sample_sweep(20,sample_x_start,sample_y_start)#sweep the sample by 20 to find next row
        sample_x=sample_x_start#start the sample x coordinate at the default start value (zero)
        
        #THIS LOOP GOES OVER THE SAMPLE IMAGE AND BACKPROPOGATES THE SAMPLES THROUGH THE NN TO TRAIN 'REJECT' EXAMPLES IN        
        while(sample_x<=4560):#inner while loop to go across row (while less than size of image)
            print(sample_x,sample_y)#prints out coordinates of each sample. (comment this ot by default)
  ####### #will do get_sample, then process_sample for each (sample_x,sample_y) here. Will then apply NN to processed sample
            #sample_list.append([sample_x,sample_y])
            sample_x+=20#increment by 20 - moving sample across.
            sample_y=sample_y-3+sample_sweep(6,sample_x,(sample_y-3))#Sweeps 3 pixels each direction for each step across
            
            #train_input is the input sample at the coordinates returned. It is flattened to a list using the process sample function
            input_sample=get_sample(sample_x,sample_y)
            input_sample=process_sample(input_sample)
            
            outnode=APF_nn.feedforward(input_sample)#feed the sample through the nn
            if(outnode>detect_threshold):
                print(sample_x,sample_y)#print out coordinates of candidate if detection threshold exceeded
         
        sample_y_start+=20#move sample down by 20 to start new row


#plt.imshow(scidata,cmap='Greys_r',interpolation='none')##Plot the whole spectrum