# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:31:16 2016

@author: Alex
"""

import math
import random
import numpy as np


class nn:
    def __init__(self,NI,NH,NO): #number of input, hidden and output nodes
        self.ni=NI
        self.nh=NH
        self.no=NO
    
        #initialise node activations
        self.ai,self.ah,self.ao=[],[],[]
        self.ai=[1.0]*self.ni
        self.ah=[1.0]*self.nh
        self.ao=[1.0]*self.no
    
        #create matrices of node weights
        self.wi=makeMatrix(self.ni,self.nh)  #links between input and hidden nodes
        self.wo=makeMatrix(self.nh,self.no)  #links between hidden and output nodes
        #randomize matrix (may not be necessary but leaving in for now)
        randomizeMatrix(self.wi,-0.2,0.2)
        randomizeMatrix(self.wo,-0.2,0.2)
        #matrix of last change in weights (not entirley sure what this is for)
        self.ci=makeMatrix(self.ni,self.nh)
        self.co=makeMatrix(self.nh,self.no)
        
        
       
    
        
    def feedforward(self,inputs): #the 'inputs' are the list of pixels
        for i in range(self.ni):
            self.ai[i]=inputs[i]  #activation of input nodes
         
        #hidden nodes
        for j in range(self.nh): #go through each hidden node and find input
            Sum=0.0
            for i in range(self.ni):
                Sum += (self.ai[i])*(self.wi[i][j]) #add the activation*weight for each input node to the output node
            self.ah[j]=math.tanh(Sum) # sum of inputs*weights to each hidden node is multiplied by tanh to give response
            
        #output nodes
        for k in range(self.no):
            Sum=0.0
            for j in range(self.nh):
                Sum += (self.ah[j])*(self.wo[j][k]) #take activations from above and multiply by link strengths
            self.ao[k]=math.tanh(Sum)
            
        return self.ao[0]  ##NOT SURE ABOUT THIS
        
        

                
                
    def backpropogate(self, targets, inputs, N): #targets are the example outputs, N is the learning rate
        #calculate errors for outputs
        output_deltas=[0.0]*self.no
        for k in range(self.no): #for all output nodes
            error=targets[k]-self.ao[k] #difference between desired and actual output
            output_deltas[k]=dtanh(self.ao[k])*error #the node will change according to the slope of the dtanh function for the difference between target and actual
    
        #calculate errors for hidden layer outputs
        hidden_deltas=[0.0]*self.nh
        for j in range(self.nh):
            error=0.0
            for k in range(self.no):
                error=error+output_deltas[k]*self.wo[j][k]#difference for each link is weight of the link. Sum for all links
            hidden_deltas[j]=dtanh(self.ah[j])*error #delta for each hidden node
    
        #update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change=output_deltas[k]*self.ah[j] #change weight by slope* current activation*learning rate
                self.wo[j][k]=self.wo[j][k]+N*change
                
        #update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change=hidden_deltas[j]*self.ai[i]
                self.wi[i][j]=self.wi[i][j]+N*change #as above but for input weights
                
    
    
    
def makeMatrix( I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m
  
def randomizeMatrix( matrix, a, b):
    for i in range ( len (matrix) ):
        for j in range ( len (matrix[0]) ):
            matrix[i][j] = random.uniform(a,b) 
    
    
#derivative of tanh    
def dtanh(y):
    return 1-(y*y)
    
    #TO TRAIN - FEEDFORWARD EACH EXAMPLE INPUT AND THEN BACKPROPOGATE IT
    #May do a core dump to save weights matrices