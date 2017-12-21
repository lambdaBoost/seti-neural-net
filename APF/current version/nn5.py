# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:31:16 2016

@author: Alex
"""

import math
import random
import numpy as np


class nn5:
    def __init__(self,NI,NH,NH1,NH2,NO): #number of input, hidden and output nodes
        self.ni=NI
        self.nh=NH #hidden layer 0
        self.nh1=NH1 #hidden layer 1
        self.nh2=NH2 #hidden layer 2
        self.no=NO
    
        #initialise node activations
        self.ai,self.ah,self.ao=[],[],[]
        self.ai=[1.0]*self.ni
        self.ah=[1.0]*self.nh
        self.ah1=[1.0]*self.nh1
        self.ah2=[1.0]*self.nh2
        self.ao=[1.0]*self.no
    
        #create matrices of node weights
        self.wi=makeMatrix(self.ni,self.nh)  #links between input and hidden0 nodes
        self.wh=makeMatrix(self.nh,self.nh1) #links between hidden0 and hidden 1
        self.wh1=makeMatrix(self.nh1,self.nh2) #links between hidden1 and hidden2
        self.wo=makeMatrix(self.nh2,self.no)  #links between hidden2 and output nodes
        #randomize matrix (may not be necessary but leaving in for now)
        randomizeMatrix(self.wi,-0.2,0.2)  #COMMENTED OUT TO START MATRIX WITH CONSTANT VALUE THROUGHOUT
        randomizeMatrix(self.wh,-0.2,0.2)
        randomizeMatrix(self.wh1,-0.2,0.2)
        randomizeMatrix(self.wo,-0.2,0.2)
        
        #matrix of last change in weights (not entirley sure what this is for)
        self.ci=makeMatrix(self.ni,self.nh)
        self.ch=makeMatrix(self.nh,self.nh1)
        self.ch1=makeMatrix(self.nh1,self.nh2)
        self.co=makeMatrix(self.nh2,self.no)
        
        
       
    
        
    def feedforward(self,inputs): #the 'inputs' are the list of pixels
        for i in range(self.ni):
            self.ai[i]=inputs[i]  #activation of input nodes
         
        #hidden nodes - layer 0
        for j in range(self.nh): #go through each hidden node and find input
            Sum=0.0
            for i in range(self.ni):
                Sum += (self.ai[i])*(self.wi[i][j]) #add the activation*weight for each input node to the output node
            self.ah[j]=math.tanh(Sum) # sum of inputs*weights to each hidden node is multiplied by tanh to give response

        #hidden nodes - layer 1
        for j in range(self.nh1): #go through each hidden node and find input
            Sum=0.0
            for i in range(self.nh):
                Sum += (self.ah[i])*(self.wh[i][j]) #add the activation*weight for each input node to the output node
            self.ah1[j]=math.tanh(Sum) # sum of inputs*weights to each hidden node is multiplied by tanh to give response

        #hidden nodes - layer 2
        for j in range(self.nh2): #go through each hidden node and find input
            Sum=0.0
            for i in range(self.nh1):
                Sum += (self.ah1[i])*(self.wh1[i][j]) #add the activation*weight for each input node to the output node
            self.ah2[j]=math.tanh(Sum) # sum of inputs*weights to each hidden node is multiplied by tanh to give response


            
        #output nodes
        for k in range(self.no):
            Sum=0.0
            for j in range(self.nh2):
                Sum += (self.ah2[j])*(self.wo[j][k]) #take activations from above and multiply by link strengths
            self.ao[k]=math.tanh(Sum)
            
        return self.ao  ##NOT SURE ABOUT THIS
        
        

                
                
    def backpropogate(self, targets, inputs, N): #targets are the example outputs, N is the learning rate
        #calculate errors for outputs
        output_deltas=[0.0]*self.no
        for k in range(self.no): #for all output nodes
            error=targets[k]-self.ao[k] #difference between desired and actual output
            output_deltas[k]=dtanh(self.ao[k])*error #the node will change according to the slope of the dtanh function for the difference between target and actual
    
        #calculate errors for hidden layer outputs - hidden layer 2
        hidden_deltas2=[0.0]*self.nh2
        for j in range(self.nh2):
            error=0.0
            for k in range(self.no):
                error=error+output_deltas[k]*self.wo[j][k]#difference for each link is weight of the link. Sum for all links
            hidden_deltas2[j]=dtanh(self.ah2[j])*error #delta for each hidden node

        #calculate errors for hidden layer outputs - hidden layer 1
        hidden_deltas1=[0.0]*self.nh1
        for j in range(self.nh1):
            error=0.0
            for k in range(self.nh2):
                error=error+hidden_deltas2[k]*self.wh1[j][k]#difference for each link is weight of the link. Sum for all links
            hidden_deltas1[j]=dtanh(self.ah1[j])*error #delta for each hidden node

        #calculate errors for hidden layer outputs - hidden layer 0
        hidden_deltas=[0.0]*self.nh
        for j in range(self.nh):
            error=0.0
            for k in range(self.nh1):
                error=error+hidden_deltas1[k]*self.wh[j][k]#difference for each link is weight of the link. Sum for all links
            hidden_deltas[j]=dtanh(self.ah[j])*error #delta for each hidden node

    
        #update output weights
        for j in range(self.nh2):
            for k in range(self.no):
                change=output_deltas[k]*self.ah2[j] #change weight by slope* current activation*learning rate
                self.wo[j][k]=self.wo[j][k]+N*change

        #update hidden1 weights
        for j in range(self.nh1):
            for k in range(self.nh2):
                change=hidden_deltas2[k]*self.ah1[j] #change weight by slope* current activation*learning rate
                self.wh1[j][k]=self.wh1[j][k]+N*change

        #update hidden0 weights
        for j in range(self.nh):
            for k in range(self.nh1):
                change=hidden_deltas1[k]*self.ah[j] #change weight by slope* current activation*learning rate
                self.wh[j][k]=self.wh[j][k]+N*change
        
        #update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change=hidden_deltas[j]*self.ai[i]
                self.wi[i][j]=self.wi[i][j]+N*change #as above but for input weights
                
    
    
    
def makeMatrix( I, J, fill=-0.2):#CHANGED THIS TO INITIATE MATRIX AT -0.2 THROUGHOUT
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
