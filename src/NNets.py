# -*- coding: utf-8 -*-
"""

@author: Redion
"""
import numpy as np

class NNets(object):
    def __init__(self,inputs,hidden1,hidden2):
        # inputs_train= random_image_train.flatten()/255  code sample
        self.dimensions=[inputs,hidden1,hidden2,1] #
        numberofLayer=3
        self.model = []
    #   learning_rate=0.1

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # derivative of sigmoid
    #def tanh_derivative():
    def tanhPrime(input_1):
        output= 1.0 - input_1**2 
        return output
    
    
    #def sigmoid_derivative():
        #the input should  be sigmoided  value
    def sigmoidPrime(input_1):
         output=input_1(1-input_1)
         return output
    def set_weights_bias(self,weights1,weights2,weights3,bias1,bias2,bias3):
       #numberOfLayers will be an integer starting from 2
       #nodesNumberForEachLayer will be an 1 by numberOfLayers vector
       #create a list which stores weights and bias
       model={}
       i=1

       model["bias" + str(i)]= bias1
       model["Weights" + str(i)]= weights1
       i=2
       model["bias" + str(i)]= bias2
       model["Weights" + str(i)]= weights2
       i=3
       model["bias" + str(i)]= bias3
       model["Weights" + str(i)]= weights3
       self.model = model
       return True
    
    #the forware propagation method (returns Z_l and A_L)
    def forward_prop(self,nonlinearFunction,a0,noLayers):
        memory={}  #it will store a-s and z-s which are needed for backpropagation
        memory["a0"]=a0
        #retrieve the particular weights
        for i in range(1,noLayers+1):
        #   Y=WX+b

            memory["z"+ str(i)]=np.dot(self.model["Weights" + str(i)].reshape(-1,self.dimensions[i-1]),
                  memory["a" + str(i-1)].reshape(-1,1))+self.model["bias" + str(i)].reshape(-1,1)
            # output=f(Y) (nonlinear function)
            if i == noLayers:
                memory["a"+ str(i)]= memory["z"+ str(i)]
            elif nonlinearFunction == "tanh":
                memory["a"+ str(i)]=np.tanh( memory["z"+ str(i)])
            elif nonlinearFunction == "sigmoid":
                memory["a"+ str(i)]= 1/(1 + np.exp(-memory["z"+ str(i)]))
        self.memory = memory
        return memory["a"+ str(noLayers)]

