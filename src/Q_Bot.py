#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 22:54:49 2018

@author: redion
"""
import numpy as np
import random
import bot_functions
import feature_calc

class Q_Bot(object):
    
    def __init__(self):
        self.alpha_par=0.1
        self.gamma_par=0.05
        self.eps_par=0.3
        self.iteration =  5
        self.Q_matrix= self.initializeQ_matrix()
        self.matrix = None
        self.old_matrix = None
        self.falling_tetromino = None
        self.last_played_state = 0
        self.last_played_action = 0
        self.score = 0
        self.oldScore = 0
        self.limits = np.array([[0,5],[2,7],[5,10]])
        self.scores = []
    def initializeQ_matrix(self):
        Q_Matrix = np.zeros((9072,20))
        return Q_Matrix
        
    
    def calcReward(self,table):
        old_score = self.calcScore(self.oldScore,self.old_matrix,table)
        new_score = self.calcScore(self.score,self.matrix,table)
        return  4*(self.score-self.oldScore-20) + new_score - old_score - 10
    
    def calcScore(self,score,matrix,table):
        sc = score + feature_calc.partialLine(matrix,self.limits[table][0],self.limits[table][1])-2*feature_calc.average_height(matrix)
        return sc
        
    def setScore(self,score,old):
        if old:
            self.oldScore = score
        else:
            self.score = score        
        
       
    def set_matrix(self,gameMatrix,old):
        gameMatrix = feature_calc.convert2NumpyMatrix(gameMatrix)
        if old:
            self.old_matrix = gameMatrix
        else:
            self.matrix = gameMatrix
        
        
    def findMapping(self,inputvector_size_4):
        for i in range(4):
           if (inputvector_size_4[0,i]==0):
               inputvector_size_4[0,i]=0
           elif(inputvector_size_4[0,i]==1):
               inputvector_size_4[0,i]=1
           elif(inputvector_size_4[0,i]==-1 ):
             inputvector_size_4[0,i]=2
           elif(inputvector_size_4[0,i]==2):
             inputvector_size_4[0,i]=3
           elif(inputvector_size_4[0,i]==-2):
             inputvector_size_4[0,i]=4
           else :
               inputvector_size_4[0,i]=5
            
        return inputvector_size_4
               
    def map_tetromino(self,tetromino):
        tetr_map = 0
        if tetromino.name == 'long':
            tetr_map = 0
        elif tetromino.name == 'square':
            tetr_map = 1
        elif tetromino.name == 'hat':
            tetr_map = 2           
        elif tetromino.name == 'left_snake':
            tetr_map = 3
        elif tetromino.name == 'right_snake':
            tetr_map = 4
        elif tetromino.name == 'left_gun':
            tetr_map = 5
        elif tetromino.name == 'right_gun':
            tetr_map = 6
        return tetr_map

            
    def findtheState(self,stateMatrix,tetromino_unmapped):
        #stateMatrix = feature_calc.convert2NumpyMatrix(stateMatrix)
        dimensions=stateMatrix.shape
        skyline=np.zeros([1,10])
        tetromino = self.map_tetromino(tetromino_unmapped)
        for i in range(dimensions[1]):
           if(np.sum(stateMatrix[:,i])==0):
               skyline[0,i]=0
           else :
                column=stateMatrix[:,i]
                index=column.argmax(axis=0)
                skyline[0,i]=22-index
         #find the difference
        difference_1=np.ediff1d(skyline[0,0:5]).reshape(1,4)
        difference_2=np.ediff1d(skyline[0,2:7]).reshape(1,4)
        difference_3=np.ediff1d(skyline[0,5:10]).reshape(1,4)
        #create the mapping
        difference_1=self.findMapping(difference_1)
        difference_2=self.findMapping(difference_2)
        difference_3=self.findMapping(difference_3)
        stateNumber1=difference_1[0,0]+difference_1[0,1]*6+difference_1[0,2]*36+difference_1[0,3]*216+tetromino*1296
        stateNumber2=difference_2[0,0]+difference_2[0,1]*6+difference_2[0,2]*36+difference_2[0,3]*216+tetromino*1296
        stateNumber3=difference_3[0,0]+difference_3[0,1]*6+difference_3[0,2]*36+difference_3[0,3]*216+tetromino*1296


        return int(stateNumber1),int(stateNumber2),int(stateNumber3)
       
        
    def calculateImmediateReward():
        disp("Calculate immediate Reward")
        
    def isFeasible(self,rotation,column,tetr):
#        print('Feasible')
#        print(tetr.name)
#        print(rotation)
        feas = column<=5-bot_functions.dict_tetr_len[tetr.name][rotation]
        return feas,(rotation%bot_functions.dict_tetr_shape[tetr.name])   
        
    
    def move_tetr(self,tetr,shape,position,table):
        moves = []
        #print(position)
        for j in range(shape):
            moves = np.append(moves,'UP')
        
        
        if tetr.name == 'long' and shape%2 == 1:
            position = position - 2
        elif tetr.name == 'left_gun' and shape%4 == 1:
            position = position -1
        elif tetr.name == 'hat' and shape%4 == 1:
            position = position -1
        elif tetr.name == 'left_snake' and shape%2 == 1:
            position = position -1
        elif tetr.name == 'right_snake' and shape%2 == 1:
            position = position -1
        elif tetr.name == 'right_gun' and shape%4 == 1:
            position = position -1
        
        if table == 0:
            position = position
        elif table == 1:
            position = position+2
        else:
            position = position + 5

        if tetr.name == 'square':
            if  position < 4:
                for i in range(4-position):
                    moves = np.append(moves,'LEFT')
            else:
                for i in range(position-4):
                    moves = np.append(moves,'RIGHT')
        else:
            if  position < 4:
                for i in range(3-position):
                    moves = np.append(moves,'LEFT')
            else:
                for i in range(position-3):
                    moves = np.append(moves,'RIGHT')
        #moves = np.append(moves,'SPACE')
        #print('Time to move')
        return moves
    
    def  best_move(self,tetromino):
#         print("Q learning")   
         #hyperparameters
    #     for i in range(10000): #we will check this range
    #     print("Calculate best move")
         #find the state number
         #gameMatrix = feature_calc.convert2NumpyMatrix(self.matrix)
         state_1,state_2,state_3=self.findtheState(self.matrix,tetromino)#change it
         #choose an action (act greedily)
         sampled_number= random.uniform(0,1)
         state_no = 0
         if(sampled_number<self.eps_par):
             ended=False
             while(ended==False):
                 #take a random action 
                 index_action=random.randint(0,19) # take a random action
                 rotation_ToMake = index_action//5
                 column_ToBeSet=index_action%5
                 table = random.randint(0,2)
                 ended,rotation_ToMake = self.isFeasible(rotation_ToMake,column_ToBeSet,tetromino)
         else:
             #take an action
             ended=False
             firstTraverseEnded=False
             while (ended==False):
#                   print(type(self.Q_matrix))
                   max_value=np.max(np.array([self.Q_matrix[state_1,:],self.Q_matrix[state_2,:],
                                              self.Q_matrix[state_3,:]]))
#                   print(max_value)
#                   print(np.nanmax(np.concatenate((self.Q_matrix[state_1,:],
#                                                   self.Q_matrix[state_2,:],
#                                                   self.Q_matrix[state_3,:]))))
##                   print(np.nanmax(self.Q_matrix[state_1,:]))
#                   print(np.nanmax(self.Q_matrix[state_2,:]))
#                   print(np.nanmax(self.Q_matrix[state_3,:]))
                   if (max_value==np.nanmax(self.Q_matrix[state_1,:])):
                       index_action=np.argmax(self.Q_matrix[state_1,:])
                       state_no = state_1
                       table = 0
                   elif (max_value==np.nanmax(self.Q_matrix[state_2,:])):
                       index_action=np.argmax(self.Q_matrix[state_2,:])
                       state_no = state_2
                       table = 1
                   elif  (max_value==np.nanmax(self.Q_matrix[state_3,:])):
                       index_action=np.argmax(self.Q_matrix[state_3,:])
                       state_no = state_3
                       table = 2
                   else:
                        print('Error')
                   rotation_ToMake= index_action//5
                   column_ToBeSet=index_action%5
                   ended,rotation_ToMake = self.isFeasible(rotation_ToMake,column_ToBeSet,tetromino)
                   if not ended:
                       self.Q_matrix[state_no,index_action]  =  -10
         self.last_played_action = index_action
         self.last_played_state = state_no
         return rotation_ToMake,column_ToBeSet,table
         #find the next state by using that action
    
    def update_Q_matrix(self,table,tetromino,reward_next):
             alpha_par = self.alpha_par
             gamma_par = self.gamma_par
             next_states=self.findtheState(self.matrix,tetromino) # bu da bizim findstate'le mi yapacaz ?   #change these     
             next_state = next_states[table]
         #   reward_next = self.calculateImmediateReward()
         #do the sample estimate
             sample_estimate=alpha_par * (reward_next + gamma_par * np.max(self.Q_matrix[next_state,:]))
         #update the Q-values
             self.Q_matrix[self.last_played_state,self.last_played_action] = (1-alpha_par)*self.Q_matrix[self.last_played_state,self.last_played_action]+sample_estimate
             self.iteration = self.iteration + 1
             if(self.iteration%50000 == 0):
                  print(np.nonzero(self.Q_matrix))
                  ind = np.flatnonzero(self.Q_matrix)
                  print(self.Q_matrix[ind//20,ind%20])
                  if(self.iteration%100000 == 0):
                      name = 'qmatrix'+ str(self.iteration)+'.txt'
                      np.savetxt(name,self.Q_matrix)
                      name = 'scores' + str(self.iteration)+'.txt'
                      np.savetxt(name,self.scores)
#             print(np.min(self.Q_matrix))

        
    def save_score(self,score):
        self.scores = np.append(self.scores,score)
