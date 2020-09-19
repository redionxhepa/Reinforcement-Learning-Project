

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:27:22 2018

@author: Alperen and Redion
"""

import numpy as np
import bot_functions


def convert2NumpyMatrix(dictionarymatrix):
    allocate_array=[]
    for key in dictionarymatrix:
        
        if (dictionarymatrix[key] is None ):
          allocate_array=np.append(allocate_array,0)
        else :
          allocate_array=np.append(allocate_array,1)
    allocate_array=np.reshape(allocate_array,(22,10))     
    return allocate_array

def eval_possible_states(matrix,tetromino,weight):
    #weight = weight[:-1]
    matrix_array = convert2NumpyMatrix(matrix)
#    matrix_array = matrix
    shape_tetr = bot_functions.dict_tetr_shape[tetromino.name]
    #length_tetr = bot_functions.dict_tetr_len[tetromino.name]
    score = 0
    best = -20000000000
    move = [0,0]
    for j in range(shape_tetr):
        length_tetr = bot_functions.dict_tetr_len[tetromino.name][j]
        for i in range(matrix_array.shape[1]-length_tetr+1):
            lines_cleared,matrix,gameover = future_matrix_calc(matrix_array,tetromino,i,j)
            score = np.dot(calc_all_features(matrix,lines_cleared,gameover).reshape(1,-1),weight)
            if(score > best):
                best = score
                move = [i,j]
            #scores = np.append(scores,score)
    return move

def eval_possible_states_neural(matrix,tetromino,weight,nnet):
    #weight = weight[:-1]
    matrix_array = convert2NumpyMatrix(matrix)
#    matrix_array = matrix
    shape_tetr = bot_functions.dict_tetr_shape[tetromino.name]
    #length_tetr = bot_functions.dict_tetr_len[tetromino.name]
    score = 0
    best = -20000000000
    move = [0,0]
    for j in range(shape_tetr):
        length_tetr = bot_functions.dict_tetr_len[tetromino.name][j]
        for i in range(matrix_array.shape[1]-length_tetr+1):
            lines_cleared,matrix,gameover = future_matrix_calc(matrix_array,tetromino,i,j)
            score = nnet.forward_prop('sigmoid',calc_all_features(matrix,lines_cleared,gameover),3)
            if(score > best):
                best = score
                move = [i,j]
            #scores = np.append(scores,score)
    return move


def future_matrix_calc(matrix,tetromino,col,rotation):
    m,gameover = bot_functions.drop_tetr(matrix,tetromino,col,rotation)
    lines_cleared,matrix = bot_functions.sim_board(m)
    return (lines_cleared,matrix,gameover)

def calc_all_features(matrix,lines,gameover):
    features = []
    features = np.append(features,calc_weighted_blocks(matrix)/2)
    features = np.append(features,calc_number_holes(matrix))
    features = np.append(features,calc_number_conn_holes(matrix))
    features = np.append(features,lines)
    features = np.append(features,calc_roughness(matrix))
    features = np.append(features,lines//4)
    features = np.append(features,pitHolePercentCenter(matrix))
    features = np.append(features,clearableLine(matrix))
    features = np.append(features,22 - deepestWell(matrix))
    features = np.append(features,numColHoles(matrix))
    features = np.append(features,int(gameover))
    
    #print(features)
    return features

def partialLine(matrix,col1,col2):
    rows = np.sum(matrix[:,col1:col2],axis = 1)
    lines = (rows == (col2-col1))
    return np.sum(lines)
    

def numColHoles(array_tetris):
    number_rows=array_tetris.shape[0]
    number_columns=array_tetris.shape[1]
    number_colums_with_holes=0
    for i in range(number_columns):
        isColumn=False
        for j in range (number_rows):
            #check if a column starts
            if array_tetris[j,i] ==1 :
                isColumn=True
            if isColumn==True and array_tetris[j,i] ==0:
                number_colums_with_holes=number_colums_with_holes+1
                break
    return number_colums_with_holes

def pitHolePercentCenter(array_tetris):
    number_rows=array_tetris.shape[0]
    number_columns=array_tetris.shape[1]
    numnerOFpits=0
      #find the number of pits
    for i in range(0,number_columns): #traverse each columns
        for j in  range(0,number_rows): #traverse each row
            if(array_tetris[j,i]==1):# otherwise it might create a hole
                break
             #check the borders
            if(i==0 and array_tetris[j,0]==0 and array_tetris[j,1]==1 ):
                numnerOFpits=numnerOFpits+1
                continue
            #check the borders
            if( i== 9 and array_tetris[j,9]==0 and array_tetris[j,8]==1 ):
                numnerOFpits=numnerOFpits+1
                continue
            if(array_tetris[j,i]==0 and array_tetris[j,i-1]==1 and array_tetris[j,i+1]==1 ):
                 numnerOFpits=numnerOFpits+1
    #find the number of holes
    holes = calc_number_holes(array_tetris)
    return numnerOFpits/(numnerOFpits + holes) if not (numnerOFpits + holes) == 0 else 0

def clearableLine(array_tetris):
       sum_rows=np.sum(array_tetris, axis=0)
       indeces=[]
       for i in sum_rows:
           if i==6:
               indeces=np.append(indeces,i)
       #we know the candidate rows, we will select the lowest one
       if(len(indeces) == 0):
           lines=0
       else :
           lines=1 
       return lines
   
def deepestWell(array_tetris):
     #take the shape of the array
    number_rows=array_tetris.shape[0]
    number_columns=array_tetris.shape[1]
    maximum_uptoknow=0
    for i in range(number_columns):
        temporary_max_column=0
        for j in range(number_rows):
            if (array_tetris[j,i]==0):
                temporary_max_column=j
            else:
                break
        #assign the new row
        if(temporary_max_column>maximum_uptoknow):
           maximum_uptoknow=temporary_max_column
    return maximum_uptoknow

def create_weight_vector(low,high,size):
    return np.random.uniform(low,high,size)


def average_height(matrix):
    sum_sf = 0
    for i in range(matrix.shape[1]):
        sum_sf = sum_sf + (22-matrix[:,i].argmax(axis = 0))
    sum_sf = sum_sf / matrix.shape[1]
    return sum_sf
   
def calc_weighted_blocks(matrix):
    sum_sf = 0
    for i in range(matrix.shape[0]):
        sum_sf = sum_sf + (matrix.shape[0]-i)*np.sum(matrix[i,:])
    return sum_sf

def calc_number_holes(matrix):
    count = 0
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            if (matrix[j,i]== 0) and not(np.sum(matrix[:j,i]) == 0):
                count = count + 1
    return count

def calc_roughness(matrix):
    roughness = 0
    skyline= []
    for i in range(matrix.shape[1]-1):
        top = matrix.shape[0]
        j = 0
        while j < matrix.shape[0] and matrix[j,i] == 0:
            j = j + 1
            top = top-1
        np.append(skyline,top)
    for i in range(len(skyline)):
        roughness = roughness + abs(skyline[i] - skyline[i+1])
    return roughness
        
def calc_number_conn_holes(matrix):
    count = 0
    for i in range(matrix.shape[1]):
        if np.equal(matrix[-2:,i],[0,0]).all() and not(np.sum(matrix[:,i]) == 0):
                count = count + 1
        for j in range(1,matrix.shape[0]-1):
            if np.equal(matrix[j-1:j+2,i],[0,0,1]).all() and not(np.sum(matrix[:j,i]) == 0):
                count = count + 1
    return count 
    
mt = np.zeros((22,10))
ind_arr = [(21,4),(21,5),(21,7),(21,8),(21,9),(20,4),(19,4),(18,4)]
ind_arr2 = [(21,3),(21,4),(21,5),(21,7),(21,8),(21,9),(20,4),(19,4),(18,4),
            (18,3),(18,2),(17,3),(16,3),(16,2),(20,8),(19,8),(18,8),(18,7)]

for ind in ind_arr:
    mt[ind] = 1
import tetrominoes
tetromino = tetrominoes.list_of_tetrominoes[0]
mt2 = bot_functions.drop_tetr(mt,tetromino,0,3)
q = calc_weighted_blocks(mt)
holes = calc_number_holes(mt)
conn_holes = calc_number_conn_holes(mt)
length_tetr = bot_functions.dict_tetr_len[tetromino.name][0]
print(length_tetr)
