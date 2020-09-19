


import feature_calc
import numpy as np
import Evolution
from NNets import NNets

class Neural_Bot(object):
    
    def __init__(self):
        self.game = None;
        self.MAX_ITERATIONS = 80
        self.POP_SIZE = 20
        self.INITIAL = 1
        self.hidden_layer1 = 5
        self.hidden_layer2 = 2
        self.inputs = 11
        self.gene_number = self.hidden_layer1*self.hidden_layer2+self.hidden_layer1*self.inputs + self.hidden_layer2 + self.hidden_layer1 + self.hidden_layer2 + 1
        self.pop,self.labels = Evolution.create_initial_population(self.POP_SIZE*self.INITIAL,self.gene_number)
#        self.pop[:] = [[-3.57186],[0.34468494],[-0.44975404],[0.37229567],[-0.63908 ],
#                 [-1.88027692],[-0.74647466],[ 1.25289468],[-1.1205533],[-0.00261427],[0]]
        print(self.pop.shape)
#        self.pop[0,:] = -8
#        self.pop[3,:] = 8
        self.nnets = NNets(self.inputs,self.hidden_layer1,self.hidden_layer2)
        self.scores = []
        print('Bot Created')
        self.gen_scores = []
        self.gen_scores_save = []
        self.all_scores=[]
        self.pop_index = 0
        self.iteration_index = 0
        self.weight = self.pop[:,self.pop_index]
        [first,second,out,bias1,bias2,bias3] = self.seperate_weights(self.weight)                
        self.nnets.set_weights_bias(first,second,out,bias1,bias2,bias3)
#        print(self.weight)
        self.new_gen = False
                    
    def best_move(self,tetr):
        position,shape = feature_calc.eval_possible_states_neural(self.matrix,tetr,self.weight,self.nnets)
        #position = int(np.argmax(scores)/4)
        #shape = np.argmax(scores)%4
        if (tetr.name) == 'long':
            position = position
        return (position,shape)
    
    def move_tetr(self,tetr,position,shape):
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

    def next_agent(self):
        print(self.scores)
        if self.iteration_index == 0:
            if not self.new_gen:
                self.gen_scores = np.append(self.gen_scores,np.mean(self.scores))
                self.gen_scores_save = np.append(self.gen_scores_save,self.scores)
            self.scores = []
            if self.pop_index < self.POP_SIZE*self.INITIAL-1:
                self.pop_index += 1
                self.weight = self.pop[:,self.pop_index]
                [first,second,out,bias1,bias2,bias3] = self.seperate_weights(self.weight)
                self.nnets.set_weights_bias(first,second,out,bias1,bias2,bias3)
                #print(self.weight)
                self.new_gen = False
                #self.weight = np.load('One_best.npy')
                #self.weight = np.load('best_gene.npy')
            else:
                self.next_gen()
        else:
            if not self.new_gen:
                self.gen_scores = np.append(self.gen_scores,np.mean(self.scores))
                self.gen_scores_save = np.append(self.gen_scores_save,self.scores)
            self.scores = []
            if self.pop_index < self.POP_SIZE-1:
                self.pop_index += 1
                self.weight = self.pop[:,self.pop_index]
                self.weight = self.pop[:,self.pop_index]
                [first,second,out,bias1,bias2,bias3] = self.seperate_weights(self.weight)
                self.nnets.set_weights_bias(first,second,out,bias1,bias2,bias3)
#                print(self.weight)
                self.new_gen = False
#                print(self.weight)
                #self.weight = np.load('best_gene_new1.npy')
                #self.weight = np.load('One_best.npy')
            else:
                self.next_gen()
    
    def next_gen(self):
        name_score = 'scores_of_gen' + str(self.iteration_index)
        np.save(name_score,self.gen_scores_save)        
        self.new_gen = True
        if self.iteration_index == 0:
            print('New generation : ', self.iteration_index, ' . Best of previous gen : ',np.max(self.gen_scores), 
                  ' .Average of generation : ', np.mean(self.gen_scores))
            self.iteration_index += 1
            self.pop,self.labels = Evolution.create_next_gen_hard(self.gen_scores,self.pop,1/self.INITIAL,self.labels)
#            self.pop,self.labels = Evolution.create_next_gen(self.gen_scores,self.pop,self.labels)
            self.pop_index = -1
            #self.all_scores = np.append(self.all_scores,self.gen_scores,0)
            self.gen_scores = []
            self.gen_scores_save = []
            self.next_agent()
        
        elif self.iteration_index < self.MAX_ITERATIONS:
            print('New generation : ', self.iteration_index, ' . Best of previous gen : ',np.max(self.gen_scores), 
                  ' .Average of generation : ', np.mean(self.gen_scores))
            self.iteration_index += 1
            self.pop,self.labels = Evolution.create_next_gen(self.gen_scores,self.pop,self.labels)
            self.pop_index = -1
            self.all_scores = np.append(self.all_scores,self.gen_scores,0)
            self.gen_scores = []
            self.gen_scores_save = [] 
            if self.iteration_index % 5 == 0:
                name = 'scores' + str(self.iteration_index)
                np.save(name,self.all_scores)
                name_genes = 'genes' + str(self.iteration_index)
                np.save(name_genes,self.pop)
                name_score = 'scores_for_gen' + str(self.iteration_index)
                np.save(name_score,self.gen_scores)
            self.next_agent()

        else:
            np.save('scores',self.all_scores)
            while True:
                i = 1
        
    def save_score(self,score):
        self.scores = np.append(self.scores,score)
    def seperate_weights(self,weights):
        a = self.hidden_layer1*self.inputs
        b = a+self.hidden_layer1*self.hidden_layer2
        c = b + self.hidden_layer2*1
        inputs_to_first = weights[0:a]
        first_to_second = weights[a:b]
        second_to_out = weights[b:c]
        bias1 = weights[c:c+self.hidden_layer1]
        bias2 = weights[c+self.hidden_layer1:c+self.hidden_layer1+self.hidden_layer2]
        bias3 = weights[c+self.hidden_layer1+self.hidden_layer2]
        return inputs_to_first,first_to_second,second_to_out,bias1,bias2,bias3
    
    def set_matrix(self,matrix):
        self.matrix = matrix
