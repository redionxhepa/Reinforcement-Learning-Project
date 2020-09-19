

import feature_calc
import numpy as np
import Evolution
class Bot(object):

    def __init__(self):
        self.game = None;
        self.MAX_ITERATIONS = 80
        self.POP_SIZE = 10
        self.INITIAL = 1
        self.gene_number = 11
        self.pop,self.labels = Evolution.create_initial_population(self.POP_SIZE*self.INITIAL,self.gene_number)
#        self.pop[:] = [[-3.57186],[0.34468494],[-0.44975404],[0.37229567],[-0.63908 ],
#                 [-1.88027692],[-0.74647466],[ 1.25289468],[-1.1205533],[-0.00261427],[0]]
#        self.pop[:] = [[-0.53854854], [-1.60903557],  [1.56770598], [-2.28885673],  [0.83760923],  [0.53685226],
# [-1.75321933], [1.45448524], [ 0.14393141],[ -0.44144691], [ 0.82662996]]
        self.pop[:] =[[-0.7729166],[-1.66007687],[1.43303584],[-3.48686149],[0.63306077],[0.09769052],
         [-2.11997579],[0.80672387],[-0.06324216],[-0.64053525],[0.22088197]]
        self.pop[:] =[[-0.84876553],[ -0.95773762],[1.4080699],[-2.32832832],[1.30429547],[-1.43197305],
                 [-0.63502829],[1.50546105],[-0.12530199],[-1.70731413],[-1.01129018]]

        print(self.pop.shape)
#        self.pop[0,:] = -8
#        self.pop[3,:] = 8

        self.scores = []
        print('Bot Created')
        self.gen_scores = []
        self.gen_scores_save = []
        self.all_scores=[]
        self.pop_index = 0
        self.iteration_index = 0
        self.weight = self.pop[:,self.pop_index]
#        print(self.weight)
        self.new_gen = False
                    
    def best_move(self,tetr):
        position,shape = feature_calc.eval_possible_states(self.matrix,tetr,self.weight)
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
                print(self.weight)
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
                print(self.weight)
                self.new_gen = False
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
        
    def set_matrix(self,matrix):
        self.matrix = matrix
