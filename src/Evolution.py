

import feature_calc
import numpy as np


def create_initial_population(pop_size,genes):
    initial = np.random.uniform(-2.5,2.5,(genes,pop_size))
    labels = []
    for i in range(pop_size):
        labels = np.append(labels,str(i))
    return (initial,labels)

def create_next_gen(scores,pop,labels):
    print(labels)
    print(scores)
    elitist_select,labels1 = selection(scores,pop,0.4,labels)
    next_gen = elitist_select
    next_labels = labels1
    co_select,co_labels = crossover(scores,pop,0.2,labels)
    next_gen = np.append(next_gen,co_select,axis=1)
    next_labels = np.append(next_labels,co_labels)
    mut,mut_lab = mutation(elitist_select,labels1)
    next_gen = np.append(next_gen,mut,axis=1)
    next_labels = np.append(next_labels,mut_lab)
    return next_gen,next_labels

def create_next_gen_hard(scores,pop,hardness,labels):
    print(labels)
    print(scores)

    elitist_select,labels1 = selection(scores,pop,0.4*hardness,labels)
    next_gen = elitist_select
    next_labels = labels1
    co_select,co_labels = crossover(scores,pop,0.2*hardness,labels)
    next_gen = np.append(next_gen,co_select,axis=1)
    next_labels = np.append(next_labels,co_labels)
    mut,mut_lab = mutation(elitist_select,labels1)
    next_gen = np.append(next_gen,mut,axis=1)
    next_labels = np.append(next_labels,mut_lab)
    return next_gen,next_labels

def mutation(pop,labels):
    ## Partial
    #return pop + np.random.uniform(-0.4,0.4,(pop.shape))
    # Substitute
    a = np.random.uniform(0,1,(pop.shape))
    mut = np.random.uniform(-2.5,2.5,(pop.shape))
    #ind = (a < 0.5)
    ind_pop = (a >= 0.5)
    mutants = pop*ind_pop + mut#*ind
    for i in range(labels.size):  
        labels[i] = labels[i] +  'm'

    return (mutants,labels)

    
def selection(scores,pop,elitism,labels,gene = 11):
    ind = scores.argsort()[-int((np.floor(elitism*len(scores)))):]
    selected = pop[:,ind]
    sel_labels = labels[ind]
    selected = selected.reshape(gene,-1)
    return (selected,sel_labels)

def crossover(scores, pop, co_slct,labels):

    parents,co_labels = selection(scores,pop,co_slct,labels)
    a1 = np.random.uniform(0,1,(parents.shape))
    a2 = 1 - a1
    ## Averaging
#   children = (np.transpose(parents*a1) + np.random.permutation(np.transpose(parents*a2)))
    children = np.transpose(parents*a1) + np.transpose(parents*a2)[::-1]
#    a = np.random.uniform(0,1,(parents.shape))
#    ind = (a < 0.5)
#    ind_c = (a >= 0.5)    
#    children = (np.transpose(parents*ind) + np.random.permutation(np.transpose(parents*ind_c)))
    q = np.transpose(children)
    for i in range(co_labels.size):  
        co_labels[i] = co_labels[i] +  'c'
