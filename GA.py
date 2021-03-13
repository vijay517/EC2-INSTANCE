# -*- coding: utf-8 -*-
"""
@author: Shahruj Rashid
"""
from random import random
from random import uniform
import math
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import random
from matplotlib import cm
import time
import sys
import json
import ast


#iterat = int(sys.argv[1])
#part = int(sys.argv[2])

iterat = 100
part = 30
#model2 = pickle.load(open("model.dat", "rb"))
#model2 = keras.models.load_model('my_model_2')
#model = pickle.load(open("model.dat", "rb"))


# function returns the output of the model when passed in a particle's parameters
# add models here

def Invoke_Model(data):
    ''' The function  is invoke inference from the endpoint 
        sample data : [[0.23,0.45,0.34,0.35,0.24,0.69,0.92,0.85],[0.23,0.45,0.34,0.35,0.24,0.69,0.92,0.85]]
    '''
    import boto3
    client = boto3.client('runtime.sagemaker')
    #data = data.tolist()
    response = client.invoke_endpoint(EndpointName='sagemaker-tensorflow-serving-2021-02-21-fypmodel-endpoint',
                                      ContentType='application/json',
                                      Body=json.dumps(data))

    result = json.loads(response['Body'].read().decode())
    return np.asarray(result['predictions']).flatten()


def evaluate_position(temp, online, limit):
    SMout = Invoke_Model(temp)
    SMout = (SMout*(limit[1]-limit[0]))+limit[0]
    return SMout


def fitness(arr):
    ideal = 100
    temp = np.asarray(arr)
    temp = np.absolute(arr-ideal)
    return temp


class Population:
    def __init__(self, n, dim, x, exclude, bound, is_online):
        self.pop = np.zeros((n, dim+2))
        self.bestpop = np.zeros((4, dim+1))
        self.dimension = dim
        self.pop_size = n
        self.x = x
        self.exclude = exclude
        self.bounds = bound[0:8]
        self.limit = bound[8]
        self.is_online = is_online
        for i in range(0, n):
            for j in range(0, self.dimension):
                self.pop[i, j] = random.randint(
                    int(self.bounds[j][0]), int(self.bounds[j][1]))

    def evaluate_gene(self):
        temp = np.divide(np.subtract(self.pop[:, 0:self.dimension], self.bounds[:, 0]), np.subtract(
            self.bounds[:, 1], self.bounds[:, 0]))
        self.pop[:, self.dimension] = evaluate_position(
            temp.tolist(), self.is_online, self.limit).transpose()

    def evaluate_fitness(self):
        self.pop[:, self.dimension+1] = fitness(self.pop[:, self.dimension])

    def selection(self):
        arg = self.pop[:, self.dimension+1].argsort()[0:4]
        self.bestpop = self.pop[arg, :]

    def crossover(self):
        self.pop[0:4, :] = self.bestpop
        for i in range(4, int(4+(self.pop_size-4)*0.5)):
            sel = [0, 1, 2, 3]
            a = sel.pop(random.randint(0, len(sel)-1))
            b = sel.pop(random.randint(0, len(sel)-1))
            self.pop[i, 0:self.dimension //
                     2] = self.bestpop[a, 0:self.dimension//2]
            self.pop[i, self.dimension//2:self.dimension] = self.bestpop[b,
                                                                         self.dimension//2:self.dimension]

    def mutate(self):
        for i in range(int(4+(self.pop_size-4)*0.5), self.dimension):
            sampl = np.random.uniform(low=0, high=1, size=(1, self.dimension))
            a = random.randint(0, 3)
            self.pop[i, 0:self.dimension] = np.add(self.bestpop[a, 0:self.dimension], np.multiply(
                self.bestpop[a, 0:self.dimension], sampl))
            for j in range(0, self.dimension):
                # prevents search space from exceeding the bound (upper limit)
                if self.pop[i, j] >= self.bounds[j, 1]:
                    self.pop[i, j] = self.bounds[j, 1]
                # prevents search space from exeeding the bound (lower limit)
                if self.pop[i, j] <= self.bounds[j, 0]:
                    self.pop[i, j] = self.bounds[j, 0]

    def refresh(self):
        for i in self.exclude:
            self.pop[:, i] = self.x[i]

    def print_generation(self):
        print(self.pop)


np.set_printoptions(suppress=True)


def GA(bounds, n_particles, max_gen, dimension, exclude, x0, is_online):
    print(exclude)
    t = time.time()
    pop = Population(n_particles, dimension, x0, exclude, bounds, is_online)
    generation = 0
    hist = []
    while(generation <= max_gen):
        print("generation: "+str(generation)+":"+str(pop.pop[0, -2]))
        pop.refresh()
        pop.evaluate_gene()
        pop.evaluate_fitness()
        pop.selection()
        pop.crossover()
        pop.mutate()
        #print(pop.pop)
        hist.append(pop.pop[0, dimension])
        generation += 1
    print(time.time()-t)
    print("optimisation done")
    # plt.plot(hist)
    #input1 = ['cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age','compressive strength']
    # for i in range(len(input1)):
    #    print(input1[i]+": "+str(pop.pop[0,i]))
    pop.pop = pd.DataFrame(pop.pop)
    pop.pop.iloc[:, 9] = pop.pop.iloc[:, 8]
    # print(np.array(x0).reshape(1,dimension))
    # print(type(np.array(x0).reshape(1,dimension)))
    for i in range(0, len(x0)):
        x0[i] = int(x0[i])
    temp = np.divide(np.subtract(np.array(x0), bounds[0:8, 0]), np.subtract(
        bounds[0:8, 1], bounds[0:8, 0]))
    #temp = Invoke_Model(temp.tolist())
    temp = (Invoke_Model(temp.tolist())[
            0]*(bounds[8, 1]-bounds[8, 0]))+bounds[8, 0]
    pop.pop.iloc[:, 8] = temp
    pop.pop['10'] = np.zeros(pop.pop.shape[0])
    pop.pop['11'] = np.zeros(pop.pop.shape[0])
    pop.pop['12'] = np.zeros(pop.pop.shape[0])
    return pop.pop

# bounds = np.array([[102,540],[0,359],[0,200],[121,247],[0,32],[801,1145],[594,992],[1,365]])
# x0 =[527, 243, 101, 166, 29, 948, 870, 28]
# exclude =[7]
# GA(bounds,part,iterat,8,exclude,x0)
