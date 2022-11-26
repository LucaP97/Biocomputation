import random, copy, matplotlib.pyplot as plt
import numpy as np
import math

# class to create network
class network:
    def __init__(self):
        self.hweight = [[0 for i in range(inpNodesNum+1)] for j in hidNODES]
        self.oweight = [[0 for i in range(hidNodesNum+1)] for j in outNODES]
        self.error = 0

############################################################
# global variables 
############################################################

# population
P = 50
# generations
G = 50
# min gene
MIN = -5.12
# max gene
MAX = 5.12

population = []
offspring = []

MUTRATE = 0.8
MUTSTEP = 0.3

# lists to plot 
popAverage = []
popBest = []
popWorst = []

# node quantity
inpNodesNum = 6
hidNodesNum = 3
outNodesnum = 1

# node lists
inpNODES = [0 for _ in range(inpNodesNum)]
inpNodeOut = [0 for _ in range(inpNodesNum)]
hidNODES = [0 for _ in range(hidNodesNum)]
hidNodeOut = [0 for _ in range(hidNodesNum)]
outNODES = [0]

data = []
expectedOutput = []

############################################################
# functions 
############################################################

# sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# imports the data
def importData(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in lines:
            currLine = i.split()
            for j in range(len(currLine)):
                currLine[j] = float(currLine[j])
                if currLine[j] == float(0) or currLine[j] == float(1):
                    expectedOutput.append(int(currLine[j]))
                    currLine.remove(currLine[j])
            data.append(currLine)
    return len(data)

# *** needs to be moved to final loop ***
# DATASIZE = importData("data1.txt")

# initalises the population
def initalisePopulation():
    for x in range (P):
        temphweight = [[] for y in range(hidNodesNum)]
        tempoweight = [[] for y in range(outNodesnum)]
        for y in range (hidNodesNum):
            for x in range(inpNodesNum):
                temphweight[y].append( random.uniform(MIN, MAX))
        for y in range(outNodesnum):
            for x in range(hidNodesNum):
                tempoweight[y].append(random.uniform(MIN, MAX))
        newind = network()
        newind.hweight = temphweight.copy()
        newind.oweight = tempoweight.copy()
        population.append(newind)
    return population

# fitness
def test_function(ind):
    ind.error = 0
    # for every line in data
    for t in range(DATASIZE):
        # for every hidden node
        for i in range(hidNodesNum):
            # initalise hidden node
            hidNodeOut[i] = 0
            # for every input node
            for j in range(inpNodesNum):
                # calculate hidden node output for every input node
                # ind.hweight[i][j] -> hidden node weight from input node j to hidden node i
                hidNodeOut[i] += (ind.hweight[i][j] * data[t][j])
            # bias added
            hidNodeOut[i] += ind.hweight[i][-1]
            # sigmoid function
            hidNodeOut[i] = sigmoid(hidNodeOut[i])
        for i in range(outNodesnum):
            outNODES[i] = 0
            for j in range(hidNodesNum):
                outNODES[i] += (ind.oweight[i][j] * hidNodeOut[j])
            outNODES[i] += ind.oweight[i][-1]
            outNODES[i] = sigmoid(outNODES[i])
        if expectedOutput[t] == 1.0 and outNODES[0] < 0.5:
            ind.error += 1.0
        if expectedOutput[t] == 0.0 and outNODES[0] >= 0.5:
            ind.error += 1.0

# creates offspring
def newGeneration():
    for i in range(len(population)):
        parent1 = random.randint(0, P - 1)
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint(0, P - 1)
        off2 = copy.deepcopy(population[parent2])
        if off1.error < off2.error:
            offspring.append(off1)
        else:
            offspring.append(off2)

# mutation
def mutation (population):
    for i in range(0, P):
        newind = network()
        newind.hweight = [[0 for i in range(inpNodesNum+1)] for j in hidNODES]
        newind.oweight = [[0 for i in range(hidNodesNum+1)] for j in outNODES]
        for j in range(0, hidNodesNum):
            for x in range(len(hidNODES)):
                hweight = population[i].hweight[j][x]
                mutprob = random.random()
                if mutprob < MUTRATE:
                    alter = random.uniform(-MUTSTEP, MUTSTEP)
                    hweight += alter
                    if hweight > MAX:
                        hweight = MAX
                    if hweight < MIN:
                        hweight = MIN
            newind.hweight.append(hweight)
        for j in range(0, outNodesnum):
            for x in range(len(outNODES)):
                oweight = population[i].oweight[j][x]
                mutprob = random.random()
                if mutprob < MUTRATE:
                    alter = random.uniform(-MUTSTEP, MUTSTEP)
                    oweight += alter
                    if oweight > MAX:
                        oweight = MAX
                    if oweight < MIN:
                        oweight = MIN
            newind.oweight.append(oweight)
        population[i] = copy.deepcopy(newind)
    return population

# eliteism
def eliteism():
    popBest, popBestIndex = population[0].error, 0
    offspringWorst, offspringWorstIndex = offspring[0].error, 0
    for index, ind in enumerate(population):
        if ind.error < popBest:
            popBest = ind.error
            popBestIndex = index
    for index, ind in enumerate(offspring):
        if ind.error > offspringWorst:
            offspringWorst = ind.error
            offspringWorstIndex = index
    offspring[offspringWorstIndex] = copy.deepcopy(population[popBestIndex])

# plot graphs
def addPopAverage(population):
    total = 0
    for i in range(len(population)):
        total += population[i].error
    return total / len(population)

def addPopBest(population):
    curr = population[0].error
    for i in population:
        if i.error > curr:
            curr = i.error
    return curr

def addPopWorst(population):
    curr = population[0].error
    for i in population:
        if i.error < curr:
            curr = i.error
    return curr

############################################################
# printing generations 
############################################################

DATASIZE = importData("data1.txt")

# initalisePopulation()
# for i in range(P):
#     test_function(population[i])
# popAverage.append(addPopAverage(population))
# popBest.append(addPopBest(population))
# popWorst.append(addPopWorst(population))
# print("generation 1: Average:" + str(addPopAverage(population)) + " Best:" + str(addPopBest(population)) + " Worst:" + str(addPopWorst(population)))
# for i in range(G - 1):
#     newGeneration()
#     offspring = mutation(offspring)
#     for j in range(P):
#         test_function(offspring[j])
#     eliteism()
#     for j in range(P):
#         test_function(offspring[j])
#     popAverage.append(addPopAverage(offspring))
#     popBest.append(addPopBest(offspring))
#     popWorst.append(addPopWorst(offspring))
#     print("generation " + str(i + 2) + ": Average:" + str(addPopAverage(offspring)) + " Best:" + str(addPopBest(offspring)) + " Worst:" + str(addPopWorst(offspring)))
#     population = offspring.copy()
#     offspring.clear()

# plt.plot(np.array(popAverage))
# plt.plot(np.array(popWorst))
# plt.plot(np.array(popBest))
# plt.show()

# debugging

initalisePopulation()
# for every member in population
for p in range(P):
    print("member " + str(p+1) + ":\nHidden Nodes:")
    # for every array in hidden node
    print(str(population[p].hweight))
    # for h in range(len(population[p].hweight)):
    #     for i in range(hidNodesNum):
    #         print(population[p].hweight[h][i])
