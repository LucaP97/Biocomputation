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
bestInd = []

MUTRATE = 0.4
MUTSTEP = 1.5

# lists to plot 
popAverage = []
popHighest = []
popLowest = []

# node quantity, inpNodeNum to be overwritten by importData()
inpNodesNum = 0
hidNodesNum = 3
outNodesnum = 1

# node lists
inpNODES = [0 for _ in range(inpNodesNum)]
inpNodeOut = [0 for _ in range(inpNodesNum)]
hidNODES = [0 for _ in range(hidNodesNum)]
hidNodeOut = [0 for _ in range(hidNodesNum)]
outNODES = [0]

# classification data
trainingData = []
validationData = []

# expected output
trainingExpectedOutput = []
validationExpectedOutput = []



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
        trainingSize = math.ceil((len(lines) / 3) * 2)
        validationSize = math.floor(len(lines) / 3)
        for i in range(0, trainingSize):
            currLine = lines[i].split()
            for j in range(len(currLine)):
                currLine[j] = float(currLine[j])
                if currLine[j] == float(0) or currLine[j] == float(1):
                    trainingExpectedOutput.append(int(currLine[j]))
                    currLine.remove(currLine[j])
            trainingData.append(currLine)
        for i in range(trainingSize, len(lines)):
            currLine = lines[i].split()
            for j in range(len(currLine)):
                currLine[j] = float(currLine[j])
                if currLine[j] == float(0) or currLine[j] == float(1):
                    validationExpectedOutput.append(int(currLine[j]))
                    currLine.remove(currLine[j])
            validationData.append(currLine)
    return trainingSize + validationSize, len(currLine), trainingSize, validationSize

# initalises the population
def initalisePopulation():
    for x in range (P):
        temphweight = [[] for y in range(hidNodesNum)]
        # temphweight = [[0 for i in range(inpNodesNum+1)] for j in hidNODES]
        tempoweight = [[] for y in range(outNodesnum)]
        for y in range (hidNodesNum):
            for x in range(inpNodesNum):
                temphweight[y].append( random.uniform(MIN, MAX))
            temphweight[y].append(random.uniform(-1, 1))
        for y in range(outNodesnum):
            for x in range(hidNodesNum):
                tempoweight[y].append(random.uniform(MIN, MAX))
            tempoweight[y].append(random.uniform(-1, 1))
        newind = network()
        newind.hweight = temphweight.copy()
        newind.oweight = tempoweight.copy()
        population.append(newind)
    return population

# fitness
def test_function(ind, data, expectedOutput):
    ind.error = 0
    # for every line in data
    for t in range(len(data)):
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

# generation 50 gets initialised incorrectly
def mutation(population):
    for p in range(0, P):
        newind = network()
        newind.hweight = [[] for y in range(hidNodesNum)]
        newind.oweight = [[] for y in range(outNodesnum)]
        for i in range(len(population[p].hweight)):
            for j in range(len(population[p].hweight[i])):
                hweight = population[p].hweight[i][j]
                mutprob = random.random()
                if mutprob < MUTRATE:
                    alter = random.uniform(-MUTSTEP, MUTSTEP)
                    hweight += alter
                    if hweight > MAX:
                        hweight = MAX
                    if hweight < MIN:
                        hweight = MIN
                newind.hweight[i].append(hweight)
        for i in range(len(population[p].oweight)):
            for j in range(len(population[p].oweight[i])):
                oweight = population[p].oweight[i][j]
                mutprob = random.random()
                if mutprob < MUTRATE:
                    alter = random.uniform(-MUTSTEP, MUTSTEP)
                    oweight += alter
                    if oweight > MAX:
                        oweight = MAX
                    if oweight < MIN:
                        oweight = MIN
                #newind.oweight.append(oweight)
                newind.oweight[i].append(oweight)
    population[p] = copy.deepcopy(newind)

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
    return ((total / len(population)) / DATASIZE) * 100

def addPopHighest(population):
    curr = population[0].error
    for i in population:
        if i.error > curr:
            curr = i.error
    return (curr / DATASIZE) * 100

def addPopLowest(population):
    curr = population[0].error
    for i in range(0, P):
        if population[i].error < curr:
            curr = population[i].error
    return (curr / DATASIZE) * 100


# add best ind from training to go through validation data
# def addValidationInd(population):
#     curr = population[0].error
#     for i in range(0, P):
#         if population[i].error < curr:
#             curr = population[i].error
#     return (curr / DATASIZE) * 100

# store best individual from training data
def addBestInd(population):
    currError = population[0].error
    currInd = population[0]
    for index, ind in enumerate(population):
        if ind.error < currError:
            currError = ind.error
            currInd = population[index]
    return currInd
            



############################################################
# printing generations 
############################################################

DATASIZE, inpNodesNum, trainingSize, validationSize = importData("data1.txt")


initalisePopulation()
for i in range(P):
    test_function(population[i], trainingData, trainingExpectedOutput)
popLowest.append(population)
popAverage.append(population)
print("generation 1: Average:" + str(addPopAverage(population)) + " Highest:" + str(addPopHighest(population)) + " Lowest:" + str(addPopLowest(population)))
for i in range(G - 1):
    newGeneration()
    mutation(offspring)
    for j in range(P):
        test_function(offspring[j], trainingData, trainingExpectedOutput)
    eliteism()
    for j in range(P):
        test_function(offspring[j], trainingData, trainingExpectedOutput)
    popAverage.append(addPopAverage(offspring))
    popLowest.append(addPopLowest)
    print("generation " + str(i + 2) + ": Average:" + str(addPopAverage(offspring)) + " Highest:" + str(addPopHighest(offspring)) + " Lowest:" + str(addPopLowest(offspring)))
    population = offspring.copy()
    offspring.clear()
print("\nTraining complete\n")
bestInd.append(addBestInd(population))
print(len(bestInd))
print(bestInd[0].error)
# for i in range(G):


# initalisePopulation()
# for i in range(P):
#     test_function(population[i])
# popLowest.append(addPopLowest(population))
# popAverage.append(addPopAverage(population))
# popHighest.append(addPopHighest(population))
# print("generation 1: Average:" + str(addPopAverage(population)) + " Highest:" + str(addPopHighest(population)) + " Lowest:" + str(addPopLowest(population)))
# for i in range(G - 1):
#     newGeneration()
#     mutation(offspring)
#     for j in range(P):
#         test_function(offspring[j])
#     eliteism()
#     for j in range(P):
#         test_function(offspring[j])
#     popAverage.append(addPopAverage(offspring))
#     popHighest.append(addPopHighest(offspring))
#     popLowest.append(addPopLowest(offspring))
#     print("generation " + str(i + 2) + ": Average:" + str(addPopAverage(offspring)) + " Highest:" + str(addPopHighest(offspring)) + " Lowest:" + str(addPopLowest(offspring)))
#     population = offspring.copy()
#     offspring.clear()

# plt.plot(np.array(popAverage))
# plt.plot(np.array(popLowest))
# plt.plot(np.array(popHighest))
# plt.show()
