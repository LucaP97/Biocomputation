import random, copy, matplotlib.pyplot as plt
import numpy as np
import math


# class to create a node
class individual:
    def __init__ (self):
        # random.uniform returns a floating point rather than int
        self.gene = random.uniform(MIN, MAX) * N
        self.fitness = 0

# class to create network
class network:
    def __init__(self):
        self.hweight = [[0 for i in range(inpNodesNum+1)] for j in hidNODES]
        self.oweight = [[0 for i in range(hidNodesNum+1)] for j in outNODES]
        self.error = 0

############################################################
# global variables 
############################################################

# ---- GA ----

# genes
N = 10
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

MUTRATE = 0.05
MUTSTEP = 0.1

# lists to plot 
popAverage = []
popLowest = []
popHighest = []

# ---- NN ----


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

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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


# DATASIZE will be the value of the return of the imported file
DATASIZE = importData("data1.txt")

############################################################
# initalise population
############################################################

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
        #newind = individual()
        newind = network()
        # *** not sure if this should be commented out?
        #newind.gene = tempgene.copy()
        newind.hweight = temphweight.copy()
        newind.oweight = tempoweight.copy()
        population.append(newind)
    return population




# print("here:" + str(len(population)))
# for i in range(len(population)):
#     for j in range(len(population[i].hweight)):
#         print("node: " + str(i) + " hweight: "+ str(j) + str(population[i].hweight[j]))
############################################################
# functions 
############################################################

# ---- GA ----

# fitness
# def test_function( ind ):
#     utility = 0.0
#     for i in range(N):
#         utility += pow(ind.gene[i], 2) - (10 * math.cos( 2 * math.pi * (ind.gene[i])))
#     utility += 10*N
#     return utility

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




# new generation 

# def newGeneration():
#     for i in range(0, P):
#         parent1 = random.randint(0, P - 1)
#         off1 = copy.deepcopy(population[parent1])
#         parent2 = random.randint(0, P - 1)
#         off2 = copy.deepcopy(population[parent2])
#         if off1.fitness < off2.fitness:
#             offspring.append(copy.deepcopy(off1))
#         else:
#             offspring.append(copy.deepcopy(off2))
#     return offspring

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


# crossover

# def crossover (P, G, offspring):
#     toff1 = individual()
#     toff2 = individual()
#     temp = individual()
#     for i in range (0, P, 2):
#         toff1 = copy.deepcopy(offspring[i])
#         toff2 = copy.deepcopy(offspring[i+1])
#         temp = copy.deepcopy(offspring[i])
#         crosspoint = random.randint(1, N)
#         for j in range (crosspoint, N):
#             toff1.gene[j] = toff2.gene[j]
#             toff2.gene[j] = temp.gene[j]
#         #if ((test_function(offspring[i]) + test_function(offspring[i+1]))) > (test_function(toff1) + test_function(toff2)):
#         offspring[i] = copy.deepcopy(toff1)
#         offspring[i + 1] = copy.deepcopy(toff2)
#     return offspring


# mutation

# def mutation (P, N, offspring):
#     for i in range( 0, P ):
#         newind = individual()
#         newind.gene = []
#         for j in range ( 0, N ):
#             gene = offspring[i].gene[j]
#             mutprob = random.random()
#             if mutprob < MUTRATE:
#                 alter = random.uniform(-MUTSTEP, MUTSTEP)
#                 gene += alter
#                 if gene > MAX:
#                     gene = MAX
#                 if gene < MIN:
#                     gene = MIN
#             newind.gene.append(gene)
#         #if test_function(offspring[i]) > test_function(newind):
#         offspring[i] = copy.deepcopy(newind)
#     return offspring

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


# def mutation():
#     for i in range(len(population)):
#         newind = network()
#         newind.hweight = [[]]
#         newind.oweight = [[]]
#         for j 



# tournament selection

# def eliteism(population, offspring):
#     popBest = population[0].fitness
#     offspringWorst = offspring[0].fitness
#     popBestIndex = 0
#     offspringWorstIndex = 0
#     for index, ind in enumerate(population):
#         if ind.fitness < popBest:
#             popBest = ind.fitness
#             popBestIndex = index
#     for index, ind in enumerate(offspring):
#         if ind.fitness > offspringWorst:
#             offspringWorst = ind.fitness
#             offspringWorstIndex = index
#     offspring[offspringWorstIndex] = copy.deepcopy(population[popBestIndex])
#     return offspring

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

# fill plot lists

# def addPopAverage (P, population):
#     total = 0
#     for i in range(len(population)):
#         total += population[i].fitness
#     return total / P

# def addPopHighest (population):
#     curr = population[0].fitness
#     for i in population:
#         if (i.fitness > curr):
#             curr = i.fitness
#     return curr

# def addPopLowest (population):
#     curr = population[0].fitness
#     for i in population:
#         if (i.fitness < curr):
#             curr = i.fitness
#     return curr

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


# ---- NN ----



############################################################
# printing generations 
############################################################

initalisePopulation()
for i in range(len(population)):
    test_function(population[i])
popAverage.append(addPopAverage(population))
popLowest.append(addPopWorst(population))
popHighest.append(addPopBest(population))
for i in range(G - 1):
    newGeneration()
    mutation(offspring)
    for j in range(len(offspring)):
        test_function(offspring[j])
    eliteism()
    for j in range(len(offspring)):
        test_function(offspring[j])
    popAverage.append(addPopAverage(offspring))
    popLowest.append(addPopWorst(offspring))
    popHighest.append(addPopBest(offspring))
    population = offspring.copy()
    offspring.clear()


# initalisePopulation()
# for i in range(len(population)):
#     test_function(population[i])
# popAverage.append(addPopAverage(population))
# popLowest.append(addPopWorst(population))
# popHighest.append(addPopBest(population))
# newGeneration()
# mutation(offspring)
# eliteism()
# for i in range(len(offspring)):
#     test_function(offspring[i])
# popAverage.append(addPopAverage(offspring))
# popLowest.append(addPopWorst(offspring))
# popHighest.append(addPopBest(offspring))
# print("highest: " + str(popHighest))
# print("average: " + str(popAverage))
# print("lowest: " + str(popLowest))
# print(len(population))
# eliteism()
# for i in range(len(offspring)):
#     test_function(offspring[i])
# popAverage.append(addPopAverage(offspring))
# popLowest.append(addPopWorst(offspring))
# popHighest.append(addPopBest(offspring))



# generation 1 is created sequentially, so we can print that first

#assignFitness(population)
# for i in range(len(population)):
#     population[i].fitness = test_function(population[i])
# popAverage.append(addPopAverage(P, population))
# popLowest.append(addPopLowest(population))
# popHighest.append(addPopHighest(population)) 
# print("generation 1: " + str(addPopAverage(P, population)))

# for i in range (G - 1):
#     newGeneration(P, population)
#     crossover(P, N, offspring)
#     mutation(P, N, offspring)
#     for j in range(len(offspring)):
#         offspring[j].fitness = test_function(offspring[j])
#     eliteism(population, offspring)
#     for j in range(len(offspring)):
#         offspring[j].fitness = test_function(offspring[j])
#     print("Generation " + str(i + 2) + ": " + str(addPopAverage(P, offspring)))
#     popAverage.append(addPopAverage(P, offspring))
#     popLowest.append(addPopLowest(offspring))
#     popHighest.append(addPopHighest(offspring))
#     population = offspring.copy()
#     offspring.clear()

plt.plot(np.array(popAverage))
plt.plot(np.array(popLowest))
plt.plot(np.array(popHighest))
plt.show()