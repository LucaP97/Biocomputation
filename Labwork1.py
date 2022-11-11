from audioop import cross
import random, copy, matplotlib.pyplot as plt
import numpy as np
import math

# class to create a node
class individual:
    def __init__ (self):
        # random.uniform returns a floating point rather than int
        self.gene = random.uniform(MIN, MAX) * N
        #self.gene = [0.0, 1.0]*N
        self.fitness = 0

############################################################
# global variables 
############################################################

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

MUTRATE = 0.2
MUTSTEP = 0.4

# lists to plot 
popAverage = []
popLowest = []
popHighest = []

############################################################
# initalise population
############################################################

for x in range (0, P):
    tempgene=[]
    for y in range (0, N):
        tempgene.append( random.randint(0,1))
    newind = individual()
    newind.gene = tempgene.copy()
    population.append(newind)

############################################################
# functions 
############################################################


# fitness
def test_function( ind ):
    utility = 0.0
    for i in range(N):
        utility += pow(ind.gene[i], 2) - (10 * math.cos( 2 * math.pi * (ind.gene[i])))
    utility += 10*N
    return utility

def assignFitness (population):
    for i in population:
        i.fitness = test_function(i)
    return population

def popFitness (population):
    fitCount = 0
    for i in population:
        fitCount += i.fitness
    return fitCount


# new generation 

def newGeneration(P, population):
    for i in range(0, P):
        parent1 = random.randint(0, P - 1)
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint(0, P - 1)
        off2 = copy.deepcopy(population[parent2])
        print(off1.gene)
        print(off2.gene)
        if test_function(off1) > test_function(off2.fitness):
            population[i] = copy.deepcopy(off1)
        else:
            population[i] = copy.deepcopy(off2)
    return population


# crossover

def crossover (P, G, population):
    toff1 = individual()
    toff2 = individual()
    temp = individual()
    for i in range (0, P, 2):
        toff1 = copy.deepcopy(population[i])
        toff2 = copy.deepcopy(population[i+1])
        temp = copy.deepcopy(population[i])
        crosspoint = random.randint(1, G)
        for j in range (crosspoint, G):
            toff1.gene[j] = toff2.gene[j]
            toff2.gene[j] = temp.gene[j]
        if ((test_function(population[i]) + test_function(population[i+1]))) < (test_function(toff1) + test_function(toff2)):
            population[i] = copy.deepcopy(toff1)
            population[i + 1] = copy.deepcopy(toff2)
    return population


# mutation

# def mutation (P, N, population):
#     for i in range( 0, P ):
#         newind = individual()
#         newind.gene = []
#         for j in range ( 0, N ):
#             gene = population[i].gene[j]
#             mutprob = random.random()
#             if mutprob < MUTRATE:
#                 if (gene == 1):
#                     gene = 0
#                 else:
#                     gene = 1
#             newind.gene.append(gene)
#         if test_function(population[i]) < test_function(newind):
#             population[i] = copy.deepcopy(newind)
#     return population

# new mutation

def mutation (P, N, population):
    for i in range( 0, P ):
        newind = individual()
        newind.gene = []
        for j in range ( 0, N ):
            gene = population[i].gene[j]
            mutprob = random.random()
            if mutprob < MUTRATE:
                alter = random.uniform(-MUTSTEP, MUTSTEP)
                gene += alter
                if gene > MAX:
                    gene = MAX
                if gene < MIN:
                    gene = MIN
            newind.gene.append(gene)
        if test_function(population[i]) < test_function(newind):
            population[i] = copy.deepcopy(newind)
    return population


# tournament selection


def tournamentSelection(population):
    return population


# fill plot lists

def addPopAverage (P, population):
    total = popFitness(population)
    return total / P

def addPopHighest (population):
    curr = population[0].fitness
    for i in population:
        if (i.fitness > curr):
            curr = i.fitness
    return curr

def addPopLowest (population):
    curr = population[0].fitness
    for i in population:
        if (i.fitness < curr):
            curr = i.fitness
    return curr

############################################################
# printing generations 
############################################################

# generation 1 is created sequentially, so we can print that first
# test_function(population)
# print("generation 1: " + printFitness(population))

crossover(P, N, population)
mutation(P, N, population)
assignFitness(population)
popAverage.append(addPopAverage(P, population))
popLowest.append(addPopLowest(population))
popHighest.append(addPopHighest(population))
print("generation 1: " + str(popFitness(population)))


for i in range(G - 1):
    newGeneration(P, population)
    crossover(P, N, population)
    mutation(P, N, population)
    assignFitness(population)
    popAverage.append(addPopAverage(P, population))
    popLowest.append(addPopLowest(population))
    popHighest.append(addPopHighest(population))
    print("generation " + str(i + 2) + ": " + str(popFitness(population)))

# print(popAverage)
# print(popLowest)
# print(popHighest)


plt.plot(np.array(popAverage))
plt.plot(np.array(popLowest))
plt.plot(np.array(popHighest))
plt.show()
