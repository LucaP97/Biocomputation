from audioop import cross
import random, copy, matplotlib.pyplot as plt
import numpy as np
import math

# class to create a node
class individual:
    def __init__ (self):
        # random.uniform returns a floating point rather than int
        self.gene = random.uniform(MIN, MAX) * N
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

MUTRATE = 0.1
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
        tempgene.append( random.uniform(MIN, MAX))
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
        if test_function(off1) > test_function(off2):
            offspring.append(copy.deepcopy(off1))
        else:
            offspring.append(copy.deepcopy(off1))
    return offspring


# crossover

def crossover (P, G, offspring):
    toff1 = individual()
    toff2 = individual()
    temp = individual()
    for i in range (0, P, 2):
        toff1 = copy.deepcopy(offspring[i])
        toff2 = copy.deepcopy(offspring[i+1])
        temp = copy.deepcopy(offspring[i])
        crosspoint = random.randint(1, G)
        for j in range (crosspoint, G):
            toff1.gene[j] = toff2.gene[j]
            toff2.gene[j] = temp.gene[j]
        if ((test_function(offspring[i]) + test_function(offspring[i+1]))) < (test_function(toff1) + test_function(toff2)):
            offspring[i] = copy.deepcopy(toff1)
            offspring[i + 1] = copy.deepcopy(toff2)
    return offspring


# mutation

def mutation (P, N, offspring):
    for i in range( 0, P ):
        newind = individual()
        newind.gene = []
        for j in range ( 0, N ):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < MUTRATE:
                alter = random.uniform(-MUTSTEP, MUTSTEP)
                gene += alter
                if gene > MAX:
                    gene = MAX
                if gene < MIN:
                    gene = MIN
            newind.gene.append(gene)
        if test_function(offspring[i]) < test_function(newind):
            offspring[i] = copy.deepcopy(newind)
    return offspring



# tournament selection

def eliteism(population, offspring):
    popBest = population[0].fitness
    offspringWorst = offspring[0].fitness
    popBestIndex = 0
    offspringWorstIndex = 0
    for index, ind in enumerate(population):
        if ind.fitness > popBest:
            popBest = ind.fitness
            popBestIndex = index
    for index, ind in enumerate(offspring):
        if ind.fitness < offspringWorst:
            offspringWorst = ind.fitness
            offspringWorstIndex = index
    offspring[offspringWorstIndex] = copy.deepcopy(population[popBestIndex])
    return offspring


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

assignFitness(population)
popAverage.append(addPopAverage(P, population))
popLowest.append(addPopLowest(population))
popHighest.append(addPopHighest(population)) 
print("generation 1: " + str(popFitness(population) / P))

for i in range (G - 1):
    newGeneration(P, population)
    crossover(P, N, offspring)
    mutation(P, N, offspring)
    assignFitness(offspring)
    eliteism(population, offspring)
    assignFitness(offspring)
    print("Generation " + str(i + 2) + ": " + str(popFitness(offspring) / P))
    popAverage.append(addPopAverage(P, offspring))
    popLowest.append(addPopLowest(offspring))
    popHighest.append(addPopHighest(offspring))
    population = offspring.copy()
    offspring.clear()

plt.plot(np.array(popAverage))
plt.plot(np.array(popLowest))
plt.plot(np.array(popHighest))
plt.show()