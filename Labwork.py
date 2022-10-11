import random, copy
#import copy
# class to create a node
class individual:
    def __init__ (self):
        self.gene = [0]*N
        self.fitness = 0

# N is a gene, P is population
N = 10
P = 50

# array that contains the population
population = []
offspring = []

# probability for mutation
MUTRATE = 0.1

# fills the population full of nodes
for x in range (0, P):
    tempgene=[]
    for y in range (0, N):
        tempgene.append( random.randint(0,1))
    newind = individual()
    newind.gene = tempgene.copy()
    population.append(newind)

# fitness function
def test_function( ind ):
    utility=0
    for i in range(N):
        utility = utility + ind.gene[i]
    return utility

# assigning fitness value to nodes in population
def assignFitness(checkPop):
    for i in checkPop:
        i.fitness = test_function(i)
    return checkPop

def printFitness(checkPop):
    count = 0
    for i in checkPop:
        count += i.fitness
        #print(i.fitness)
    print("Sum of fitness: " + str(count))

assignFitness(population)

printFitness(population)

# selects parents and create offspring
for i in range (0, P):
    parent1 = random.randint( 0, P-1 )
    off1 = copy.deepcopy(population[parent1])
    parent2 = random.randint( 0, P-1 )
    off2 = copy.deepcopy(population[parent2])
    # two offspring is created, best of the two is added to array
    if off1.fitness > off2.fitness:
        offspring.append( off1 )
    else:
        offspring.append( off2 )

assignFitness(offspring)
print("offspring fitness:")
printFitness(offspring)


# Crossover (switching the tails)
toff1 = individual()
toff2 = individual()
temp = individual()
# looping through population, incrementing by 2 as we are working in pairs
for i in range( 0, P, 2 ):
    toff1 = copy.deepcopy(offspring[i])
    toff2 = copy.deepcopy(offspring[i+1])
    temp = copy.deepcopy(offspring[i])
    # deciding the crosspoint
    crosspoint = random.randint(1,N)
    # looping through every gene in node to swap
    for j in range (crosspoint, N):
        toff1.gene[j] = toff2.gene[j]
        toff2.gene[j] = temp.gene[j]
    # assinging the offspring to the new values
    offspring[i] = copy.deepcopy(toff1)
    offspring[i+1] = copy.deepcopy(toff2)


# checking fitness after crossover
print("------------ fitness after crossover ------------")
assignFitness(offspring)
printFitness(offspring)


# loop through entire population
for i in range( 0, P ):
    newind = individual()
    newind.gene = []
    # loop through every gene per node
    for j in range( 0, N ):
        gene = offspring[i].gene[j]
        mutprob = random.random()
        # MUTRATE needs to be defined, as mutprob is variable
        if mutprob < MUTRATE:
            if( gene == 1):
                gene = 0
            else:
                gene = 1
        newind.gene.append(gene)
    # overwrite offspring with the new mutated version
    offspring[i] = copy.deepcopy(newind)
#append new individual or overwrite offspring.

print("------------ fitness after mutation ------------")

assignFitness(offspring)
printFitness(offspring)