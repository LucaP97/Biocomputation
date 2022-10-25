from audioop import cross
import random, copy, matplotlib.pyplot as plt

# class to create a node
class individual:
    def __init__ (self):
        self.gene = [0.0, 1.0]*N
        self.fitness = 0

############################################################
# global variables 
############################################################

# genes
N = 10 
# population
P = 50
# generations
G = 5000

population = []
offspring = []

MUTRATE = 0.5

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
    utility=0
    for i in range(N):
        utility = utility + ind.gene[i]
    return utility

def assignIndFitness (population):
    for i in population:
        i.fitness = test_function(i)
    return population

def assignPopFitness (population):
    fitCount = 0
    for i in P:
        fitCount += i.fitness
    return fitCount


# new generation 

def newGeneration(P, population):
    for i in range(0, P):
        parent1 = random.randint(0, P - 1)
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint(0, P - 1)
        off2 = copy.deepcopy(population[parent2])
        if off1.fitness > off2.fitness:
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

def mutation (P, N, population):
    for i in range( 0, P ):
        newind = individual()
        newind.gene = []
        for j in range ( 0, N ):
            gene = population[i].gene[j]
            mutprob = random.random()
            if mutprob < MUTRATE:
                if (gene == 1):
                    gene = 0
                else:
                    gene = 1
            newind.gene.append(gene)
        if test_function(population[i]) < test_function(newind):
            population[i] = copy.deepcopy(newind)
    return population


############################################################
# printing generations 
############################################################

# generation 1 is created sequentially, so we can print that first
# test_function(population)
# print("generation 1: " + printFitness(population))

crossover(P, N, population)
mutation(P, N, population)
assignIndFitness(population)
print("generation 1: " + str(printFitness(population)))


for i in range(G - 1):
    newGeneration(P, population)
    crossover(P, N, population)
    mutation(P, N, population)
    assignFitness(population)
    print("generation " + str(i + 2) + ": " + str(printFitness(population)))

plt.show()
