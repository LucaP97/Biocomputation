import random
import copy
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
for i in population:
    i.fitness = test_function(i)

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

for i in offspring:
    print(i.fitness)