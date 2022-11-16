import math
import LabworkMinimisation

with open('data1.txt') as f:
    lines = f.read()

#print(lines)

class network:
    def __init__(self):
        self.hweight = hidNODES, inpNODES + 1
        self.oweight = outNODES, hidNODES + 1
        self.error = 0

inputNodes = 6
hiddenNodes = 6
inpNODES = []
hidNODES = []
outNODES = []
hidNodesOutput = []
inpNodesOutput = []

DATASIZE = len(lines.split(' '))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# fitness function is used to populate the weights for the network
# once done, input values are passed through hidden layer, then outputs are fed to output layer until output is found
# this output is then compared with associated input

def fitness(ind):
    for t in range(DATASIZE):
        for i in range(hidNODES)


# def sigmoid(node):


# linesCut = lines.split(' ')

# for i in range(0,7)