import random, copy, matplotlib.pyplot as plt
import numpy as np
import math
import csv
import time

print('running')

def runMain(MUTRATE, MUTSTEP, G, P, hidNodesNum, fileNameTest, csvFileName):
    startTime = time.time()

    

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
    G = 5
    # min gene
    MIN = -5.12
    # max gene
    MAX = 5.12

    population = []
    offspring = []
    bestInd = []

    MUTRATE = 0.1
    MUTSTEP = 0.1

    # lists to plot 
    trainingPopAverage = []
    trainingPopHighest = []
    trainingPopLowest = []
    validationPopAverage = []
    validationPopHighest = []
    validationPopLowest = []
    bestIndPlot = []

    # node quantity, inpNodeNum to be overwritten by importData()
    inpNodesNum = 0
    hidNodesNum = 0
    outNodesnum = 1

    # # node lists
    # inpNODES = [0 for _ in range(inpNodesNum)]
    # inpNodeOut = [0 for _ in range(inpNodesNum)]
    # hidNODES = [0 for _ in range(hidNodesNum)]
    # hidNodeOut = [0 for _ in range(hidNodesNum)]
    # outNODES = [0]

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
        return int(trainingSize + validationSize), len(currLine), trainingSize, validationSize

    # initalises the population
    def initalisePopulation():
        for x in range (P):
            temphweight = [[] for y in range(hidNodesNum)]
            # temphweight = [[0 for i in range(inpNodesNum+1)] for j in hidNODES]
            tempoweight = [[] for y in range(outNodesnum)]
            for y in range (hidNodesNum):
                for x in range(inpNodesNum):
                    temphweight[y].append(random.uniform(MIN, MAX))
                temphweight[y].append(random.uniform(-1, 1)) # better results when i make this 1
            for y in range(outNodesnum):
                for x in range(hidNodesNum):
                    tempoweight[y].append(random.uniform(MIN, MAX))
                tempoweight[y].append(random.uniform(-1, 1)) # better results when i make this 1
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
                hidNodeOut[i] += ind.hweight[i][-1] # could consider making -1 as InpNodesNum
                # sigmoid function
                hidNodeOut[i] = sigmoid(hidNodeOut[i])
            for i in range(outNodesnum):
                outNODES[i] = 0
                for j in range(hidNodesNum):
                    outNODES[i] += (ind.oweight[i][j] * hidNodeOut[j])
                outNODES[i] += ind.oweight[i][-1] # could consider making -1 as hidNodesNum
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
        return total / len(population)

    def addPopHighest(population):
        curr = population[0].error
        for i in population:
            if i.error > curr:
                curr = i.error
        return curr

    def addPopLowest(population):
        curr = population[0].error
        for i in range(0, P):
            if population[i].error < curr:
                curr = population[i].error
        return curr


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

    DATASIZE, inpNodesNum, trainingSize, validationSize = importData(fileNameTest)

    hidNodesNum = 3 #int(math.floor(inpNodesNum / 3))

    # node lists
    inpNODES = [0 for _ in range(inpNodesNum)]
    inpNodeOut = [0 for _ in range(inpNodesNum)]
    hidNODES = [0 for _ in range(hidNodesNum)]
    hidNodeOut = [0 for _ in range(hidNodesNum)]
    outNODES = [0]

    #hidNodesNum = 3 #math.floor(inpNodesNum / 3)

    # training
    initalisePopulation()
    for i in range(P):
        test_function(population[i], trainingData, trainingExpectedOutput)
    trainingPopLowest.append((addPopLowest(population) / trainingSize) * 100)
    trainingPopAverage.append((addPopAverage(population) / trainingSize) * 100)
    trainingPopHighest.append((addPopHighest(population) / trainingSize) * 100)
    bestInd.append(addBestInd(population))
    for j in range(len(bestInd)):
        test_function(bestInd[j], validationData, validationExpectedOutput)
    bestIndPlot.append((bestInd[0].error / validationSize) * 100)
    bestInd.clear()
    print("generation 1: Average:" + str(addPopAverage(population)) + " Lowest:" + str(addPopLowest(population)) + " Highest: " + str(addPopHighest(population)))
    for i in range(G - 1):
        newGeneration()
        mutation(offspring)
        for j in range(P):
            test_function(offspring[j], trainingData, trainingExpectedOutput)
        eliteism()
        for j in range(P):
            test_function(offspring[j], trainingData, trainingExpectedOutput)
        trainingPopAverage.append((addPopAverage(offspring) / trainingSize) * 100)
        trainingPopLowest.append((addPopLowest(offspring) / trainingSize) * 100)
        trainingPopHighest.append((addPopHighest(offspring) / trainingSize) * 100)
        print("generation " + str(i + 2) + ": Average:" + str(addPopAverage(offspring)) + " Lowest:" + str(addPopLowest(offspring)) + " Highest: " + str(addPopHighest(offspring)))
        bestInd.append(addBestInd(offspring))
        for j in range(len(bestInd)):
            test_function(bestInd[j], validationData, validationExpectedOutput)
        bestIndPlot.append((bestInd[0].error / validationSize) * 100)
        print("Best ind: " + str(bestInd[0].error))
        population = offspring.copy()
        offspring.clear()
        bestInd.clear()


    # trainingPopHighest
    # trainingPopAverage
    # trainingPopLowest
    # bestIndPlot

    print("\npop highest:\n")
    for i in range(len(trainingPopHighest)):
        print(trainingPopHighest[i])

    print("\npop average:\n")
    for i in range(len(trainingPopAverage)):
        print(trainingPopAverage[i])

    print("\npop lowest:\n")
    print(len(trainingPopLowest))
    for i in range(len(trainingPopLowest)):
        print(trainingPopLowest[i])

    print("\nbest ind:\n")
    print(len(bestIndPlot))
    for i in range(len(bestIndPlot)):
        print(bestIndPlot[i])


    # with open('results.csv', 'w', newline='') as csvfile:
        
    #     fieldnames = ['average', 'lowest', 'highest', 'bestInd']

    #     thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     thewriter.writeheader()

    #     for 

    # validation set 
    # for i in range(P):
    #     test_function(population[i], validationData, validationExpectedOutput)
    # validationPopLowest.append((addPopLowest(population) / validationSize) * 100)
    # validationPopAverage.append((addPopAverage(population) / validationSize) * 100)
    # validationPopHighest.append((addPopHighest(population) / validationSize) * 100)
    # print("generation 1: Average:" + str(addPopAverage(population)) + " Lowest:" + str(addPopLowest(population)))
    # for i in range(G - 1):
    #     newGeneration()
    #     mutation(offspring)
    #     for j in range(P):
    #         test_function(offspring[j], validationData, validationExpectedOutput)
    #     eliteism()
    #     for j in range(P):
    #         test_function(offspring[j], validationData, validationExpectedOutput)
    #     validationPopAverage.append((addPopAverage(offspring) / validationSize) * 100)
    #     validationPopLowest.append((addPopLowest(offspring) / validationSize) * 100)
    #     validationPopHighest.append((addPopHighest(offspring) / validationSize) * 100)
    #     print("generation " + str(i + 2) + ": Average:" + str(addPopAverage(offspring)) + " Lowest:" + str(addPopLowest(offspring)))
    #     population = offspring.copy()
    #     offspring.clear()


    # graphs

    plt.plot(np.array(trainingPopAverage), label="training average")#, label="training average"))
    plt.plot(np.array(trainingPopLowest), label="training lowest")#, label="training lowest"))
    # plt.plot(np.array(trainingPopHighest))
    # plt.plot(np.array(validationPopAverage))#, label="validation average"))
    # plt.plot(np.array(validationPopLowest))#, label="validation lowest"))
    # plt.plot(np.array(validationPopHighest))
    # # plt.plot(np.array(popHighest))
    plt.plot(np.array(bestIndPlot), label="best ind")

    plt.xlabel("Generations")
    plt.ylabel("Error")
    plt.legend()

    # plt.show()

    ###### 
    # code here to save results to file

    endTime = time.time()

    elapsedTime = endTime - startTime

    print("csvfilename: " + csvFileName)

    with open(csvFileName, 'w+', newline="") as csvfile:

        # fieldnames = ['Highest', 'Average', 'Lowest', 'bestInd', 
        # 'time elapsed', 'mutrateNum', 'MutstepNum', 'genesNum', 
        # 'populationNum', 'fileName']

        fieldnames = ['first', 'last']

        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

        thewriter.writeheader

        thewriter.writerow({'first': 'test1'})
        thewriter.writerow({'first': 'test2'})
        thewriter.writerow({'last': 'test1'})
        thewriter.writerow({'last': 'test2'})

        
        # for result in trainingPopLowest:
        #     thewriter.writerow({'Lowest':result})
        
        # for result in trainingPopAverage:
        #     thewriter.writerow({'average':result})

    print('run executed')

count = 0


MutrateNum = 0.4
MutstepNum = 0.2
GenesNum = 100
populationNum = 50
hnNum = 2
fileNum = 2
fileNameTest = "data" + str(fileNum) + ".txt"
csvFileName = ""


while count < 3:
    MutrateNum = 0.4
    MutstepNum = 0.2
    GenesNum = 100
    populationNum = 50
    hnNum = 2
    fileNum = 2
    fileNameTest = "data" + str(fileNum) + ".txt"
    csvFileName = ""

    def Mutrate(MutrateNum):
        for i in range(10):
            for j in range(3):
                csvFileName = "Mutrate_" + str(i) + "_" + str(j) + ".csv"
                print(csvFileName)
                        # MUTRATE, MUTSTEP, G, P, hidNodesNum, fileNameTest, csvFileName
                runMain(MutrateNum, MutstepNum, GenesNum, populationNum, hnNum, fileNameTest, csvFileName)
            MutrateNum += 0.1
        MutrateNum = 0.
        #csvFileName = ""

    def Mutstep():
        for i in range(10):
            for j in range(3):
                csvFileName = "Mutstep:" + str(i) + ":" + str(j)
                runMain(MutrateNum, MutstepNum, GenesNum, populationNum, fileNameTest, csvFileName)
            MutstepNum += 0.1
        MutstepNum = 0.1
        csvFileName = ""

    def MutrateAndMutstep():
        for i in range(10):
            for j in range(3):
                csvFileName = "MutrateAndMutstep:" + str(i) + ":" + str(j)
                runMain(MutrateNum, MutstepNum, GenesNum, populationNum, fileNameTest, csvFileName)
            MutstepNum += 0.1
            MutrateNum += 0.1
        MutstepNum = 0.4
        MutrateNum = 0.2
        csvFileName = ""

    def genesHigh():
        for i in range(5):
            for j in range(3):
                csvFileName = "genesHigh:" + str(i) + ":" + str(j)
                runMain(MutrateNum, MutstepNum, GenesNum, populationNum, fileNameTest, csvFileName)
            GenesNum += 50
        GenesNum = 100
        csvFileName = ""

    def genesLow():
        for i in range(4):
            for j in range(3):
                csvFileName = "genesLow:" + str(i) + ":" + str(j)
                runMain(MutrateNum, MutstepNum, GenesNum, populationNum, fileNameTest, csvFileName)
            GenesNum -= 25
        GenesNum = 100
        csvFileName = ""

    def populationHigh():
        for i in range(6):
            for j in range(3):
                csvFileName = "populationHigh:" + str(i) + ":" + str(j)
                runMain(MutrateNum, MutstepNum, GenesNum, populationNum, fileNameTest, csvFileName)
            populationNum += 20
        populationNum = 100
        csvFileName = ""

    def populationLow():
        for i in range(4):
            for j in range(3):
                csvFileName = "populationLow:" + str(i) + ":" + str(j)
                runMain(MutrateNum, MutstepNum, GenesNum, populationNum, fileNameTest, csvFileName)
            populationNum -= 20
        populationNum = 100
        csvFileName = ""

    def genesHighPopHigh():
        for i in range(6):
            for j in range(3):
                csvFileName = "genesHighPopHigh:" + str(i) + ":" + str(j)
                runMain(MutrateNum, MutstepNum, GenesNum, populationNum, fileNameTest, csvFileName)
            populationNum += 20
            GenesNum += 20
        populationNum = 150
        GenesNum = 100
        csvFileName = ""



    def genesHighPopLow():
        for i in range(4):
            for j in range(3):
                csvFileName = "genesHighPopLow:" + str(i) + ":" + str(j)
                runMain(MutrateNum, MutstepNum, GenesNum, populationNum, fileNameTest, csvFileName)
            populationNum -= 20
        populationNum = 100
        GenesNum = 50
        csvFileName = ""


    # def genesLowPopLow():
    #     for i in range(4):
    #         for j in range(3):
    #             runMain(MutrateNum, MutrateNum, GenesNum, populationNum, fileName)
    #         populationNum -= 20
    #     populationNum = 100
    #     GenesNum = 30

    # def genesLowPopHigh():

    def hiddenNodes():
        for i in range(7):
            for j in range(3):
                csvFileName = "hiddenNodes:" + str(i) + ":" + str(j)
                runMain(MutrateNum, MutstepNum, GenesNum, populationNum, fileNameTest, csvFileName)
            hnNum += 1
        hnNum = 2
        csvFileName = ""

    count += 1

    print("hi")
    #runMain(MutrateNum, MutstepNum, GenesNum, populationNum, hnNum, fileNameTest, csvFileName)

    Mutrate(MutrateNum)
        