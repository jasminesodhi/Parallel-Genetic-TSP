from re import L
import matplotlib.pyplot as plt
import numpy as np
from serial.helper import Population, Tour, getDistMatrix, getRandomCity
import random

'''
CROSSOVER_STRATEGY = ORDER_CROSSOVER | TWO_POINT_CROSSOVER | ONE_POINT_CROSSOVER | GREEDY
'''

CROSSOVER_STRATEGY = "TWO_POINT_CROSSOVER"

class GeneticAlgorithm :

    def __init__(self, nbrGenerations, popSize, elitismSize, poolSize, mutationRate):
        self.nbrGenerations = nbrGenerations
        self.popSize = popSize
        self.elitismSize = elitismSize
        self.poolSize = poolSize
        self.mutationRate = mutationRate
        self.initialPopulation = Population(self.popSize , True)
        self.fitnesses = np.zeros(self.nbrGenerations)
        self.distMatrix = getDistMatrix()
        print("Initial Fitness : " , self.initialPopulation.fittest.fitness)
        print("Best Tour : ",self.initialPopulation.fittest)
        newPopulation = self.initialPopulation
        generationCounter = 0
        for i in range(self.nbrGenerations):
            newPopulation = self.reproduction(newPopulation)
            self.fitnesses[generationCounter] = newPopulation.fittest.fitness
            generationCounter += 1

            print("Generation : ", generationCounter  )
            print("Fitness : ", newPopulation.fittest.fitness)
            print("Best Tour : ",newPopulation.fittest)
            print("\n\n")

        self.displayTheResult()    

    def reproduction(self, pop):
        newpop = Population(pop.popSize,False)
        elitismSubPopulation = self.elitismSelection(pop)
        
        for index in range(self.elitismSize):
            newpop.tours[index] = elitismSubPopulation[index]
        
        for i in range(index , pop.popSize): 
            parent1 = self.touranmentSelection(pop)
            parent2 = self.touranmentSelection(pop) 
              
            if CROSSOVER_STRATEGY == 'ORDER_CROSSOVER':
                child = self.Ox1CrossOver(parent1, parent2)
            elif CROSSOVER_STRATEGY == 'TWO_POINT_CROSSOVER':
                child = self.twoPointCrossOver(parent1, parent2)
            elif CROSSOVER_STRATEGY == 'ONE_POINT_CROSSOVER':
                child = self.onePointCrossOver(parent1, parent2)
            else:
                child = self.greedyCrossOver(parent1, parent2)
            
            child = self.SwapMutation(child)
            child.calculateFitness()
            newpop.tours[i] = child
        
        newpop.calculateFitnessForAll()
        newpop.sortPopulation()
        newpop.fittest = newpop.tours[0]
        return newpop

    def elitismSelection(self, pop):
        pop.sortPopulation()
        elitismSubPopulation = pop.tours[:self.elitismSize + 1]
        return elitismSubPopulation

    def touranmentSelection(self, pop):
        pool = [None] * self.poolSize
        for i in range(self.poolSize):
            index = random.randint(0,self.popSize -1)
            pool[i] = pop.tours[index]
        self.sortSubPopulation(pool)
        return pool[0]

    def sortSubPopulation(self, sub):
        for i in range(self.poolSize):
            index = i
            for j in range(i+1 , self.poolSize):
                if sub[j].compare(sub[i]) > 0 :
                    index = j 
            tmp = sub[i]
            sub[i] = sub[index]
            sub[index] = tmp
    
    def Ox1CrossOver(self, parent1, parent2):
        child = Tour(False)

        start = random.randint(0, parent1.nbrCities)
        end   = random.randint(0, parent1.nbrCities)

        while start >= end :
            start = random.randint(0, parent1.nbrCities)
            end = random.randint(0, parent1.nbrCities)   
        
        for i in range(start,end):
            child.cities[i] = parent1.cities[i]            
        
        for i in range(parent2.nbrCities):
            if not child.contain(parent2.cities[i]) :
                for j in range(parent2.nbrCities):
                    if child.cities[j] is None :
                        child.cities[j] = parent2.cities[i]
                        break
        return child
    
    def twoPointCrossOver(self, parent1, parent2):
        child = Tour(False)
        for i in range(parent1.nbrCities):
            child.cities[i] = parent1.cities[i]
        child.fitness = parent1.fitness
        
        start = random.randint(0,parent1.nbrCities)
        end   = random.randint(0,parent1.nbrCities)
        
        while start >= end:
            start = random.randint(0,parent1.nbrCities)
            end = random.randint(0,parent1.nbrCities)
        
        for i in range(start, end):
            city = parent2.cities[i]
            indexOfCity = child.getIndexOf(city)
            child.cities[indexOfCity] = child.cities[i]
            child.cities[i] = city
        
        return child
    
    def onePointCrossOver(self, parent1, parent2):
        crossOverPoint = random.randint(0,parent1.nbrCities)
        
        while (crossOverPoint == 0 or crossOverPoint == parent1.nbrCities):
            crossOverPoint = random.randint(0,parent1.nbrCities)
            
        for i in range(crossOverPoint):
            city1 = parent1.cities[i]
            indexOfCity = parent2.getIndexOf(city1)
            city2 = parent2.cities[i]
            parent2.cities[i] = city1
            parent2.cities[indexOfCity] = city2
        
        return parent1

    def greedyCrossOver(self, parent1, parent2):
        child = Tour(False)
        child.cities[0] = parent1.cities[0]
        child_current = 0
        parent1_current = 1
        parent2_current = 0

        while child_current<child.nbrCities-1:
            best_neighbor = None
            if self.distMatrix[child.cities[child_current].id-1][parent1.cities[parent1_current].id-1]<self.distMatrix[child.cities[child_current].id-1][parent2.cities[parent2_current].id-1]:
                if not child.contain(parent1.cities[parent1_current]):
                    best_neighbor = parent1.cities[parent1_current]
                    parent1_current += 1
            else:
                if not child.contain(parent2.cities[parent2_current]):
                    best_neighbor = parent2.cities[parent2_current]
                    parent2_current += 1
            
            if best_neighbor == None:
                while True:
                    best_neighbor = getRandomCity()
                    if not child.contain(best_neighbor):
                        break
            child_current += 1
            child.cities[child_current] = best_neighbor
        
        return child

    def SwapMutation(self, child):
        for i in range(child.nbrCities):
            mutationProbability = random.random()
            if mutationProbability < self.mutationRate :
                    mutationPoint = random.randint(0 , child.nbrCities -1)
                    tmp = child.cities[mutationPoint]
                    child.cities[mutationPoint] = child.cities[i]
                    child.cities[i] = tmp
        return child

    def displayTheResult(self):
        x = np.arange(0,self.nbrGenerations)
        plt.plot(x,self.fitnesses)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Value Over Generations ")
        plt.show()
        