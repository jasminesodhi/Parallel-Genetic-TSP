import random
import os
from math import sqrt, pow
import json
import numpy as np

cities = []
def readDataSet(filename):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.dirname(dir_path) + "/dataset/" + filename
    
    with open(file_path, 'r') as f:
        temp = np.array(json.load(f))
        for location in temp:
            cities.append(City(int(location[0]),int(location[1]),int(location[2])))

def getDistMatrix():
    dist = [[0 for i in cities] for j in cities]
    for i in range(len(cities)):
        for j in range(len(cities)):
            dist[i][j] = cities[i].getDistance(cities[j])
    return dist

def getRandomCity():
    index  = random.randint(0, len(cities)-1)   
    return cities[index]

class Population:
    def __init__(self,populationSize,init):
        self.popSize = populationSize
        self.tours = [None] * self.popSize
        if init :
            self.initPopulation()
            self.calculateFitnessForAll()
            self.sortPopulation()           
            self.fittest = self.tours[0]

    def initPopulation(self):
        for i in range(self.popSize):
            self.tours[i] = Tour(True)                

    def sortPopulation(self):
        for i in range(self.popSize-1):
            index = i
            for j in range(i+1 , self.popSize):
                if self.tours[j].compare(self.tours[i]) > 0 :
                    index = j 
            tmp = self.tours[i]
            self.tours[i] = self.tours[index]
            self.tours[index] = tmp
    
    def calculateFitnessForAll(self):
        for i in range(self.popSize):
            self.tours[i].calculateFitness()  
                  
class Tour:
    def __init__(self , init):
        self.nbrCities = len(cities)
        self.cities = [None] * self.nbrCities
        if init :
            self.initTour()
            self.fitness = self.calculateFitness()
        
    def __str__(self):
        path = ""
        for i in range(self.nbrCities):
            if i != self.nbrCities -1 :
                path += str(self.cities[i].id) + " -> "
            else :
                path += str(self.cities[i].id) + " . "             
        return path                   

    def initTour(self):
        for i in range(self.nbrCities):
                city = getRandomCity()
                while self.contain(city) :
                    city = getRandomCity()
                self.cities[i] = city   

    def contain(self,city):
        for i in range(self.nbrCities):
            if self.cities[i] != None :
                if self.cities[i].id == city.id :
                    return True
        return False

    def getIndexOf(self,city):
        for i in range(self.nbrCities):
            if self.cities[i].id == city.id :
                return i
        return -1        

    def calculateFitness(self):
        self.fitness = 0
        for i in range(self.nbrCities -1):
            self.fitness += self.cities[i].getDistance(self.cities[i+1])
        self.fitness += self.cities[len(self.cities) -1].getDistance(self.cities[0])
   
    def compare(self, other):
        return 1 if self.fitness < other.fitness else -1      
    
class City:
    def __init__(self,id,x,y):
        self.id = id
        self.x = x
        self.y = y
    
    def __str__(self):
        return "city {" + str(self.id) + " , " + str(self.x) +" , " + str(self.y) + "}"
    
    def getDistance(self,c):
        if self != None and c != None :
            distance = sqrt( pow(self.x - c.x , 2) + pow(self.y - c.y , 2))
            return distance   
