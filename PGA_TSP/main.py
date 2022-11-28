#%%
import json
import numpy as np
import matplotlib.pyplot as plt
import time

import json

from mapreduce import islandGA

from serial.ga import GeneticAlgorithm
from serial import helper

class Timer:   
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

class Driver():
    def __init__(self):
        self.model = None

    def run(self, num_repetitions=1):
        if self.model == None:
            self.args += ['--num-migrations', str(self.options.num_migrations)]
            self.args += ['--num-islands', str(self.options.num_islands)]
            self.model = islandGA(self.args)
            self.formatFile()
        print("Running '{}' model.".format(self.options.model_type))
        
        sh = []
        times = []
        for _ in range(num_repetitions):
            with Timer() as t:
                result = self._run_mrjob()
            
            sh.append(result[1][-1])
            times.append(t.interval)

        print("\nFinished \n", self.options)
        print('Job finished in {} +- {} seconds'.format(np.mean(times), np.std(times)))
        print('Shortest distance is {} +- {}\n'.format(np.mean(sh), np.std(sh)))
        
        if self.options.plot:
            self.plot(result)

    def formatFile(self):
        num_lines = self.options.population_size
        if self.options.model_type == 'island':
            num_lines = self.options.num_islands 

        lines = np.arange(num_lines, dtype=int)
        np.savetxt('dataset/input.txt', lines, fmt='%d')

    def plot(self, result):
        dna, history = result
        self.getLocations()

        if self.locations.size == 2:
            self.plotRoute(dna)
            plt.show(block=False)

        self.plotTrend(history)
        plt.show()

    def plotRoute(self, dna):
        dna = np.pad(dna, (0, 1), 'wrap')
        loc = self.locations.T[:,dna]
        plt.figure()
        plt.scatter(*loc)
        plt.plot(*loc)
        plt.title("Best route path after {} iterations".format(self.options.num_iterations))
        plt.xlabel("x")
        plt.ylabel("y")

    def plotTrend(self, arr):
        plt.figure()
        plt.plot(arr)
        plt.title("Convergence rate")
        plt.xlabel("Number of iterations")
        plt.ylabel("Distance")

    def _run_mrjob(self):
        with self.model.make_runner() as runner:
            runner.run()

            if self.options.model_type == "island":
                for idx, dist in self.model.parse_output(runner.cat_output()):
                    distances = dist
                    
            else:
                return None, [0]
                
        return np.array(idx, dtype=int), np.array(distances)

    def getLocations(self):
        with open('data/locations.json', 'r') as f:
            temp = np.array(json.load(f))
            self.locations = np.array([[points[1],points[2]] for points in temp])

if __name__ == '__main__':
    type = input('Input algorithm type (serial / parallel): ')
    
    if type == 'serial':
        helper.readDataSet("cities50.json")
        algo_serial = GeneticAlgorithm(
                nbrGenerations=600, 
                popSize=400, 
                elitismSize=25, 
                poolSize=10, 
                mutationRate=0.1)
    else:
        algo_parallel = Driver()
        algo_parallel.run()
