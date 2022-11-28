import json, random
import numpy as np

class DNA():

    @staticmethod
    def calc_fitness(cities):
        d = np.roll(cities,-1, axis=0) - cities
        if len(d.shape) == 1:
            dist = np.sum(abs(d))
        else:
            dist_sqr = np.sum(d**2, axis=1)
            dist = np.sum(np.sqrt(dist_sqr))

        return dist

    @staticmethod
    def crossover(dna1, dna2):
        dna1 = np.array(dna1)
        dna2 = np.array(dna2)

        start, end = DNA.calc_points(len(dna1))
        
        section1, section2 = dna1[start:end], dna2[start:end]
        leftover1 = np.setdiff1d(dna2, section1, assume_unique=True)
        leftover2 = np.setdiff1d(dna1, section2, assume_unique=True)

        child1, child2 = np.empty_like(dna1), np.empty_like(dna2)

        child1[:start] = leftover1[:start]
        child1[start:end] = section1
        child1[end:] = leftover1[start:]

        child2[:start] = leftover2[:start]
        child2[start:end] = section2
        child2[end:] = leftover2[start:]

        return child1, child2

    @staticmethod
    def mutation(dna, mrate):
        if (np.random.rand() < mrate):
            start, end = DNA.calc_points(dna)
            dna[start:end] = np.flip(dna[start:end])
        return dna

    @staticmethod
    def crossoverAndMutation(dna1, dna2, mrate=None):
        child1, child2 = DNA.crossover(dna1, dna2)

        if (mrate != None):
            child1 = DNA.mutation(child1, mrate)
            child2 = DNA.mutation(child2, mrate)

        return child1, child2

    @staticmethod
    def calc_points(length):
        return np.sort(np.random.choice(length, 2, replace=False))

class GA():
    def getParents(self):
        population_len = self.options.population_size
        elite_size = round(population_len*self.options.elite_fraction*0.5) *2 
        number_of_parents = (population_len-elite_size)//2
        return elite_size, number_of_parents

    def getLocation(self, filename):
        with open(filename, 'r') as f:
            self.locations = np.array(json.load(f))[:self.options.num_locations]

    def getIndex(self, num_parents, normalized_fitness):
        return np.random.choice(self.options.population_size, \
                    size=(num_parents, 2), p=normalized_fitness)
        
    def getPopulation(self):
        np.random.seed(random.randint(0, 10000))
        
        num_locations = self.options.num_locations
        population_size = self.options.population_size

        initial_population = np.empty((population_size, num_locations), dtype=int)
        for i in range(population_size):
            initial_population[i] = np.random.permutation(num_locations)

        return initial_population

    def getFitnessValues(self, population):
        fitnesses = np.empty(len(population))
        total = 0
        best = np.inf
        for i, idx in enumerate(population):
            fitness = DNA.calc_fitness(self.locations[idx])
            fitnesses[i] = 1/fitness
            total += fitnesses[i]

            if fitness < best:
                best = fitness

        return fitnesses/total, best
    