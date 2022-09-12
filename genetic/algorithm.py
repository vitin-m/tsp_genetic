import numpy as np

from problem import TravellingSalesperson

class GeneticAlgorithm:
    def __init__(
        self,
        tsp : TravellingSalesperson,
        selection : function,
        crossover : function,
        mutation : function,
        **kwargs,
    ):
        self.tsp = tsp
        self.select = selection
        self.crossover = crossover
        self.mutate = mutation
        
        self.pop_size = (kwargs.get('pop_size')
            if kwargs.get('pop_size') is not None else 100)
            
        self.crossover_prob = (kwargs.get('crossover_prob')
            if kwargs.get('crossover_prob') is not None else 0.5)
            
        self.mutation_prob = (kwargs.get('mutation_prob')
            if kwargs.get('mutation_prob') is not None else 0.01)
            
        self.elitism_ratio = (kwargs.get('elitism_ratio')
            if kwargs.get('elitism_ratio') is not None else 0.1)
        
        self.max_iter = kwargs.get('max_iter')
        self.max_iter_wo_improv = kwargs.get('max_iter_wo_improv')
        
        if self.max_iter_wo_improv is None:
            if self.max_iter is None: self.max_iter = 100
            
            def max_iter_stop(self) -> bool:
                return self.iter > self.max_iter
            
            self.should_stop = max_iter_stop
        
        if self.max_iter_wo_improv is not None:
            if self.max_iter is None: self.max_iter = 100
            
            def max_iter_wo_improv_stop(self) -> bool:
                return self.iter_wo_improv > self.max_iter_wo_improv or\
                    self.iter > self.max_iter
            
            self.should_stop = max_iter_wo_improv_stop
    
    
    def run(self):
        self.iter = 0
        self.iter_wo_improv = 0
        
        self.pop = self.tsp.gen_random_pop(100)
        self.fit = np.asarray([self.tsp.fitness(perm)
                               for perm in self.pop])
        
        gbestIdx = self.fit.argmin()
        self.gbest_fit = self.fit[gbestIdx]
        self.gbest_pop = self.pop[gbestIdx]
        
        n_crossovers = (self.crossover_prob * self.pop_size) // 2
        n_elitism = self.elitism_ratio * self.pop_size
        
        
        while(not self.should_stop()):
            # Get elite (implementar)
            # elite = np.argpartition(self.pop, -n_elitism, axis=1)[-n_elitism:]
            
            temp_fit = list(self.fit[:])
            temp_pop = list(self.pop[:])
            new_pop = list()
            for _ in range(n_crossovers):
                # Selection
                idx_parent1, idx_parent2 = self.select(temp_fit, 2)
                
                # Crossover w/ replacement
                offspring1, offspring2 = self.crossover(temp_pop[idx_parent1], temp_pop[idx_parent2])
                
                del temp_pop[idx_parent1]
                del temp_pop[idx_parent2]
                
                del temp_fit[idx_parent1]
                del temp_fit[idx_parent2]
                
                # Mutation on offsprings
                offspring1 = self.mutate(offspring1, self.mutation_prob)
                offspring2 = self.mutate(offspring2, self.mutation_prob)
                
                new_pop.append([offspring1, offspring2])
            
            temp_pop.append(new_pop)
            
            self.pop = np.asarray(temp_pop)
            self.fit = np.asarray([self.tsp.fitness(perm)
                               for perm in self.pop])
            
            gbestIdx = self.fit.argmin()
            self.gbest_fit = self.fit[gbestIdx]
            self.gbest_pop = self.pop[gbestIdx]
            
            # avaliar pop gerada p/ crossover e mutation?
    

def main():
    x = GeneticAlgorithm()


if __name__ == '__main__':
    main()