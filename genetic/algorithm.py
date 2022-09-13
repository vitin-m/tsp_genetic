import numpy as np
from typing import Callable

from permutation import get_n_idx
from problem import TravellingSalesperson

class GeneticAlgorithm:
    def __init__(
        self,
        tsp : TravellingSalesperson,
        selection : Callable[[np.ndarray, int], np.ndarray],
        crossover : Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        mutation : Callable[[np.ndarray, float], np.ndarray],
        **kwargs,
    ):
        self.tsp = tsp
        self.select = selection
        self.crossover = crossover
        self.mutate = mutation
        
        self.crossover_with_replacement = (kwargs.get('crossover_with_replacement')
            if kwargs.get('crossover_with_replacement') is not None else True)
        
        # Possible = 'regenerate', 'randomize', 'penalize'
        self.out_of_bounds_fix = (kwargs.get('out_of_bounds_fix')
            if kwargs.get('out_of_bounds_fix') is not None else 'ignore')
        
        self.pop_size = (kwargs.get('pop_size')
            if kwargs.get('pop_size') is not None else 200)
            
        self.crossover_prob = (kwargs.get('crossover_prob')
            if kwargs.get('crossover_prob') is not None else 0.5)
            
        self.mutation_prob = (kwargs.get('mutation_prob')
            if kwargs.get('mutation_prob') is not None else 0.1)
            
        self.elitism_ratio = (kwargs.get('elitism_ratio')
            if kwargs.get('elitism_ratio') is not None else 0.15)
        
        self.max_iter = kwargs.get('max_iter')
        self.max_iter_wo_improv = kwargs.get('max_iter_wo_improv')
        
        if self.max_iter_wo_improv is None:
            if self.max_iter is None: self.max_iter = 100
            self.should_stop = self.max_iter_stop
        
        if self.max_iter_wo_improv is not None:
            if self.max_iter is None: self.max_iter = 100
            self.should_stop = self.max_iter_wo_improv_stop
    
    def max_iter_stop(self) -> bool:
        return self.iter >= self.max_iter
    
    def max_iter_wo_improv_stop(self) -> bool:
        return self.iter_wo_improv > self.max_iter_wo_improv or\
            self.iter >= self.max_iter
    
    def run(self):
        self.iter = 0
        self.iter_wo_improv = 0
        
        self.pop = self.tsp.get_random_pop(self.pop_size)
        self.fit = np.asarray([self.tsp.fitness(perm)
                               for perm in self.pop])
        
        sort = np.argsort(self.fit)
        self.fit = self.fit[sort]
        self.pop = self.pop[sort]
        
        self.pop_hist = [self.pop]
        
        gbestIdx = self.fit.argmin()
        self.gbest_fit = self.fit[gbestIdx]
        self.gbest_pop = self.pop[gbestIdx]
        
        n_crossovers = int((self.crossover_prob * self.pop_size))
        n_elitism = int(self.elitism_ratio * self.pop_size)
        
        while(not self.should_stop()):            
            temp_fit = list(self.fit[:])
            temp_pop = list(self.pop[:])
            new_pop = list()
            
            Xcount = 0
            while Xcount < n_crossovers:
                # Selection
                idx_parent1, idx_parent2 = self.select(temp_fit, 2)
                
                if np.any(temp_pop[idx_parent1] is None) or np.any(temp_pop[idx_parent2] is None):
                    print('Selection miss')
                
                # Crossover w/ replacement
                offspring1, offspring2 = self.crossover(temp_pop[idx_parent1], temp_pop[idx_parent2])
                
                if np.any(offspring1 is None) or np.any(offspring2 is None):
                    print('Crossover miss')
                
                # Mutation on offsprings
                offspring1m = self.mutate(offspring1, self.mutation_prob)
                offspring2m = self.mutate(offspring2, self.mutation_prob)
                
                if np.any(offspring1m is None) or np.any(offspring2m is None):
                    print('Mutation miss')
                
                if self.out_of_bounds_fix == 'regenerate':
                    if not self.tsp.is_valid(offspring1m) or not self.tsp.is_valid(offspring2m):
                        continue
                
                if self.crossover_with_replacement:
                    del temp_pop[idx_parent1]
                    del temp_fit[idx_parent1]
                    
                    if idx_parent1 > idx_parent2:
                        del temp_pop[idx_parent2]
                        del temp_fit[idx_parent2]
                    elif idx_parent1 < idx_parent2:
                        del temp_pop[idx_parent2 - 1]
                        del temp_fit[idx_parent2 - 1]
                
                Xcount += 2
                new_pop.extend([offspring1m, offspring2m])
            
            if self.crossover_with_replacement:
                temp_pop = temp_pop[len(new_pop):]
            
            temp_pop.extend(new_pop)
            temp_pop = np.asarray(temp_pop)
            old_elite = get_n_idx(self.fit, n_elitism)
            new_weak = get_n_idx(self.fit, n_elitism, False)
            
            for idx_old, idx_new in zip(old_elite, new_weak):
                temp_pop[idx_new] = self.pop[idx_old]
            
            self.pop = temp_pop[:]
            self.fit = np.asarray([self.tsp.fitness(perm)
                               for perm in self.pop])
            
            sort = np.argsort(self.fit)
            self.fit = self.fit[sort]
            self.pop = self.pop[sort]
            
            self.iter += 1
            gbestIdx = self.fit.argmin()
            if self.gbest_fit == self.fit[gbestIdx]:
                self.iter_wo_improv += 1
            else:
                self.iter_wo_improv = 0
            self.gbest_fit = self.fit[gbestIdx]
            self.gbest_pop = self.pop[gbestIdx]
            self.pop_hist.append(self.pop)
            
            print(f'Iter: {self.iter} -> Best fit: {self.gbest_fit}')