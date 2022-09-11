import numpy as np

from problem import TravellingSalesperson

class GeneticAlgorithm:
    def __init__(
        self,
        tsp : TravellingSalesperson,
        select : function,
        crossover : function,
        mutation : function,
        **kwargs,
    ):
        self.tsp = tsp
        self.select = select
        self.crossover = crossover
        self.mutation = mutation
        
        self.pop_size = kwargs.get('pop_size')\
            if kwargs.get('pop_size') is not None else 100
            
        self.crossover_prob = kwargs.get('crossover_prob')\
            if kwargs.get('crossover_prob') is not None else 0.5
            
        self.mutation_prob = kwargs.get('mutation_prob')\
            if kwargs.get('mutation_prob') is not None else 0.01
            
        self.elitism_ratio = kwargs.get('elitism_ratio')\
            if kwargs.get('elitism_ratio') is not None else 0.1
        
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
        
        while(not self.should_stop()):
            # fitness
            self.fit = np.asarray([self.tsp.fitness(perm)
                               for perm in self.pop])
            # select
            # crossover
            # mutation
            # avaliar pop gerada p/ crossover e mutation?
    

def main():
    x = GeneticAlgorithm()


if __name__ == '__main__':
    main()