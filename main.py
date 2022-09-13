from problem import TravellingSalesperson
from genetic.algorithm import GeneticAlgorithm
from genetic.selection import wheel
from genetic.crossover import OrderX, PartiallyMappedX
from genetic.mutation import CentreInverseM, DisplacementM, ExchangeM

tsp = TravellingSalesperson(r'instances/test.tsp')

algorithm = GeneticAlgorithm(
    tsp,
    wheel,
    OrderX,
    CentreInverseM,
    crossover_with_replacement=True,
    out_of_bounds_fix='ignore',  # 'ignore', 'regenerate', 'randomize'
    pop_size=300,
    crossover_prob=0.70,
    mutation_prob=0.25,
    elitism_ratio=0.20,
    max_iter=500,
    max_iter_wo_improv=100,
    # verbose=True,
)

algorithm.run()