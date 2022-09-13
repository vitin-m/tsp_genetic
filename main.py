from problem import TravellingSalesperson
from genetic.algorithm import GeneticAlgorithm
from genetic.selection import wheel
from genetic.crossover import OrderX
from genetic.mutation import CentreInverseM, DisplacementM, ExchangeM

tsp = TravellingSalesperson(r'instances/test_trab.tsp')

algorithm = GeneticAlgorithm(
    tsp,
    wheel,
    OrderX,
    CentreInverseM,
    crossover_with_replacement=True,
    out_of_bounds_fix='regenerate',  # 'ignore', 'regenerate', 'randomize', 'penalize'
    pop_size=10,
    crossover_prob=0.70,
    mutation_prob=0.25,
    elitism_ratio=0.10,
    max_iter=1000,
    max_iter_wo_improv=200,
)

algorithm.run()