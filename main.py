from problem import TravellingSalesperson
from genetic.algorithm import GeneticAlgorithm
from genetic.selection import wheel

tsp = TravellingSalesperson('instances/test.txt')
algorithm = GeneticAlgorithm(
    tsp,
    wheel,
    # crossover
    # mutation
)

algorithm.run()