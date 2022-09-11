from problem import TravellingSalesperson
from genetic.algorithm import GeneticAlgorithm
from genetic.selection import wheel
from genetic.crossover import OrderX
from genetic.mutation import CentreInverseM

tsp = TravellingSalesperson('instances/test.txt')
algorithm = GeneticAlgorithm(
    tsp,
    wheel,
    OrderX,
    CentreInverseM
)

algorithm.run()