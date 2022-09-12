from problem import TravellingSalesperson
from genetic.algorithm import GeneticAlgorithm
from genetic.selection import wheel
from genetic.crossover import OrderX
from genetic.mutation import CentreInverseM

tsp = TravellingSalesperson('instances/test.txt')
# print(tsp.fitness([
#     1, 28, 6, 12, 9, 26,
#     3, 29, 5, 21, 2, 20,
#     10, 4, 15, 18, 14,
#     17, 22, 11, 19, 25,
#     7, 23, 8, 16, 13, 24,
# ]))
algorithm = GeneticAlgorithm(
    tsp,
    wheel,
    OrderX,
    CentreInverseM,
    pop_size=300,
    elitism_ratio=0.25,
    max_iter=200,
    # max_iter_wo_improv=50,
)

algorithm.run()