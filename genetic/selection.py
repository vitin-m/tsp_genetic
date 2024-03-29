import numpy as np
from typing import Any

def wheel(fit : np.ndarray, amt : int) -> np.ndarray[Any, int]:
    total_fitness = np.sum(fit)
    individual_prob = np.flip(fit / total_fitness)
    cumulative_prob = np.cumsum(individual_prob)
    
    idxes = np.searchsorted(cumulative_prob, np.random.random(amt), side='right')
    return idxes

def main():
    pop = np.asarray([[1,2,3],[2,3,1],[2,1,3]])
    fit = np.asarray([2, 3, 5])
    print(wheel(fit, 2))


if __name__ == '__main__':
    main()