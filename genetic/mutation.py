import numpy as np

def CentreInverseM(individual : np.ndarray, mutation_prob : float) -> np.ndarray:
    if np.random.random(1) > mutation_prob: 
        return individual
    
    div = np.random.randint(0, len(individual))
    new_ind = np.zeros_like(individual)
    new_ind[div:] = np.flip(individual[div:])
    new_ind[:div] = np.flip(individual[:div])
    
    return new_ind