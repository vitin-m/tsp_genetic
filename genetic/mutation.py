import numpy as np

def CentreInverseM(individual : np.ndarray):
    div = np.random.randint(0, len(individual))
    new_ind = np.empty_like(individual)
    new_ind = np.flip(individual[div:])
    new_ind = np.flip(individual[:div])
    
    return new_ind
    