import numpy as np

def CentreInverseM(individual : np.ndarray, mutation_prob : float) -> np.ndarray:
    if np.random.random(1) > mutation_prob: 
        return individual
    
    div = np.random.randint(0, len(individual) - 1)
    new_individual = np.empty_like(individual)
    new_individual[:div] = np.flip(individual[:div])
    new_individual[div:] = np.flip(individual[div:])
    
    return new_individual

def DisplacementM(individual : np.ndarray, mutation_prob : float) -> np.ndarray:
    if np.random.random(1) > mutation_prob:
        return individual
    
    lo = np.random.randint(0, len(individual) - 1)
    hi = np.random.randint(lo + 1, len(individual))
    mi = np.random.randint(0, len(individual) - (hi - lo))
    
    new_individual = list()
    new_individual.extend(individual[:lo])
    new_individual.extend(individual[hi:])
    new_individual[mi:mi] = list(individual[lo:hi])
    new_individual = np.asarray(new_individual)
    
    return new_individual

def ExchangeM(individual : np.ndarray, mutation_prob : float) -> np.ndarray:
    if np.random.random(1) > mutation_prob:
        return individual
    
    new_individual = individual[:]
    for mi in range(1, len(individual) - 1):
        if np.random.random(1) > mutation_prob: continue
        new_individual[mi-1:mi-1] = individual[mi+1]
        new_individual[mi+1:mi+1] = individual[mi-1]

    return new_individual