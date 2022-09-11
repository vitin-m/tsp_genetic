import numpy as np

def OrderX(parent1 : np.ndarray, parent2 : np.ndarray):
    lo = np.random.randint(00, len(parent1))
    hi = np.random.randint(lo, len(parent1))
    
    missing_seq1 = np.asarray([x for x in parent1 if x not in parent2[lo:hi]])
    missing_seq2 = np.asarray([x for x in parent2 if x not in parent1[lo:hi]])
    
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)
    
    if hi != len(parent1):
        offspring1[hi:] = missing_seq2[:len(offspring1[hi:])]
        offspring2[hi:] = missing_seq1[:len(offspring1[hi:])]
    
    offspring1[lo:hi] = parent1[lo:hi]
    offspring2[lo:hi] = parent2[lo:hi]
    
    if lo != 0:
        offspring1[:lo] = missing_seq2[-lo:]
        offspring2[:lo] = missing_seq1[-lo:]
    
    return offspring1, offspring2


def main():
    p1, p2 = np.asarray([1,3,4,2]), np.asarray([4,3,2,1])
    print(OrderX(p1, p2))


if __name__ == '__main__':
    main()