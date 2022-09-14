import numpy as np

def longest_common_subsequence(ind1 : np.ndarray, ind2 : np.ndarray) -> int:
    mem = np.ndarray((len(ind1), len(ind2)))
    longest = 0
    
    for i in range(len(ind1)):
        for j in range(len(ind2)):
            if ind1[i] == ind2[j]:
                if i == 0 or j == 0:
                    mem[i, j] = 1
                else:
                    mem[i, j] = mem[i - 1, j - 1] + 1
                if mem[i, j] > longest:
                    longest = mem[i, j]
            else:
                mem[i, j] = 0
    
    return longest


def levenshtein(ind1 : np.ndarray, ind2 : np.ndarray) -> int:
    mem = np.zeros((2, len(ind1)))
    
    for i in range(len(ind2)):
        mem[1,0] = i + 1
    
    for i in range(len(ind2)):
        for j in range(len(ind1)):
            if j < len(ind1) - 1:
                cdel = mem[0, j + 1] + 1
            cins = mem[1, j] + 1
            if ind1[i] == ind2[j]:
                csub = mem[0, j]
            else:
                csub = mem[0, j] + 1
            
            print(cdel, cins, csub)
            if j < len(ind1) - 1:
                mem[1, j + 1] = np.min((cdel, cins, csub))

        mem[[0, 1]] = mem[[1, 0]]
    
    return mem[0, -1]
    
    


def main():
    p1 = np.asarray([1, 2, 3])
    p2 = np.asarray([2, 3, 4])
    print(levenshtein(p1, p2))


if __name__ == '__main__':
    main()