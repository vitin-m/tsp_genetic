class PermutationDict(dict):
    def _tupOrderPermut(self, k : tuple[str, str]) -> str:
        if len(k) != 2: raise Exception('Wrong tuple size: must be 2.')
        if k[0] is False or k[1] is False: raise Exception('Two arguments required.')
        
        if k[0] < k[1]: return k[0], k[1]
        return k[1], k[0]
    
    def __setitem__(self, k : tuple[str, str], w: int):
        k = self._tupOrderPermut(k) 
        return super().__setitem__(k, w)
    
    def __getitem__(self, k : tuple[str, str]):
        k = self._tupOrderPermut(k)
        return super().__getitem__(k)
    
    def __delitem__(self, k : tuple[str, str]) -> None:
        k = self._tupOrderPermut(k)
        return super().__delitem__(k)
    
    def __contains__(self, k : tuple[str, str]) -> bool:
        k = self._tupOrderPermut(k)
        return super().__contains__(k)
    
    def get(self, k1 : str, k2 : str, default = None) -> str | None:  
        try:
            return self.__getitem__((k1, k2))
        except Exception:
            return default
        
    def pop(self, k1 : str, k2 : str, default = None) -> str | None:
        self.__delitem__(k1, k2)
        return self._tupOrderPermut(k1, k2)
    
    def setdefault(self, k1 : str, k2 : str, default = None) -> str | None:
        try:
            return self.__getitem__(k1, k2)
        except Exception:
            self.__setitem__((k1, k2), default)


class Aresta:
    def __init__(self, v1, v2, w = 0):
        if v1 < v2: 
            self.v1 = v1
            self.v2 = v2
        else:
            self.v1 = v2
            self.v2 = v1
        
        self.w = w
    
    def __hash__(self):
        return (self.v1, self.v2).__hash__()
    
    def __repr__(self):
        return (self.v1, self.v2).__repr__()
        return f'Edge ({self.v1}, {self.v2}), w = {self.w}'


def natural_ordering_pair(o1, o2):
    return (o1, o2) if o1 < o2 else (o2, o1)


def main():
    pd = PermutationDict({('A', 'B') : 3, ('B', 'C') : 2})
    print(pd)

    pd['D', 'C'] = 1
    print(pd)

    # print(pd.get('CD'))
    # print(pd.get('C-D'))
    print(pd.get('C', 'D'))


if __name__ == '__main__':
    main()