from typing import Iterable
import numpy as np

from permutation import natural_ordering_pair as nat

class TravellingSalesperson():
    def __init__(
        self,
        graph_filename : str,
        encoding : str = 'permutation',
    ):
        self.name, self.comment, self.dimension, self.vertexes, self.edges =\
            read_graph(graph_filename)
        
        self.encoding = encoding
        
        if self.encoding == 'permutation':
            self.fitness = self.dec_permutation
            self.get_random_pop = self.random_pop_permutation
        else: 
            raise ValueError(f'Encoding {encoding} not available.')
    
    
    def random_pop_permutation(self, size : int) -> np.ndarray:
        return np.asarray([np.random.permutation(len(self.vertexes)) 
                          for _ in range(size)])
    
    
    def dec_permutation(self, perm : np.ndarray):
        if len(perm) != self.dimension: print('fufu')
        return np.sum([self.edges.get(nat(v1, v2), 0)
                       for v1, v2 in zip(perm, perm[1:])]) +\
                           self.edges.get(nat(perm[0], perm[-1]))
    
    def __repr__(self):
        return f'''
Problema do Caixeiro Viajante - Nome: {self.name}
Comentários: {self.comment}
Dimensão: {self.dimension}

Vértices: 
{self.vertexes}

Arestas: 
{self.edges}
'''


def read_graph(filename : str):
    name = ''
    comment = ''
    dimension = 0
    edge_weight_type = 'EXPLICIT'
    edge_weight_format = 'FUNCTION'
    supported_formats = ('FUNCTION', 'FULL_MATRIX')
    edge_data_format = ''  # EDGE_LIST, ADJ_LIST
    node_coord_type = 'TWOD_COORDS'  # NO_COORDS, TWOD_COORDS, THREED_COORDS
    display_data_type = 'NO_DISPLAY'
    
    def euclidean(p1 : Iterable, p2 : Iterable) -> float:
        return np.sqrt(np.sum((n2 - n1) ** 2 for n1, n2 in zip(p1, p2)))
    
    def manhattan(p1 : Iterable, p2 : Iterable) -> int:
        return np.sum(np.abs(n2 - n1) for n1, n2 in zip(p1, p2))
    
    weight_func = euclidean
    edges = dict()
    out_vertexes = np.empty(0)
    
    with open(filename, 'r') as fr:
        ln = fr.readline()
        if 'NAME' in ln: name = ' '.join(ln.split()[1:])
        
        fr.readline()
        ln = fr.readline()
        if 'COMMENT' in ln: comment = ' '.join(ln.split()[1:])
        
        ln = fr.readline()
        if 'DIMENSION' in ln: dimension = int(ln.split()[1])
        
        for lnl in fr:
            if 'EDGE_WEIGHT_TYPE' in lnl: edge_weight_type = lnl.split()[1]
            if 'EDGE_WEIGHT_FORMAT' in lnl:
                if lnl.split()[1] not in supported_formats: 
                    raise ValueError('Edge weight format not supported')
                edge_weight_format = lnl.split()[1]
            if 'DISPLAY_DATA_TYPE' in lnl: display_data_type = lnl.split()[1]
            if 'NODE_COORD_TYPE' in lnl: node_coord_type = lnl.split()[1]
            if 'NODE_COORD_SECTION' in lnl or 'EDGE_WEIGHT_SECTION' in lnl: 
                ln = lnl
                break
        
        if edge_weight_format == 'FUNCTION':
            if edge_weight_type in ('EUC_2D', 'EUC_3D'):
                weight_func = euclidean
            elif edge_weight_type in ('MAN_2D', 'MAN_3D'):
                weight_func = manhattan
        
        def readvertexes(dim):
            vertexes = np.ndarray((dimension, dim))
            
            for lnl in fr:
                if lnl == 'EOF': break
                spl = lnl.split()
                v, p = int(spl[0]), tuple(float(x) for x in spl[1:])
                v -= 1
                for idx in range(dim):
                    vertexes[v][idx] = p[idx]
                
            return vertexes
            
        if 'NODE_COORD_SECTION' in ln:
            if node_coord_type == 'TWOD_COORDS':
                vertexes = readvertexes(2)
            
            if node_coord_type == 'THREED_COORDS':
                vertexes = readvertexes(3)
            
            for v1 in range(len(vertexes)):
                for v2 in range(v1 + 1, len(vertexes)):
                    edges[nat(v1, v2)] = weight_func(vertexes[v1], vertexes[v2])
            
        if 'EDGE_WEIGHT_SECTION' in ln:
            if edge_weight_format == 'FULL_MATRIX':
                for nline, lnl in enumerate(fr):
                    if 'EOF' in lnl or 'DISPLAY_DATA_SECTION' in lnl: 
                        ln = lnl
                        break
                    
                    for idx in range(nline):
                        cols = lnl.split()
                        edges[nat(idx, nline)] = float(cols[idx])

            if 'DISPLAY_DATA_SECTION' in ln:
                if display_data_type == 'TWOD_DISPLAY':
                    vertexes = readvertexes(2)
                        
                elif display_data_type == 'THREED_DISPLAY':
                    vertexes = readvertexes(3)
            
            out_vertexes = vertexes
    
    return name, comment, dimension, out_vertexes, edges


def main():
    tsp = TravellingSalesperson(r'instances/test.txt')
    print(tsp)
    # print(tsp.fitness(np.asarray([0, 1, 2])))
    # print(tsp.gen_random_pop(1))
    # print(tsp.fitness(*tsp.gen_random_pop(1)))

if __name__ == '__main__':
    main()