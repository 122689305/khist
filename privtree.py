import numpy as np
import networkx as nx
import cvxpy as cp
from itertools import product
import math

class Node():
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        self.c = None
        self.depth = None
        self.var = cp.Constant(0)
        
    def __str__(self):
        var = self.var
        var_val = -1 if var.value is None else int(var.value)
        #return str(var)
        return f'Depth:{self.depth}\nRange:[{self.lb[0]:.0f}, {self.ub[0]:.0f}]\nCount:{self.c}, Est:{var_val:d}'
        
def range_count(A, lb, ub):
    # We treat [lb, ub] as closed intervals
    # To simulate open intervals, please make tiny shift of the boundaries
    c = np.all((A >= lb.reshape(1, -1)) & (A <= ub.reshape(1, -1)), axis=1).sum()
    return c

def int_splitter(lb, ub):
    '''
    lb and ub are scalars
    '''
    assert lb <= ub, 'lower bound should be <= upper bound'
    if lb == ub:
        # no split
        return [(lb, ub)]
    else:
        # split into halves
        # if odd, push middle to the upper side
        m = (lb + ub) // 2 
        return [(lb, m), (m+1, ub)]
    
def real_splitter(lb, ub):
    '''
    lb and ub are scalars
    '''
    assert lb < ub, 'lower bound should be < upper bound'
    m = (lb + ub) / 2
    return [(lb, m), (m + np.nextafter(0, 1), ub)]

def load_splitters(column_types):
    splitters = [eval(f'{ctype}_splitter') for ctype in column_types]
    return splitters
        
def partition_at_all_dim(lb, ub, splitters=None):
    partition_list = []
    d = len(lb)
    if splitters == None:
        splitters = [real_splitter]*d
    producted_partitions = product(*[splitter(_lb, _ub) 
                                     for _lb, _ub, splitter 
                                     in zip(lb, ub, splitters)])
    partition_list = [tuple(map(lambda arr: np.array(arr), 
                                zip(*producted_partition)))
                      for producted_partition 
                      in producted_partitions]
    return partition_list

def partition_at_dim_i(lb, ub, i):
    m = (lb[i] + ub[i])/2

    part_l_lb = lb.copy()
    part_l_ub = ub.copy()
    part_l_ub[i] = m

    part_r_lb = lb.copy()
    part_r_ub = ub.copy()
    part_r_lb[i] = m  
    
    return ((part_l_lb, part_l_ub), (part_r_lb, part_r_ub))

def get_sub_A(A, lb, ub):
    return A[np.all((A >= lb.reshape(1, -1)) & (A <= ub.reshape(1, -1)), axis=1)]

def get_leaves(G):
    leaves = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
    return leaves

class PrivTree():
    def __init__(self, A, eps, column_bounds, column_types=None):
        '''
        Assuming A is a n*d matrix normalized in [0, 1]
        '''
        assert A.shape[1] == len(column_bounds) == len(column_types)
        self.A = A
        self.eps = eps
        self.non_part_score = 0
        self.column_bounds = column_bounds
        self.splitters = load_splitters(column_types)
        
        n, d = self.A.shape
        beta = 2**d
        self.lam = (2*beta - 1) / (beta - 1) / eps
        self.theta = 0
        self.delta = self.lam * np.log(beta)
        print(self.lam, self.theta, self.delta)
        
    def partition(self):
        G = nx.DiGraph()
        n, d = self.A.shape
        column_bounds = self.column_bounds
        lb = np.array([cbound[0] for cbound in column_bounds])
        ub = np.array([cbound[1] for cbound in column_bounds])
        root = Node(lb, ub)
        epsilon_0 = self.eps
        current_depth = 0 
        root.depth = current_depth
        self._partition(self.A, G, root)
        self.G = G
        self.root = root
        return G, root
    
    def _partition(self, A, G, root):
        unvisited = [(root, None, 0)]
        epsilon_0 = self.eps
        n, d = self.A.shape
        while len(unvisited) > 0:
            node, parent, num_visited = unvisited.pop()
            if num_visited == 0:
                unvisited.append((node, parent, 1))
                current_depth = node.depth
                lb = node.lb
                ub = node.ub
                splitters = self.splitters
                c = range_count(A, lb, ub)
                node.c = c
                c = c - current_depth * self.delta
                c = max(c, self.theta - self.delta)
                b = c + np.random.laplace(scale=self.lam)
                if b <= self.theta:
                    # no split
                    #print('no split')
                    node.var = cp.Variable()
                    pass
                else:
                    # partition the current node
                    #print(c, current_depth, -np.log((ub[0]-lb[0]))/np.log(2), b, lb, ub)
                    partition_list = partition_at_all_dim(lb, ub, splitters)
                    #current_dim = current_depth % d
                    #partition_list = partition_at_dim_i(lb, ub, current_dim)
                    for partition in partition_list:
                        lb, ub = partition
                        child = Node(lb, ub)
                        child.depth = current_depth+1
                        unvisited.append((child, node, 0))
            elif num_visited == 1:
                if parent is not None:
                    parent.var += node.var
                    G.add_edge(parent, node)
                
    def mle(self, n = None):
        G = self.G
        root = self.root
        theta = self.theta
        delta = self.delta
        lam = self.lam
        constraints = []
        objs = []
        for x in G.nodes():
            if G.out_degree(x)==0 and G.in_degree(x)==1:
                # Leaf Node
                # Add constraints as x - depth * delta <= theta - delta
                constraints.append(x.var - x.depth * delta <= theta - delta)
                constraints.append(x.var >= 0)
            else:
                # Non-Leaf Node
                # Add constraints as x - depth * delta >= theta
                # Add objective as log(1 - 0.5 * exp((theta-x)/lam))
                constraints.append(x.var - x.depth * delta >= theta)
                objs.append(cp.log(1-0.5*cp.exp((theta-x.var)/lam)))
        if n is not None:
            constraints.append(root.var <= n)
            
        obj = cp.Maximize(sum(objs))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return self.G
        
    def naive_est(self):
        G = self.G
        root = self.root
        theta = self.theta
        delta = self.delta
        lam = self.lam
        for x in G.nodes():
            if G.out_degree(x)==0 and G.in_degree(x)==1:
                x.var.value = max(0, theta + (x.depth - 1) * delta)
        return self.G
        