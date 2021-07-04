import numpy as np
import networkx as nx
import cvxpy as cp
from itertools import product

class Node():
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        self.c = None
        self.depth = None
        self.var = cp.Constant(0)
        
    def __str__(self):
        var_val = -1 if self.var.value is None else int(self.var.value)
        return f'{self.lb[0]:.3f}\n{self.ub[0]:.3f}\n{self.c}, {var_val:d}'
        
def range_count(A, lb, ub):
    return np.all((A > lb.reshape(1, -1)) & (A < ub.reshape(1, -1)), axis=1).sum()
        
def partition_at_all_dim(lb, ub):
    partition_list = []
    d = len(lb)
    for part in product(*[[0, 1] for i in range(d)]):
        part_lb = lb.copy()
        part_ub = ub.copy()
        for i in range(d):
            m = (lb[i] + ub[i])/2
            if part[i] == 0: 
                # take the lower half
                part_ub[i] = m 
            else:
                # take the upper half
                part_lb[i] = m
        partition_list.append((part_lb, part_ub))
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
    return A[np.all((A > lb.reshape(1, -1)) & (A < ub.reshape(1, -1)), axis=1)]

def get_leaves(G):
    leaves = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
    return leaves

class PrivTree():
    def __init__(self, A, eps):
        '''
        Assuming A is a n*d matrix normalized in [0, 1]
        '''
        self.A = A
        self.eps = eps
        self.non_part_score = 0
        
        n, d = self.A.shape
        beta = 2**d
        self.lam = (2*beta - 1) / (beta - 1) / eps
        self.theta = 0
        self.delta = self.lam * np.log(beta)
        print(self.lam, self.theta, self.delta)
        
    def partition(self):
        G = nx.DiGraph()
        n, d = self.A.shape
        lb = np.zeros(d)
        ub = np.ones(d)
        root = Node(lb, ub)
        epsilon_0 = self.eps
        current_depth = 0 
        root.depth = current_depth
        self._partition(self.A, G, root, current_depth)
        self.G = G
        self.root = root
        return G, root
    
    def _partition(self, A, G, node, current_depth):
        epsilon_0 = self.eps
        lb = node.lb
        ub = node.ub
        n, d = self.A.shape
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
            partition_list = partition_at_all_dim(lb, ub)
            #current_dim = current_depth % d
            #partition_list = partition_at_dim_i(lb, ub, current_dim)
            for partition in partition_list:
                lb, ub = partition
                child = Node(lb, ub)
                child.depth = current_depth+1
                G.add_edge(node, child)
                self._partition(get_sub_A(A, lb, ub), G, child, current_depth+1)
                node.var += child.var
                
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
        