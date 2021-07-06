import numpy as np

def solve_khist(A, B, k):
    '''
    A is an np arrary with n elements
    B is an np arrary with n elements
    
    A = [a1, a2, ..., an]
    B = [b1, b2, ..., bn]
    sum(A) = 1, sum(B) = 1
    ai \in [0, 1], bi \in [0, 1]
    
    Let X = [x1, x2, ..., xk]
    xi \in Z+
    
    L = [1, x1+1, x1+x2+1, ..., sum(xi, i=1...k-1)+1]
    R = [x1, x1+x2, ..., sum(xi, i=1..k)]
    
    loss = sum(i=1...k) 
                (sum(j=Li...Ri) aj) * 
                (sum(j=Li...Ri) bj) 
    
    The goal is to find X such that loss is minized.
    
    @TODO: Add an entropy term to maximize the entropy
    '''
    
    # This algo is O(n^2 k).
    n = len(A)
    _dp = np.empty((n+1, k+1))
    _dp[:] = np.nan
    def dp(n, k):
        if not np.isnan(_dp[n,k]):
            return _dp[n,k]
        #print(n, k)
        
        if k == 1:
            _dp[n, k] = A[:n].sum() * B[:n].sum()
        elif n == k:
            _dp[n, k] = (A[:n] * B[:n]).sum()
        elif n < k:
            _dp[n, k] = np.inf
        else:
            losses = []
            for i in range(k-1, n):
                loss = dp(i, k-1) + A[i:n].sum() * B[i:n].sum()    
                losses.append(loss)
            _dp[n, k] = np.min(losses)
        return _dp[n, k]
    
    dp(n, k)
    current_n = n
    X = np.zeros(k, dtype=int)
    for j in range(k-1, 0, -1):
        losses = []
        for i in range(j, current_n):
            loss = dp(i, j) + A[i:current_n].sum() * B[i:current_n].sum()    
            losses.append(loss)
        next_n = np.argmin(losses) + j
        X[j] = current_n - next_n
        current_n = next_n
    X[0] = next_n
    return X, dp

def heuristic_khist(partitions):
    '''
    partitions are a list contains the tuples as this:
    (data density, parition fraction, partition set)
    
    we use the following variables for indicating 
    the elements in each tuple:
    (weight, volume, components)
    '''
    scored_partition = [
        ( weight, volume, components)
        for weight, volume, components
        in partitions
    ]
    
    