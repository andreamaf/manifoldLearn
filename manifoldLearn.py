import itertools
import numpy 
import scipy
import scipy.spatial.distance as Dist
import scipy.sparse as Sparse
import scipy.linalg as Linalg
from eigensolver import solver
from time import time


__name__ == "manifoldLearn"


class pf:
    """Perona and Freeman (1998)"""
    
    def __init__(self): pass
    # todo: SparseLinalg.eigen of AffinityMatrix 

class sm:
    """Shi and Malik (1997)"""
    
    def __init__(self): pass
    # Get laplacian and normalized Affinity Matrix
    # todo: SparseLinalg.eigen of Laplacian 

class slh:
    """Scott and Longuet-Higgins (1990)"""
    
    def __init__(self): pass
    # Get affinityMatrix then V then Q
    # todo: SparseLinalg.eigen of Q 

class bn:
    """Belkin and Niyogi (2001)"""
    
    def __init__(self): pass

class isomap:
    
    def __init__(self): pass

class lle:
    
    def __init__(self, k, d_out): self.k, self.d_out = k, d_out
    
    def __call__(self, X): return self.eigenmap(X)
    
    def distance_metric(self): return Dist.euclidean
   
    def eigenmap(self, X):
        
        n, d = X.shape
        W = Sparse.lil_matrix((n, n), dtype = 'float')
        self.getWNN(W, X) 
        W.tocsr()
        W -= Sparse.identity(n, format = 'csr')
        M = W.T * W #scipy.dot(W.T, W)
        
        eigval, eigvec = solver(M, self.d_out+1)
        
        eigval = eigval[1:self.d_out+1]
        eigvec = eigvec[:,1:self.d_out+1]
        
        eigvec /= eigval
        eigvec -= eigvec.mean(axis=0)
        eigvec /= eigvec.std(axis=0)
        #print eigvec,eigvec.shape,eigvec.mean(axis=0),eigvec.var(axis=0)
        return eigvec

    def getWNN(self, W, X, eps=.0001):
        n, d = X.shape
        k = self.k
        I = numpy.ones(k)
        diagIx = numpy.diag_indices(k)
        _Knn = Knn(X, k)
        for i, p in enumerate(X):
            ixNN = _Knn(p)
            NN_p = X[ixNN] - p
            Cx = scipy.dot(NN_p, NN_p.T)
            ##############################################################
            # If k>D or data not in general positions => Cx's conditioning
            ##############################################################
            if k>d: Cx[diagIx] += eps*numpy.trace(Cx)/k 
            Wx = Linalg.solve(Cx, I)
            Wx /= Wx.sum()
            W[i, ixNN] = Wx #[:]


def Knn(X, k):
    ##############################################################
    # TODO Actually a brute-force O(N^2*logN), 
    #      no KD-tree or asymptotically better solutions.  
    ##############################################################
    #def _heap(i, p):
    #    """Using a heap structure to get first K neighbours"""
    #    ds, c = [], 0
    #    for j, q in enumerate(X):
    #        if i == j: continue
    #        d = dist(p,q)
    #        if c < k:
    #            ds.append((1/d, j))
    #            c += 1
    #            if c == k: heapq.heapify(ds)
    #            continue
    #        heapq.heappushpop(ds, (1/d, j))
    #    return [heapq.heappop(ds)[1] for _ in xrange(k)][::-1]
    def _sortnn(p):
        return numpy.argsort(((X-p)**2).sum(axis=1))[1:k+1]
    return _sortnn   

def AffinityMatrix(X, graph = "K", weight = "H"):
    """Affinity Matrix can be built in different ways:
       see Belkin, Niyogi (2001) as reference.
       - graph  == "K" => K nearest neighborhoods 
       - graph  == "E" => Epsilon neighborhoods
       - weight == "H" => Heat kernel
       - weight == "S" => Simple-minded
    """
    
    pass

def Epsnn(X, sigma = 1., dist_threshold = 1.5,
          distance_metric = Dist.euclidean):
    ######################################################################
    # TODO Review the function to threshold distances in order to produce
    #      a sparse W in output, as with Knn function
    #      Brute-force O(n^2)  
    ######################################################################
    
    pass
    """
    dist, exp = distance_metric, numpy.exp
    n = X.shape()[0] ; W = numpy.zeros((n, n))
    it0, it1 = itertools.tee(X)
    for i, p in enumerate(it0):
        it1.next() ; it1, it2 = itertools.tee(it1)
        for j, q in enumerate(it2, start=i+1):
            d = dist((p,q))
            if d >= dist_threshold: continue 
            W[i,j] = W[j,i] = exp(-d/(2*sigma**2))
    return W        
    """

