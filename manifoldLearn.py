try:
    import numpy 
    import scipy
    import scipy.spatial.distance as Dist
    import scipy.sparse as Sparse
    import scipy.linalg as Linalg
    import scipy.sparse.linalg as SparseLinalg
    import itertools
    from time import time
except ImportError: print "Impossible to import necessary libraries"


__name__ == "manifoldLearn"


class isomap:
    def __init__(self): pass

class pf:
    """Perona and Freeman (1998)"""
    def __init__(self): pass
    # to do: SparseLinalg.eigen of AffinityMatrix 

class sm:
    """Shi and Malik (1997)"""
    def __init__(self): pass
    # Get laplacian and normalized Affinity Matrix
    # then do: SparseLinalg.eigen of Laplacian 

class slh:
    """Scott and Longuet-Higgins (1990)"""
    def __init__(self): pass
    # Get affinityMatrix then V then Q
    # to do: SparseLinalg.eigen of Q 

class bn:
    """Belkin and Niyogi (2001)"""
    def __init__(self): pass

class lle:
    
    def __init__(self, k, d_out): self.k, self.d_out = k, d_out
    def __call__(self, X): return self.eigenmap(X)
    def distance_metric(self): return Dist.euclidean
    def eigenmap(self, X):
        
        n, d = X.shape
        W = Sparse.lil_matrix((n, n))
        self.getWNN(W, X) 
        W.tocsr()
        W -= Sparse.identity(n, format = 'csr')
        M = scipy.dot(W.T, W)
        ######################################################################
        # TODO If k=d+1 (exactly what we need), eigenfunc often returns 'nan',
        #      so I ask more eigenvectors, then taking only the first [:,1:d+1].
        #      Ssme small eigenvalues can be < 0 !
        ######################################################################
        #t_start = time()
        # eigval, eigvec = SparseLinalg.eigen_symmetric(M, k=self.d_out*10, which='SA')
        _, eigval, eigvec = SparseLinalg.svd(W, k=n-1)
        #_, eigval, eigvec = Linalg.svd(W.todense())
        ixEig = numpy.argsort(eigval)
        eigval = eigval[ixEig]
        eigvec = eigvec[ixEig].T
        # eigval, eigvec = Linalg.eigh(M.todense())
        #print time()-t_start
        eigval = eigval[1:self.d_out+1]
        eigvec = eigvec[:,1:self.d_out+1]
        eigvec /= eigval
        eigvec -= eigvec.mean(axis=0)
        eigvec /= eigvec.std(axis=0)
        ######################################################################
        # OPEN: Do eigenvectors signs have to correspond to original data ??
        ######################################################################
        #print eigvec, eigvec.shape, eigvec.mean(axis=0), eigvec.var(axis=0)
        return eigvec

    def getWNN(self, W, X):
        k = self.k
        rangek = range(k)
        n, d = X.shape
        funcKnn = Knn(X, k)
        for i, p in enumerate(X):
            ixNN = funcKnn(p)
            NN_p = X[ixNN] - p
            Cx = scipy.dot(NN_p, NN_p.T)
            ######################################################################
            # When k>D or data not in general positions => Cx's conditioning
            ######################################################################
            if k>d: Cx[rangek, rangek] += 0.0001*numpy.trace(Cx)/k 
            Wx = Linalg.solve(Cx, numpy.ones(k))
            Wx /= Wx.sum()
            W[[i]*k, ixNN] = Wx[:]


def AffinityMatrix(X, graph = "K", weight = "H"):
    """There are various ways to build such matrix:
       here I follow Belkin, Niyogi (2001), ref.2 in README:
       - graph =="K" => K nearest neighborhoods 
       - graph =="E" => Epsilon neighborhoods
       - weight=="H" => Heat kernel
       - weight=="S" => Simple-minded
    """
    ######################################################################
    # Graph
    ######################################################################
    # epsilon-n
    if graph =="K": W = Epsnn(X)
    elif graph =="E": W = Knn(X) 
    ######################################################################
    # weight
    ######################################################################
    if weight =="H": W = Epsnn(X)
    elif weight =="<S-Del>SE": W = Knn(X)
    return W

def Epsnn(X, sigma = 1., dist_threshold = 1.5,
          distance_metric = Dist.euclidean):
    ######################################################################
    # TODO Review the function to threshold distances in order to produce
    #      a sparse W in output, as with Knn function
    #      Brute-force O(n^2)  
    ######################################################################
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


def Knn(X, k):
    ######################################################################
    # TODO Brute-force O(n^2) (no KD-tree)  
    ######################################################################
    #def _heap(i, p):
    #    """Using a heap structure"""
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
