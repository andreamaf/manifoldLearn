try:
    import numpy 
    import scipy
    from scipy.spatial.distance import * #pdist, squareform
    import scipy.sparse as Sparse
    import scipy.linalg as Linalg
    import scipy.sparse.linalg as SparseLinalg
    import bisect
    import heapq
    import Queue
except ImportError: print "Impossible to import necessary libraries"


__name__ == "manifoldLearn"


#class isomap:
#    def __init__(self): pass

class pf:
    def __init__(self): pass
    # to do: SparseLinalg.eigen of AffinityMatrix 

class sm:
    def __init__(self): pass
    # Get laplacian and normalized Affinity Matrix
    # then do: SparseLinalg.eigen of Laplacian 

class slh:
    def __init__(self): pass
    # Get affinityMatrix then V then Q
    # to do: SparseLinalg.eigen of Q 

class lle:
    
    def __init__(self, k, d_out): self.k, self.d_out = k, d_out
    def __call__(self, X): return self.eigenmap(X)
    def distance_metric(self): return euclidean
    def eigenmap(self, X):

        W = self.getWNN(X) 
        W.tocsr() ; lenW = W.shape[0]
        W -= Sparse.identity(lenW, format = 'csr') ; W *= -1
        M = scipy.dot(W.T, W)
        ######################################################################
        # OPEN: If k=d+1 (exactly what we need), eigen* func. returns very often 'nan',
        #       so I ask more (useless) eivectors, then I get only the first [:,1:d+1].
        # OPEN: In some cases, small (under abs()) eigenvalues are < 0 !!!
        ######################################################################
        eigval, eigvec = SparseLinalg.eigen_symmetric(M, k=self.d_out*10, which='SA', tol=1e-08)
        #print 'lowest eigvec first el.', eigvec[:d+1,:3], eigval before', eigval[:d+1]
        if eigvec[0][0]<0: eigvec *= -1
        eigval = eigval[1:self.d_out+1]
        eigvec = eigvec[:,1:self.d_out+1]
        eigvec /= eigval
        eigvec -= eigvec.mean(axis=0)
        eigvec /= eigvec.std(axis=0)
        ######################################################################
        # OPEN: Do eigenvectors signs have to correspond to original data
        #       (as proposed in the tutorial) ???
        ######################################################################
        #print eigvec, eigvec.shape, eigvec.mean(axis=0), eigvec.var(axis=0)
        return eigvec

    def getWNN(self, X):
        k = self.k
        n, d = X.shape    
        getNN = Knn(X, k, self.distance_metric())
        W = Sparse.lil_matrix((n, n))
        def findWNN(i, p):
            ixNN = getNN[i] ; NN = X[ixNN] ; NN_p = p - NN
            Cx = scipy.dot(NN_p, NN_p.T)
            #CxInv = Linalg.inv(Cx) ; CxInvsum = CxInv.sum();
            #_Wx = CxInv.sum(axis=0) / CxInvsum
            ######################################################################
            # When k>D or data not in general positions => Cx's conditioning
            ######################################################################
            if k>d: Cx[range(k), range(k)] += 0.0001*numpy.trace(Cx)/k 
            Wx = Linalg.solve(Cx, numpy.ones(k))
            Wx /= Wx.sum()
            W[[i]*k, ixNN] = Wx[:]
        for _i, _p in enumerate(X): findWNN(_i,_p)
        return W

def affinityMatrix(X, *args, **kwargs): pass

def Knn(X, k, distance_metric = None):
    ######################################################################
    # OPEN Brute-force O(n^2) (no KD-tree)  
    ######################################################################
    dist = euclidean
    if distance_metric: dist = distance_metric
    def KnnHeap(i, p):
        """Heap based sol."""
        ds, c = [], 0
        for j, q in enumerate(X):
            if i == j: continue
            if c < k:
                ds.append((1/dist(p,q), j)); c += 1
                if c == k: heapq.heapify(ds)
                continue
            heapq.heappushpop(ds, (1/dist(p,q), j))
        return [j for _, j in ds]
    def KnnBinSearch(i, p):
        """Binary-search based solution"""
        ds = [(numpy.Inf, None)] * k
        for j, q in enumerate(X):
            if i == j: continue
            tup = (dist(p, q), j)    
            ix = bisect.bisect(ds, tup)
            if ix >= k: continue
            ###########################################
            # NB Following commands order is mandatory
            ###########################################
            ds[ix+1:] = ds[ix:k-1] ; ds[ix] = tup
        return [j for i, j in ds]  
    return numpy.array([KnnHeap(i,p) for i,p in enumerate(X)])   
