import scipy
import scipy.sparse.linalg as SparseLinalg
import scipy.linalg as Linalg


def solver(M, _k, _sigma=0., _tol=1e-7):

    #t_start = time()
    try:
        if scipy.__version__.split('.', 2)[1] == '10':
            #
            # eigsh sparse eigensolver, with sigma setting (in scipy>=0.10) 
            #
            eigval, eigvec = SparseLinalg.eigsh(M, k=_k, sigma=_sigma, tol=_tol)
        elif scipy.__version__.split('.', 2)[1] in ('8', '9'):
            #
            # eigsh sparse eigensolver, no sigma setting (in scipy<0.10) 
            # ask more then _k eigvecs, otherwise solver is unstable
            #
            eigval, eigvec = SparseLinalg.eigsh(M, k=_k*10, which='SM')
            #_, eigval, eigvec = SparseLinalg.svds(W, k=_k*10)
    except SparseLinalg.arpack.ArpackNoConvergence as excobj:
        print "ARPACK iteration did not converge"
        eigval, eigvec = excobj.eigenvalues, excobj.eigenvectors
        eigval = scipy.hstack((eigval, numpy.zeros(_k-eigval.shape[0])))
        eigvec = scipy.hstack((eigvec, numpy.zeros((n,_k-eigvec.shape[1]))))
        #
        # If eigval/eigvec pairs are not sorted on eigvals value
        #
        #ixEig = numpy.argsort(eigval)
        #eigval = eigval[ixEig]
        #eigvec = eigvec[:,ixEig]
        #print 'Eigen-values/vectors found in %.6fs' % (time()-t_start)
    return eigval, eigvec
