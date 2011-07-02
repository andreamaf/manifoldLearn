import scipy.stats as stats
from manifoldLearn import *
from time import time

try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    _plot = True
except ImportError:
    _plot = False
    pass


def S_shaped_data(samplesnr):
    """The S-shaped manifold, from Ref.6 in README"""

    angle = stats.uniform.rvs(loc = 0, scale = 3*scipy.pi/2, size = samplesnr)
    radius = 1.
    circle = numpy.array([radius*scipy.cos(angle),radius*(1+scipy.sin(angle))])
    circle = numpy.hstack((circle, -circle))
    z = stats.uniform.rvs(loc = -radius, scale = radius, size = 2*samplesnr)
    noise = stats.norm.rvs(loc = 0, scale = .01, size = (3, 2*samplesnr))
    S = numpy.vstack((circle, z))
    S += noise
    return S.T


def plot3D(X):

    if not(_plot): return
    fig1 = plt.figure(1)
    ax = fig1.gca(projection='3d')
    close = X[:,1] # numpy.linspace(0, 1, 2*samplesnr)
    ax.scatter(X[:,0], X[:,2], X[:,1], cmap = 'hsv', c=close)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


if __name__ == "__main__":
 
    # S-shaped manifold
    S = S_shaped_data(500)
        
    t_start = time()
    print "Time required: %fs" % (time() - t_start)
    
    if _plot:
        fig = plt.figure()
        plt.axis("equal")
        #plt.plot( [0],  [1], '.') 
        plt.show()
