import scipy.stats as stats
from manifoldLearn import *

try:
    import pkg_resources
    pkg_resources.require("matplotlib>=1.0.0")
except: pass
finally:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


def S_shaped_data(samplesnr):

    ############################
    # Building S-shaped dataset
    ############################
    angle = stats.uniform.rvs(loc = 0, scale = 3*scipy.pi/2, size = samplesnr)
    radius = 1.
    circle = numpy.array([radius*scipy.cos(angle),radius*(1+scipy.sin(angle))])
    circle = numpy.hstack((circle, -circle))
    z = stats.uniform.rvs(loc = -radius, scale = radius, size = 2*samplesnr)
    noise = stats.norm.rvs(loc = 0, scale = 0.05, size = (3, 2*samplesnr))
    S = numpy.vstack((circle, z))
    S += noise
    return S.T


if __name__ == "__main__":
 
    #X=numpy.array([(1,1),(0,0),(0,1),(5,6),(5,5),(6,6),(11,10),(9,9),(8,8)])
    #X = stats.norm.rvs(loc = 0, scale = 1, size = (100, 5))
    #X = numpy.array([(-20,-8),(-10,-1),(0,0.001),(10,1),(20,8),(11,-7),(12,21)])
    #eigenmap(X, 3, 3)
    
    S = S_shaped_data(100)
        
    """
    # 3D plot of S-shaped dataset
    fig1 = plt.figure(1)
    ax = fig1.gca(projection='3d')
    close = S[:,1] # numpy.linspace(0, 1, 2*samplesnr)
    ax.scatter(S[:,0], S[:,2], S[:,1], cmap = 'hsv', c=close)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    """
    
    lleS = lle(12, 2)(S).T
    
    fig = plt.figure()
    plt.axis("equal")
    plt.plot(lleS[0], lleS[1], '.') 
    plt.show()
