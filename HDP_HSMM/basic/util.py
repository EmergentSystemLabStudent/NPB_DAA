from __future__ import division
import numpy as np
from numpy.random import random
na = np.newaxis
from numpy.core.umath_tests import inner1d
import scipy.special as special
import scipy.stats as stats
import scipy.linalg
import sys, time


#####################################################
#  Stats  #
#####################################################
### data abstraction
def getdatasize(data):
    if isinstance(data,np.ndarray):
        return data.shape[0]
    elif isinstance(data,list):
        return sum(getdatasize(d) for d in data)
    else:
        assert isinstance(data,int) or isinstance(data,float)
        return 1
def getdatadimension(data):
    if isinstance(data,np.ndarray):
        assert data.ndim > 1
        return data.shape[1]
    elif isinstance(data,list):
        assert len(data) > 0
        return getdatadimension(data[0])
    else:
        assert isinstance(data,int) or isinstance(data,float)
        return 1
def combinedata(datas):
    ret = []
    for data in datas:
        if isinstance(data,np.ndarray):
            ret.append(data)
        elif isinstance(data,list):
            ret.extend(data)
        else:
            assert isinstance(data,int) or isinstance(data,float)
            ret.append(np.atleast_1d(data))
    return ret
def combinedata_distribution(datas):
    ret = []
    for data in datas:
        if isinstance(data,np.ndarray):
            ret.append(data)
        elif isinstance(data,list):
            ret.extend(data)
        else:
            assert isinstance(data,int) or isinstance(data,float)
            ret.append(np.array(data,ndmin=1)) # ndmin=1 so that we can call .shape on it
    return ret
def flattendata(data):
    # data is either an array (possibly a maskedarray) or a list of arrays
    if isinstance(data,np.ndarray):
        return data
    elif isinstance(data,list) or isinstance(data,tuple):
        if any(isinstance(d,np.ma.MaskedArray) for d in data):
            return np.ma.concatenate(data).compressed()
        else:
            return np.concatenate(data)
    else:
        # handle unboxed case for convenience
        assert isinstance(data,int) or isinstance(data,float)
        return np.atleast_1d(data)
def flattendata_distribution(data):
    # data is either an array or a list of arrays
    if isinstance(data,np.ndarray):
        return data
    elif isinstance(data,list):
        if any(isinstance(d,np.ma.MaskedArray) for d in data):
            return np.ma.concatenate(data).compressed()
        else:
            return np.concatenate(data)
    else:
        assert isinstance(data,int) or isinstance(data,float)
        return np.array(data,ndmin=1)
### Sampling functions
def sample_discrete(distn,size=[],dtype=np.int32):
    'samples from a one-dimensional finite pmf'
    distn = np.atleast_1d(distn)
    assert (distn >=0).all() and distn.ndim == 1
    cumvals = np.cumsum(distn)
    return np.sum(np.array(random(size))[...,na] * cumvals[-1] > cumvals, axis=-1,dtype=dtype)
def sample_discrete_distribution(distn,size=[],dtype=np.int):
    'samples from a one-dimensional finite pmf'
    assert (distn >=0).all() and distn.ndim == 1
    cumvals = np.cumsum(distn)
    return np.sum(random(size)[...,na] * cumvals[-1] > cumvals, axis=-1,dtype=dtype)
def sample_discrete_from_log(p_log,axis=0,dtype=np.int32):
    'samples log probability array along specified axis'
    cumvals = np.exp(p_log - np.expand_dims(p_log.max(axis),axis)).cumsum(axis) # cumlogaddexp
    thesize = np.array(p_log.shape)
    thesize[axis] = 1
    randvals = random(size=thesize) * \
            np.reshape(cumvals[[slice(None) if i is not axis else -1
                for i in range(p_log.ndim)]],thesize)
    return np.sum(randvals > cumvals,axis=axis,dtype=dtype)
def sample_niw(mu,lmbda,kappa,nu):
    '''
    Returns a sample from the normal/inverse-wishart distribution, conjugate
    prior for (simultaneously) unknown mean and unknown covariance in a
    Gaussian likelihood model. Returns covariance.
    '''
    # code is based on Matlab's method
    # reference: p. 87 in Gelman's Bayesian Data Analysis
    # first sample Sigma ~ IW(lmbda,nu)
    lmbda = sample_invwishart(lmbda,nu)
    # then sample mu | Lambda ~ N(mu, Lambda/kappa)
    mu = np.random.multivariate_normal(mu,lmbda / kappa)
    return mu, lmbda
def sample_invwishart(lmbda,dof):
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    # TODO lowmem! memoize! dchud (eigen?)
    n = lmbda.shape[0]
    chol = np.linalg.cholesky(lmbda)
    if (dof <= 81+n) and (dof == np.round(dof)):
        x = np.random.randn(dof,n)
    else:
        x = np.diag(np.sqrt(stats.chi2.rvs(dof-np.arange(n))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return np.dot(T,T.T)
### Entropy
def invwishart_entropy(sigma,nu,chol=None):
    D = sigma.shape[0]
    chol = general.cholesky(sigma) if chol is None else chol
    Elogdetlmbda = special.digamma((nu-np.arange(D))/2).sum() + D*np.log(2) - 2*np.log(chol.diagonal()).sum()
    return invwishart_log_partitionfunction(sigma,nu,chol)-(nu-D-1)/2*Elogdetlmbda + nu*D/2
def invwishart_log_partitionfunction(sigma,nu,chol=None):
    D = sigma.shape[0]
    chol = general.cholesky(sigma) if chol is None else chol
    return -1*(nu*np.log(chol.diagonal()).sum() - (nu*D/2*np.log(2) + D*(D-1)/4*np.log(np.pi) \
            + special.gammaln((nu-np.arange(D))/2).sum()))
### Predictive
def multivariate_t_loglik(y,nu,mu,lmbda):
    # returns the log value
    d = len(mu)
    yc = np.array(y-mu,ndmin=2)
    ys, LT = general.solve_chofactor_system(lmbda,yc.T,overwrite_b=True)
    return scipy.special.gammaln((nu+d)/2.) - scipy.special.gammaln(nu/2.) \
            - (d/2.)*np.log(nu*np.pi) - np.log(LT.diagonal()).sum() \
            - (nu+d)/2.*np.log1p(1./nu*inner1d(ys.T,ys.T))


#####################################################
#  General  #
#####################################################
def rle(stateseq):
    pos, = np.where(np.diff(stateseq) != 0)
    pos = np.concatenate(([0],pos+1,[len(stateseq)]))
    return stateseq[pos[:-1]], np.diff(pos)
def top_eigenvector(A,niter=1000,force_iteration=False):
    '''
    assuming the LEFT invariant subspace of A corresponding to the LEFT
    eigenvalue of largest modulus has geometric multiplicity of 1 (trivial
    Jordan block), returns the vector at the intersection of that eigenspace and
    the simplex
    A should probably be a ROW-stochastic matrix
    probably uses power iteration
    '''
    n = A.shape[0]
    np.seterr(invalid='raise',divide='raise')
    if n <= 25 and not force_iteration:
        x = np.repeat(1./n,n)
        x = np.linalg.matrix_power(A.T,niter).dot(x)
        x /= x.sum()
        return x
    else:
        x1 = np.repeat(1./n,n)
        x2 = x1.copy()
        for itr in xrange(niter):
            np.dot(A.T,x1,out=x2)
            x2 /= x2.sum()
            x1,x2 = x2,x1
            if np.linalg.norm(x1-x2) < 1e-8:
                break
        return x1
def engine_global_namespace(f):
    # see IPython.parallel.util.interactive; it's copied here so as to avoid
    # extra imports/dependences elsewhere, and to provide a slightly clearer
    # name
    f.__module__ = '__main__'
    return f


#####################################################
#  Plot  #
#####################################################
def project_data(data,vecs):
    return np.dot(data,vecs.T)
def project_ellipsoid(ellipsoid,vecs):
    # vecs is a matrix whose columns are a subset of an orthonormal basis
    # ellipsoid is a pos def matrix
    return np.dot(vecs,np.dot(ellipsoid,vecs.T))
def plot_gaussian_2D(mu, lmbda, color='b', centermarker=True,label=''):
    '''
    Plots mean and cov ellipsoid into current axes. Must be 2D. lmbda is a covariance matrix.
    '''
    assert len(mu) == 2
    t = np.hstack([np.arange(0,2*np.pi,0.01),0])
    circle = np.vstack([np.sin(t),np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(lmbda),circle)
    if centermarker:
        plt.plot([mu[0]],[mu[1]],marker='D',color=color,markersize=4)
    plt.plot(ellipse[0,:] + mu[0], ellipse[1,:] + mu[1],linestyle='-',linewidth=2,color=color,label=label)
def plot_gaussian_projection(mu, lmbda, vecs, **kwargs):
    '''
    Plots a ndim gaussian projected onto 2D vecs, where vecs is a matrix whose two columns
    are the subset of some orthonomral basis (e.g. from PCA on samples).
    '''
    plot_gaussian_2D(project_data(mu,vecs),project_ellipsoid(lmbda,vecs),**kwargs)


#####################################################
#  Text  #
#####################################################
# time.clock() is cpu time of current process
# time.time() is wall time
# to see what this does, try
# for x in progprint_xrange(100):
#     time.sleep(0.01)
# TODO there are probably better progress bar libraries I could use
def progprint_xrange(*args,**kwargs):
    xr = xrange(*args)
    return progprint(xr,total=len(xr),**kwargs)
def progprint(iterator,total=None,perline=25,show_times=True):
    times = []
    idx = 0
    if total is not None:
        numdigits = len('%d' % total)
    for thing in iterator:
        prev_time = time.time()
        yield thing
        times.append(time.time() - prev_time)
        sys.stdout.write('.')
        if (idx+1) % perline == 0:
            if show_times:
                avgtime = np.mean(times)
                if total is not None:
                    sys.stdout.write(('  [ %%%dd/%%%dd, %%7.2fsec avg, %%7.2fsec ETA ]\n' % (numdigits,numdigits)) % (idx+1,total,avgtime,avgtime*(total-(idx+1))))
                else:
                    sys.stdout.write('  [ %d done, %7.2fsec avg ]\n' % (idx+1,avgtime))
            else:
                if total is not None:
                    sys.stdout.write(('  [ %%%dd/%%%dd ]\n' % (numdigits,numdigits) ) % (idx+1,total))
                else:
                    sys.stdout.write('  [ %d ]\n' % (idx+1))
        idx += 1
        sys.stdout.flush()
    print ''
    if show_times:
        print '%7.2fsec avg, %7.2fsec total\n' % (np.mean(times),np.sum(times))
