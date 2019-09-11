import numpy as np
import math
import pandas as pd
import scipy as sc
import scipy.stats as scs


# Calculates the Gini coefficient of array
def gini(array):
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)

    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


# Calculates the share of the top p percent in array
def topsh(array, p):
    array = np.sort(array)
    n = array.shape[0]
    s = np.sum(array)
    ind = math.floor(n * p / 100)
    return 100 * np.sum(array[(n - ind):(n + 1)]) / s


# Calculates the correlation between a and b
def mycorr(a, b):
    mone = np.sum((a - np.mean(a)) * (b - np.mean(b)))
    m1 = np.sqrt(np.sum(((a - np.mean(a)) ** 2)))
    m2 = np.sqrt(np.sum(((b - np.mean(b)) ** 2)))
    mechane = m1 * m2
    c = mone / mechane
    return c


# Calculates to rank correlation (Spearman's rho) between A and B
def rankcorr(A, B):

     N = A.shape[0]
     YA = np.transpose(np.argsort(np.transpose(A), -1))
     YB = np.transpose(np.argsort(np.transpose(B), -1))
     YAA = np.ones([N])
     YBB = np.ones([N])
     for i in range(N):
         YAA[YA[i]] = i
         YBB[YB[i]] = i

     return mycorr(YAA, YBB)

    #tmp = scs.spearmanr(A, B, axis=None)
    #return tmp[1]


# Calculates to correlation between log(A) and log(B)
def ranklog(A, B):
    return mycorr(np.log(A), np.log(B))


# Returns an anonymous growth incidence curve between A and B with fractiles defined by vector p
def gic(A, B, p):
    pN = p.shape[0]
    N = A.shape[0]
    AA = np.sort(A)
    BB = np.sort(B)
    inds = p * N
    G = np.zeros([pN - 1])
    for i in range(pN - 1):
        ind1 = math.floor(inds[i])
        ind2 = math.floor(inds[i + 1]) + 1
        G[i] = np.mean(BB[ind1:ind2]) / np.mean(AA[ind1:ind2])

    return G


# Returns an non-anonymous growth incidence curve between A and B with fractiles defined by vector p
def nonanongic(A, B, p):
    pN = p.shape[0]
    N = A.shape[0]
    inds = p * N
    G = np.zeros([pN - 1])
    for i in range(pN - 1):
        ind1 = math.floor(inds[i])
        ind2 = math.floor(inds[i + 1]) + 1
        G[i] = 100 * (np.mean(B[ind1:ind2]) / np.mean(A[ind1:ind2]) - 1)

    return G

# Returns an non-anonymous growth incidence curve between A and B with fractiles defined by vector p (mean growth in log)
def nagiclog(A, B, p):
    pN = p.shape[0]
    N = A.shape[0]
    inds = p * N
    G = np.zeros([pN - 1])
    for i in range(pN - 1):
        ind1 = math.floor(inds[i])
        ind2 = math.floor(inds[i + 1]) + 1
        G[i] = 100 * (np.mean(np.log(B[ind1:ind2]) - np.log(A[ind1:ind2])))

    return G

# Returns a absolute mobility by fractile with fractiles defined by vector p
def nonanongicabs(A, B, p):
    pN = p.shape[0]
    N = A.shape[0]
    inds = p * N
    G = np.zeros([pN - 1])
    for i in range(pN - 1):
        ind1 = math.floor(inds[i])
        ind2 = math.floor(inds[i + 1]) + 1
        G[i] = absmob(A[ind1:ind2], B[ind1:ind2])

    return G

# Calculates the absolute mobility between A and B
def absmob(A, B):
    X = B - A
    N = B.shape[0]
    #tmp = np.where(X > 0)
    tmp = 0

    #count = 0
    #for i in range(N):
    #    if X[i] > 0:
    #        count = count + 1

    if N == 0:
        return 0
    else:
        #return 100 * int(len(tmp)) / N
        return 0


# Imports a stata .dta file and exports into csv
def importexportstata(filename1, filename2):
    data = pd.io.stata.read_stata(filename1)
    data.to_csv(filename2)

# Similar to matlab find
def indices(a, func):
    dd = [i for (i, val) in enumerate(a) if func(val)]
    return dd


# Throws away NaNs, zeros and InFs from x, y and returns them in a, b (also age and hours, in this order)
def nonnaninfageagehoursnoz2(x, y, aa, hh):
    ind1 = np.where(~np.isnan(x))
    ind2 = np.where(~np.isnan(y))
    ind3 = np.intersect1d(ind1, ind2)
    x = x[ind3]
    y = y[ind3]
    aa = aa[ind3]
    hh = hh[ind3]
    ind1 = np.where(x > 0)
    ind2 = np.where(y > 0)
    ind3 = np.intersect1d(ind1, ind2)
    x = x[ind3]
    y = y[ind3]
    aa = aa[ind3]
    hh = hh[ind3]
    ind1 = np.where(~np.isinf(x))
    ind2 = np.where(~np.isinf(y))
    ind3 = np.intersect1d(ind1, ind2)
    a = x[ind3]
    b = y[ind3]
    ageage = aa[ind3]
    hou = hh[ind3]
    return a, b, ageage, hou

# creates two [0,1] uniformly distributed vectors of length n with a joint rank distribution that is a Plackett copula with parameter kappa
def plackett_rnd(kappa, n):

     U = np.random.rand(n)
     t = np.random.rand(n)

     a = t * (1 - t)
     b = kappa + a * ((kappa - 1) ** 2)
     c = 2 * a * (U * (kappa ** 2) + 1 - U) + kappa * (1 - 2 * a)
     d = np.sqrt(kappa) * np.sqrt(kappa + (4 * a * U * (1 - U)) * ((1 - kappa) ** 2))
     V = (c - ((1 - 2 * t) * d)) / (2 * b)
     return U, V

# take two vectors s1 and s2 and returns the same vectors with a joint rank distribution that is a Plackett copula with parameter theta
def couple_vecs_plackett(s1, s2, theta):

    N = len(s1)
    U, V = plackett_rnd(theta, N)

    ssss1 = np.floor(scs.rankdata(U))
    ssss1 = ssss1.astype(int)-1
    ssss2 = np.floor(scs.rankdata(V))
    ssss2 = ssss2.astype(int)-1

    indsss1 = np.random.permutation(N)
    tmp1 = np.sort(s1[indsss1])

    indsss2 = np.random.permutation(N)
    tmp2 = np.sort(s2[indsss2])

    w1 = tmp1[ssss1]
    w2 = tmp2[ssss2]

    tempind1 = np.argsort(tmp1[ssss1])
    w1 = np.sort(tmp1[ssss1])
    w2 = w2[tempind1]

    return w1, w2

# true if a is sorted, falst otherwise
def issorted(a):

    for i in range(a.size-1):
        if a[i+1] < a[i]:
            return False
    return True


def mylog(x):

    return np.multiply(np.sign(x), np.log(np.abs(x)))