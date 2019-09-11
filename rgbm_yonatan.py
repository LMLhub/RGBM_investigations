import numpy as np
import time
import pysal
import matplotlib.pyplot as plt
import funi

start = time.time()

#Final time
T = 500

#number of time steps
nT = 1000

# population size
N = 100000

# time vector
tt = np.linspace(0, T, nT + 1)

dt = tt[1] - tt[0]

X = np.ones([N, nT + 1])
X[:, 0] = np.transpose(2 * (np.random.rand(N) - 0.5))
noise = np.random.randn(N, nT)
#print(np.min(X[:, 0]))
#print(np.max(X[:, 0]))

#scalars of interest
means = np.zeros([nT + 1])
maxs = np.zeros([nT + 1])
mins = np.zeros([nT + 1])
stds = np.zeros([nT + 1])
meds = np.zeros([nT + 1])
ranki = np.zeros([nT + 1])
vari_resc = np.zeros([nT + 1])
gini_resc = np.zeros([nT + 1])
numpos = np.zeros([nT + 1])

# parameters
mu = 0.021
sigma = np.sqrt(0.15)
tau = -1 / 2

plt.close('all')
#plt.yscale('log')

indind = np.where(X[:, 0] > 0)
indind = indind[0]
# number of positive wealths
numpos[0] = len(indind)

for i in range(nT):
    #print(np.min(X[:, i]))
    #print(np.max(X[:, i]))
    X[:, i + 1] = X[:, i] * (1 + mu * dt - tau * dt) + np.mean(X[:, i]) * tau * dt + np.sqrt(dt) * np.multiply(X[:, i], sigma * noise[:, i])

    #X[:, i + 1] = X[:, i] * (1 + mu * dt - tau * dt) + np.mean(X[:, i]) * tau * dt
    ranki[i] = funi.rankcorr(X[:, i], X[:, i + 1])
    mins[i] = np.min(X[:, i+1])
    maxs[i] = np.max(X[:, i + 1])
    means[i] = np.mean(X[:, i + 1]) / np.exp(tt[i + 1] * (mu - tau))
    stds[i] = np.std((X[:, i + 1] / np.exp(tt[i + 1] * (mu - tau))))
    meds[i] = np.median(X[:, i + 1])
    indind = np.where(X[:, i + 1] > 0)
    indind = indind[0]
    numpos[i + 1] = len(indind)
    #vari_resc[i] = np.var(X[indind, i + 1] / np.exp(tt[i + 1] * (mu - tau)))
    #gini_resc[i] = funi.gini(X[indind, i + 1] / np.exp(tt[i + 1] * (mu - tau)))
    #A = np.histogram(X[:, i + 1])
    #B = A[1]
    #plt.plot(B[1:], A[0])
    #if np.mod(tt[i], 10) == 0:
        #indind = np.where(X[:, i + 1] > 0)
        #indind = indind[0]
        #print(indind)
        #plt.hist((X[:, i + 1] / np.exp(tt[i + 1] * (mu - tau))), 50)
        #plt.pause(0.5)

#vari_resc[nT] = np.var(X[:, nT] / np.exp(tt[nT] * (mu - tau)))

#print(means)
#plt.plot(tt, funi.mylog(np.transpose(X)), 'k')
#plt.plot(tt, gini_resc)
#plt.plot(tt, funi.mylog(meds))
#plt.plot(tt, funi.mylog(np.exp(tt * mu)), 'b')
#plt.plot(tt, funi.mylog(np.exp(tt * (mu - sigma**2/2))), 'b')
#plt.plot(tt, funi.mylog(np.exp(tt * (mu - tau))), 'g')
#plt.plot(tt[0:nT], gini_resc[0:nT])

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('Fraction of positive wealths (%)', color=color)
ax1.plot(tt, 100 * numpos / N, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Rescaled sample mean)', color=color)  # we already handled the x-label with ax1
ax2.plot(tt, means, color=color)
#ax2.plot(tt, funi.mylog(means), color=color)
#ax2.plot(tt, ((mu -(sigma**2)/2 - tau) * tt), color='k')
#ax2.plot(tt, ((mu - tau) * tt), color='g')
#ax2.plot(tt, ((mu ) * tt), color='m')
ax2.tick_params(axis='y', labelcolor=color)
fig.set_size_inches(10.67, 6)

plt.show()
#plt.savefig('tmp.pdf')

end = time.time()
print(end - start)