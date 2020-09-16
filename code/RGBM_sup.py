#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:12:16 2019
Basic code for reallocating geometric Brownian motion
@author: obp48
"""
import numpy as np
import matplotlib.pyplot as plt

#Parameters:
N=10
nsup=10
T=15
dt=.1
mu=-0.05
sigma=1.5
tau=0.2
#Derived parameters:
time_steps=(int(T/dt))
sdt=np.sqrt(dt)

#Initially everyone has wealth 1

msup = np.zeros((time_steps, 1))
vsup = np.zeros((time_steps, 1))

#Generate noise array 
xi=np.random.normal(loc=0, scale=1, size=(time_steps, N))

#Generate wealth trajectories
for i in range (1, nsup):
    x = np.zeros((time_steps, N))
    x[0][:] = np.random.rand(1, N)
    for t in range(1, time_steps):
        x[t] = x[t-1] * (1 + mu * dt + sigma * xi[t] * sdt) - tau * (x[t-1] - np.mean(x[t-1])) * dt
    msup = msup + ((1 / nsup) * np.mean(x,1))
    vsup = vsup + ((1 / nsup) * (np.var(x,1) + (np.mean(x,1))**2))

#plot trajectories
s=np.arange(0,(t+1))*dt
plt.semilogy(s, msup, s, vsup, s, np.exp(s * (2 * mu - 2 * tau + sigma**2)))
#plt.plot(s, np.mean(x), 'linewidth', 3)
plt.show()