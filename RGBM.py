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
T=100
dt=.1
mu=0.0
sigma=0.15
tau=-.03
#Derived parameters:
time_steps=(int(T/dt))
sdt=np.sqrt(dt)

#Initially everyone has wealth 1
x = np.zeros((time_steps, N)) 
x[0][:]=1

#Generate noise array 
xi=np.random.normal(loc=0, scale=1, size=(time_steps,N))

#Generate wealth trajectories
for t in range(1,time_steps):
    x[t]=x[t-1]*(1+mu*dt+sigma*xi[t]*sdt)-tau*(x[t-1]-np.mean(x[t-1]))*dt

#plot trajectories
s=np.arange(0,(t+1))*dt
plt.plot(s,x)