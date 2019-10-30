#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.29.2019

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

H = 10
P = 2
Q = 1

T_EPS = 0.01
T_STEPS = 1000
T_END = T_EPS * T_STEPS
THRESH = 1.0

#TODO: Alpha should depend on T_EPS
alpha = 0.99
kappa = np.exp(-T_EPS)

mult = 100. / np.sqrt(H)
THETA_in = mult * np.random.normal(size=[H,P]) * T_EPS
THETA_rec = mult * np.random.normal(size=[H,H]) * T_EPS
THETA_out = np.random.normal(size=[Q,H])

# A list of lists. The inner lists contain integers, giving the index of any neuron that has spiked.
spikes = [[] for _ in range(T_STEPS+1)]

# Make up some input spikes.
seq = np.linspace(T_EPS * 10, T_EPS * T_STEPS, T_STEPS / 10)
spikes1 = seq[np.floor(seq) % 2 == 0]
spikes2 = seq[np.floor(seq) % 2 == 1]
in_spikes = [[] for _ in range(T_STEPS+1)]
for spike in spikes1:
    in_spikes[int(spike/T_EPS)] = [0]
for spike in spikes2:
    in_spikes[int(spike/T_EPS)] = [1]

inlist = [spikes1, spikes2]

Vs = np.zeros([H,T_STEPS+1])
y = np.zeros([Q, T_STEPS+1])

# Forward Simulation!

for t in range(T_STEPS):
    # Decay Potential
    Vs[:,t+1] = alpha * Vs[:,t]

    # Integrate incoming spikes
    Vs[:,t+1] += np.sum(THETA_rec[:,spikes[t]], axis = 1)

    # Integrate external stimuli
    Vs[:,t+1] += np.sum(THETA_in[:,in_spikes[t]], axis = 1)

    # Add outgoing spikes to the stack and reset potentials as necessary.
    spikevec = Vs[:,t+1] >= THRESH
    spikes[t+1] = list(np.where(spikevec)[0])
    Vs[:,t+1] *= (1-spikevec)

    # Record network output
    y[:,t+1] = np.sum(THETA_out[:,spikes[t]]) + kappa * y[:,t]

# Get a gradient!
for t in range()


print(max([len(x) for x in spikes]))
