#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.29.2019

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())

np.random.seed(123)

H = 5
P = 2
Q = 1

T_EPS = 0.01
T_STEPS = 2000
T_END = T_EPS * T_STEPS
THRESH = 1.0
ETA = 0.3 # Called gamma in the article
LEARN_RATE = 1e-6
EPOCHS = 1000

#TODO: Alpha should depend on T_EPS
alpha = np.exp(-T_EPS)
kappa = np.exp(-T_EPS)

mult = 100. / np.sqrt(H)
THETA_in = mult * np.random.normal(size=[H,P]) * T_EPS
THETA_rec = mult * np.random.normal(size=[H,H]) * T_EPS
THETA_rec -= np.diag(np.diag(THETA_rec)) # No recurrent connectivity.
THETA_out = np.random.normal(size=[Q,H])
THETA_out_rand = np.random.normal(size=[Q,H]) #Random feedback weights, see how that works.

# A list of lists. The inner lists contain integers, giving the index of any neuron that has spiked.
spikes = [[] for _ in range(T_STEPS+1)]

# Make up some input spikes.
seq = np.linspace(T_EPS * 10, T_EPS * T_STEPS, T_STEPS / 10)
#spikes1 = seq[np.floor(seq) % 2 == 0]
#spikes2 = seq[np.floor(seq) % 2 != 0]
spikes1 = seq[seq <= 10.]
spikes2 = seq[seq > 10.]
in_spikes = [[] for _ in range(T_STEPS+1)]
for spike in spikes1:
    in_spikes[int(spike/T_EPS)] = [0]
for spike in spikes2:
    in_spikes[int(spike/T_EPS)] = [1]

# Create a random objective
#target = np.ones([T_STEPS]) - 2 * np.array(np.floor(np.linspace(0,T_STEPS*T_EPS, T_STEPS)) % 2 == 0).astype(np.float64)
target = np.ones([T_STEPS]) - 2 * np.array(np.linspace(0,T_STEPS*T_EPS,T_STEPS) <= 10).astype(np.float64)

inlist = [spikes1, spikes2]

Vs = np.zeros([H,T_STEPS+1])
y = np.zeros([Q, T_STEPS+1])

elig = np.zeros([H, T_STEPS+1])
zhat = np.zeros([H, T_STEPS+1])
dydg = np.zeros([H, T_STEPS+1]) # The derivative of the output at each time with respect to readout weight.

costs = np.zeros(EPOCHS)
for epoch in tqdm(range(EPOCHS)):
    GRADS_rec = np.zeros([H,H])
    GRADS_out = np.zeros([Q,H])

    # Forward Simulation!
    cost = 0
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

        # Record eligibility trace
        #TODO: Is this definitely t+1 on the next line?
        h = ETA * np.fmax(0, 1 - np.abs(Vs[:,t+1] - THRESH) / THRESH)
        zhat[:,t+1] = alpha * zhat[:,t] + spikevec
        #elig[:,t+1] = h * zhat[:,t+1]
        
        # Compute gradient contribution...
        # ...For hidden layers
        #TODO: Is this definitely t and not t-1 in a few lines?
        #TODO: Assumes Q = 1
        delta = y[:,t+1] - target[t]
        cost += np.square(delta)
        for h1 in range(H):
            for h2 in range(H):
                if h1 != h2:
                    dedz = delta * THETA_out_rand[0,h1]
                    GRADS_rec[h1,h2] += dedz * h[h1] * zhat[h2,t]

        # ...For output read weights
        #TODO: Is this definitely t and not t-1 in a few lines?
        #TODO: Assumes Q = 1
        dydg[:,t+1] = alpha * dydg[:,t] + spikevec.astype(np.float64)
        GRADS_out[0,:] += delta * dydg[:,t+1]

    # Apply grads
    THETA_rec = THETA_rec - LEARN_RATE * GRADS_rec
    THETA_out = THETA_out - LEARN_RATE * GRADS_out

    costs[epoch] = cost


# Check cost over time
fig = plt.figure()
plt.plot(np.log(costs[:epoch]))
plt.savefig("cost.pdf")

plot_res(y, inlist)

print(max([len(x) for x in spikes]))
