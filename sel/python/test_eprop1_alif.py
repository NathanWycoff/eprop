#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/test_eprop1_lif.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.12.2020

## Test learning with e-prop 1 on a simple problem with LIF neurons.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())

np.random.seed(1234)

H = 5
P = 3
Q = 1
R = 2
mult = 100. / np.sqrt(H)

thresh = 1.0
t_eps = 0.01
tau_m = 0.2
tau_o = tau_m
tau_a = tau_m
alpha = np.exp(-t_eps/tau_m)
kappa = np.exp(-t_eps/tau_o)
rho = np.exp(-t_eps/tau_a)
beta = 0.03

THETA_in = mult * np.abs(np.random.normal(size=[H,P]) * t_eps)
THETA_rec = mult * np.random.normal(size=[H,H]) * t_eps
THETA_rec -= np.diag(np.diag(THETA_rec)) # No recurrent connectivity.
THETA_out = np.random.normal(size=[Q,H])

t_steps = 2000
gamma = 0.3 # Called gamma in the article
learn_rate = 5e-6
epochs = 100

# Make up some input spikes.
seq = np.linspace(t_eps * 10, t_eps * t_steps, t_steps / 10)
#spikes1 = seq[np.floor(seq) % 2 == 0]
#spikes2 = seq[np.floor(seq) % 2 != 0]
spikes1 = seq[seq <= 10.]
spikes2 = seq[seq > 10.]
# Add a random noise channel
n_spikes = len(spikes1)
spikes3 = np.sort(np.random.choice(t_steps, n_spikes, replace = False) * t_eps)
in_spikes = [[] for _ in range(t_steps+1)]
for spike in spikes1:
    in_spikes[int(spike/t_eps)] = [0]
for spike in spikes2:
    in_spikes[int(spike/t_eps)] = [1]
for spike in spikes3:
    in_spikes[int(spike/t_eps)] = [2]

# Create a random objective
#target = np.ones([t_steps]) - 2 * np.array(np.floor(np.linspace(0,t_steps*t_eps, t_steps)) % 2 == 0).astype(np.float64)
target = np.ones([t_steps]) - 2 * np.array(np.linspace(0,t_steps*t_eps,t_steps) <= 10).astype(np.float64)

nn_params = {'thresh' : thresh, 'alpha' : alpha, 'kappa' : kappa, 'beta' : beta,  'rho' : rho,  't_eps' : t_eps, 'gamma' : gamma}
trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}
in_params = {'in_spikes' : in_spikes}

def L(nn, yt, target):
    return (nn.trainable_params['out'] * (yt - target)).flatten()

# Give the jacobian of the state vector with respect to the trainable parameters.
# TODO: Think harder about the timing for this: we have st1 here but st in lib.py
# TODO: Think harder about naming the third param: it will sometimes be input spikes.
# First index - to neuron; second index - from neuron
def d_st_d_tp(nn, st1, zt1):
    wrt1 = np.tile(zt1, (nn.H,1))
    wrt2 = np.zeros_like(wrt1)
    ret = np.stack([wrt1, wrt2]).astype(np.float)
    ret = np.transpose(ret, [1,2,0])
    ret = np.expand_dims(ret, -1)
    return ret

# Gives the jacobian of the state vector at time t wrt the state vector at time t-1.
# Should return a tensor of shape RxRxH, element i,j,k represents the derivative of neuron state dimension i at time t wrt neuron state dimension j at time t-1 for neuron k.
def D(nn, st, st1):
    hs = 1. - np.abs((st[0,:] - nn.net_params['thresh']) / nn.net_params['thresh'])
    hs *= nn.net_params['gamma'] 
    hs *= (hs > 0).astype(np.float)

    diag1 = np.repeat(nn.net_params['alpha'], nn.H)
    diag2 = nn.net_params['rho'] - hs * nn.net_params['beta']
    superdiag = np.zeros([nn.H])
    subdiag = hs

    ret = np.stack([diag1, superdiag, subdiag, diag2], axis = -1).reshape([nn.H,2,2])
    return ret

# This is an H by R matrix to be returned, each element tells us the pseudo-derivative of the observable state of neuron h wrt to dimension r of the hidden state.
def dzds(nn, st):
    state1 = 1. - np.abs((st[0,:] - nn.net_params['thresh']) / nn.net_params['thresh'])
    state1 *= nn.net_params['gamma'] 
    state1 *= (state1 > 0).astype(np.float)
    state2 = (-1) * nn.net_params['beta'] * state1.copy()
    ret = np.stack([state1, state2])
    ret = ret.T
    return ret

eprop_funcs = {'L' : L, 'd_st_d_tp' : d_st_d_tp, 'D' : D, 'dzds' : dzds}

def f(nn, st):
    beta = nn.net_params['beta']
    thresh = nn.net_params['thresh']
    zt = (st[0,:] - beta * st[1,:]) > thresh
    return zt

def g(nn, st1, zt1, xt):
    # Initialize at previous potential.
    st = np.copy(st1)

    # Reset neurons that spiked.
    st *= (1-zt1)

    # Decay Potential.
    st[0,:] = alpha * st[0,:]

    # Update adaptive excess threshold
    st[1,:] = rho * st[1,:] + zt1

    # Integrate incoming spikes
    #st += zt1 @ nn.trainable_params['rec'].T
    st[0,:] += nn.trainable_params['rec'] @ zt1

    # Integrate external stimuli
    st[0,:] += nn.trainable_params['in'] @ xt

    ## Ensure nonnegativity
    #st *= st > 0

    return st

def get_xt(nn, ip, ti):
    xt = np.zeros([nn.P])
    xt[ip['in_spikes'][ti]] = 1
    return xt

snn = NeuralNetwork(f = f, g = g, get_xt = get_xt, R = R, H = H, P = P, Q = Q, net_params = nn_params, trainable_params = trainable_params, eprop_funcs = eprop_funcs, learn_rate = learn_rate)

ret = snn.run(t_steps = t_steps, ip = in_params, target = None, train = False, save_states = True)

costs = np.empty([epochs])
for epoch in tqdm(range(epochs)):
    ret = snn.run(t_steps = t_steps, ip = in_params, target = target, train = True, save_states = True)
    costs[epoch] = ret['cost']

y = ret['y']
S = ret['S']

fig = plt.figure(figsize=[8,8])
plt.subplot(2,2,1)
plt.plot(S[0,:,:].T)
plt.title("Potential")
plt.subplot(2,2,2)
plt.plot(S[1,:,:].T)
plt.title("Adaptive Threshold")
plt.subplot(2,2,3)
plt.plot(y.flatten())
plt.title("Output")
plt.subplot(2,2,4)
plt.plot(np.log10(costs))
plt.title("Costs")
plt.savefig("temp.pdf")
plt.close()

print("Col Norms: ")
print(np.sqrt(np.sum(np.square(snn.trainable_params['in']), axis = 0)))
