#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/mouse_adapt_bd.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.14.2019

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
exec(open("python/lib.py").read())
exec(open("python/mouse_lib.py").read())

np.random.seed(123)

# For the  mouse problem, groups of neurons fire to indicate the task parameters. We specify their properties here.
signal_duration = 1.0
spikes_per_signal = 4
neur_per_group = 1

R = 2
H = 100
P = 4*neur_per_group
Q = 1
mult = 200. / np.sqrt(H)
cue_time = 10.0

thresh = 1.0
t_eps = 0.01
tau_m = 0.2
tau_o = tau_m
tau_a = 2.
alpha = np.exp(-t_eps/tau_m)
kappa = np.exp(-t_eps/tau_o)
rho = np.exp(-t_eps/tau_a)
beta = 0.07

THETA_in = mult * np.abs(np.random.normal(size=[H,P]) * t_eps)
THETA_rec = mult * np.random.normal(size=[H,H]) * t_eps
THETA_rec -= np.diag(np.diag(THETA_rec)) # No recurrent connectivity.
THETA_out = np.random.normal(size=[Q,H])

t_steps = 2000
t_end = t_steps * t_eps
gamma = 0.3 # Called gamma in the article
mb_size = 20
learn_rate = 1e-8 / mb_size
epochs = 3600

nn_params = {'thresh' : thresh, 'alpha' : alpha, 'kappa' : kappa, 'beta' : beta,  'rho' : rho,  't_eps' : t_eps, 'gamma' : gamma}
trainable_params = {'in' : THETA_in, 'rec' : THETA_rec, 'out' : THETA_out}

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
    #TODO: It seems like we're not dividing the whole expression here, is that desired?
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

snn = NeuralNetwork(f = f, g = g, get_xt = get_xt, R = R, H = H, P = P, Q = Q, net_params = nn_params, trainable_params = trainable_params, eprop_funcs = eprop_funcs, learn_rate = learn_rate, update_every = mb_size)

costs = np.zeros(epochs)
decision = np.zeros(epochs, dtype = np.bool)
dirs = np.zeros(epochs, dtype = np.bool)

for epoch in tqdm(range(epochs)):

    ### Sample a new problem!
    coinflip, in_spikes, target = make_mouse_prob(t_eps, t_steps, signal_duration, spikes_per_signal, neur_per_group, cue_time = cue_time)
    dirs[epoch] = coinflip

    in_params = {'in_spikes' : in_spikes}

    ret = snn.run(t_steps = t_steps, ip = in_params, target = target, train = True, save_states = True, save_traces = True)

    costs[epoch] = ret['cost']
    decision[epoch] = (np.mean(ret['y'][:,int(cue_time / t_eps):]) > 0)

y = ret['y']
S = ret['S']

fig = plt.figure(figsize=[8,8])
plt.subplot(2,2,1)
plt.plot(S[0,:,:].T)
plt.title("Potential")
#plt.subplot(2,2,2)
#plt.plot(S[1,:,:].T)
#plt.title("Adaptive Threshold")
plt.subplot(2,2,2)
n_plot = 1000
toplot = np.random.choice(H*H,n_plot,replace = False)
plt.plot(ret['EPS'][:,:,1,:].reshape([H*H,t_steps]).T[:,toplot])
plt.title("Slow ET Componenet.")
plt.subplot(2,2,3)
plt.plot(y.flatten())
plt.title("Output")
plt.subplot(2,2,4)
plt.plot(np.log10(costs))
plt.title("Costs")
plt.savefig("temp.png")
plt.close()

print("Accuracy: %f"%(np.mean(decision==dirs)))

N = 100
ma = np.convolve(costs, np.ones((N,))/N, mode='valid')
fig = plt.figure()
plt.plot(ma)
plt.savefig("costs_ma.pdf")
plt.close()
