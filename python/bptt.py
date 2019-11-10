#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/tf_playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.04.2019

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
exec(open("python/lib.py").read())

np.random.seed(123)

H = 10
P = 2
Q = 1

T_EPS = 0.01
T_STEPS = 1000
T_END = T_EPS * T_STEPS
THRESH = 1.0
GAMMA = 0.3 #Smoothing constant on pseudo-gradient 
LEARN_RATE = 1e-4
EPOCHS = 100

#TODO: Alpha should depend on T_EPS
alpha = 0.99
kappa = np.exp(-T_EPS)

mult = 100. / np.sqrt(H)
THETA_in = tf.Variable(mult * np.random.normal(size=[H,P]) * T_EPS)
THETA_rec = tf.Variable(mult * np.random.normal(size=[H,H]) * T_EPS)
THETA_out = tf.Variable(np.random.normal(size=[Q,H]))
trainable_variables = [THETA_in, THETA_rec, THETA_out]

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

target = np.ones([T_STEPS]) - 2 * np.array(np.floor(np.linspace(0,T_STEPS*T_EPS, T_STEPS)) % 2 == 0).astype(np.float64)

#Vs = np.zeros([H,T_STEPS+1])
V_last = tf.Variable(np.zeros([H,1]), dtype = tf.float64)
V = tf.Variable(np.zeros([H,1]), dtype = tf.float64)
spikevec = tf.Variable(np.zeros([H,1]), dtype = tf.float64)
spikevec_last = tf.Variable(np.zeros([H,1]), dtype = tf.float64)
y = np.zeros([Q, T_STEPS+1])

@tf.custom_gradient
def snl(V):
    """
    The threshold activation function typical in LIF models, but with a smoothed gradient.
    """
    def grad(dy):
        return dy * GAMMA * tf.keras.activations.relu(1. - tf.abs(THRESH-V)) / THRESH
    val = tf.cast(V >= THRESH, tf.float64)
    return val, grad

optim = tf.keras.optimizers.SGD(learning_rate = LEARN_RATE)

costs = np.empty([EPOCHS])
for epoch in tqdm(range(EPOCHS)):
    with tf.GradientTape() as gt:
        cost = 0
        for t in range(T_STEPS):
            # Decay Potential
            V = alpha * V_last

            # Integrate incoming spikes
            V += tf.matmul(THETA_rec, spikevec_last)

            # Integrate external stimuli
            inspikevec = np.zeros([P,1])
            for i in in_spikes[t]:
                inspikevec[i] = 1.
            V += tf.matmul(THETA_in, inspikevec)

            # Add outgoing spikes to the stack and reset potentials as necessary.
            spikevec = snl(V)
            spikes[t+1] = list(np.where(spikevec==1.)[0])
            V *= (1-spikevec)

            # Calculate output
            contrib = tf.matmul(THETA_out, spikevec)
            yout = contrib + kappa * y[:,t]
            y[:,t+1] = yout.numpy()

            # Determine error
            delta = yout - target[t]
            cost += tf.square(delta)

            # Update state
            V_last = V
            spikevec_last = spikevec

    grad = gt.gradient(cost, trainable_variables)
    grads_n_vars = [(grad[i], trainable_variables[i]) for i in range(len(grad))]
    optim.apply_gradients(grads_n_vars)

    costs[epoch] = cost.numpy()

plot_res(y, inlist, "forward.pdf")
