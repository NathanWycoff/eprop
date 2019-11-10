#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/tf_playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.04.2019

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
exec(open("python/lib.py").read())

np.random.seed(123)

H = 10 # Neurons in SNN
Ht = 10 # Neurons in TEM
P = 2
Q = 1

T_EPS = 0.01
T_STEPS = 1000
T_END = T_EPS * T_STEPS
THRESH = 1.0
GAMMA = 0.3 #Smoothing constant on pseudo-gradient 
LEARN_RATE = 1e-3 #Learn rate for TEM; SNN learning rate is learnt.
EPOCHS = 3
N_PROBS = 2 # The number of problems to sample at each outer weight update (i.e. number of inner loops per outer loop).

#TODO: Alpha should depend on T_EPS
alpha = 0.99
kappa = np.exp(-T_EPS)

# Heuristic which seems to result in a reasonable initialization.
mult = 100. / np.sqrt(H)

# SNN weights 
iTHETA_in = tf.Variable(mult * np.random.normal(size=[H,P]) * T_EPS)
iTHETA_rec = tf.Variable(mult * np.random.normal(size=[H,H]) * T_EPS)
iTHETA_out = tf.Variable(np.random.normal(size=[Q,H]))

# TEM weights
PSI_in = tf.Variable(mult * np.random.normal(size=[Ht,P]) * T_EPS)
PSI_rec = tf.Variable(mult * np.random.normal(size=[Ht,Ht]) * T_EPS)
PSI_out = tf.Variable(np.random.normal(size=[H,Ht]))

trainable_variables = [PSI_in, PSI_rec, PSI_out]

# A list of lists. The inner lists contain integers, giving the index of any neuron that has spiked.
spikes = [[] for _ in range(T_STEPS+1)]
spikes_tem = [[] for _ in range(T_STEPS+1)]

# Make up some input spikes.

#Vs = np.zeros([H,T_STEPS+1])
# The potential vector for our SNN
V_last = tf.Variable(np.zeros([H,1]), dtype = tf.float64)
V = tf.Variable(np.zeros([H,1]), dtype = tf.float64)

# Spikevec is denoted a z in the paper.
spikevec = tf.Variable(np.zeros([H,1]), dtype = tf.float64)
spikevec_last = tf.Variable(np.zeros([H,1]), dtype = tf.float64)

# The potential vector for the TEM
U_last = tf.Variable(np.zeros([Ht,1]), dtype = tf.float64)
U = tf.Variable(np.zeros([Ht,1]), dtype = tf.float64)

Lout = tf.Variable(np.zeros([H,1]), dtype = tf.float64)
Lout_last = tf.Variable(np.zeros([H,1]), dtype = tf.float64)

# spikevec is denoted as zeta in the paper.
spikevec_tem = tf.Variable(np.zeros([Ht,1]), dtype = tf.float64)
spikevec_tem_last = tf.Variable(np.zeros([Ht,1]), dtype = tf.float64)

y = np.zeros([Q, T_STEPS+1]) # the network's output.
L = np.zeros([H, T_STEPS+1]) # the learning signal.

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

zhat = tf.Variable(np.zeros([H,1]), dtype = tf.float64)
zhat_last = tf.Variable(np.zeros([H,1]), dtype = tf.float64)

elig = np.zeros([H, T_STEPS+1])
dydg = np.zeros([H, T_STEPS+1]) # The derivative of the output at each time with respect to readout weight.

costs = np.empty([EPOCHS])
for epoch in tqdm(range(EPOCHS)):
    with tf.GradientTape() as gt:
        cost = 0
        for p in range(N_PROBS):
            #Sample a random problem:
            in_spikes, inlist, target = gen_problem(T_STEPS)

            # Initilize SNN weights.
            THETA_in = iTHETA_in
            THETA_rec = iTHETA_rec 
            THETA_out = iTHETA_out 

            #### Run the network once with the TEM online.
            GRADS_rec = tf.zeros([H,H], dtype = tf.float64)
            for t in range(T_STEPS):
                # Decay Potential
                V = alpha * V_last
                U = alpha * U_last

                # Integrate incoming spikes
                V += tf.matmul(THETA_rec, spikevec_last)
                U += tf.matmul(PSI_rec, spikevec_tem_last)

                # Integrate external stimuli
                inspikevec = np.zeros([P,1])
                for i in in_spikes[t]:
                    inspikevec[i] = 1.
                V += tf.matmul(THETA_in, inspikevec)
                U += tf.matmul(PSI_in, inspikevec)

                # Add outgoing spikes to the stack and reset potentials as necessary.
                spikevec = snl(V)
                spikes[t+1] = list(np.where(spikevec==1.)[0])
                V *= (1-spikevec)
                spikevec_tem = snl(U)
                spikes_tem[t+1] = list(np.where(spikevec_tem==1.)[0])
                U *= (1-spikevec_tem)

                # Calculate output
                contrib = tf.matmul(THETA_out, spikevec)
                yout = contrib + kappa * y[:,t]
                y[:,t+1] = yout.numpy()

                # Calculate learning signal
                contrib_tem = tf.matmul(PSI_out, spikevec_tem)
                Lout = contrib_tem + kappa * Lout_last
                L[:,t+1] = Lout.numpy().flatten()

                # Record eligibility trace
                #TODO: Is this definitely t+1 on the next line?
                h = GAMMA * np.fmax(0, 1 - np.abs(V - THRESH) / THRESH)
                zhat = alpha * zhat_last + spikevec
                #elig[:,t+1] = h * zhat[:,t+1]

                # Compute gradient contribution...
                # ...For hidden layers
                #TODO: Is this definitely t and not t-1 in a few lines?
                #TODO: Assumes Q = 1
                GRADS_rec += tf.transpose(Lout) * h * tf.transpose(zhat)

                # Update state
                V_last = V
                U_last = U
                Lout_last = Lout
                spikevec_last = spikevec
                spikevec_tem_last = spikevec_tem
                zhat_last = zhat

            THETA_rec = THETA_rec - LEARN_RATE * GRADS_rec

            #### Rerun in "test" mode: TEM deactivated, record loss
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

                # Update state
                V_last = V
                spikevec_last = spikevec

                # Determine error
                delta = yout - target[t]
                cost += tf.square(delta) / float(T_STEPS*N_PROBS)

    grad = gt.gradient(cost, trainable_variables)
    grads_n_vars = [(grad[i], trainable_variables[i]) for i in range(len(grad))]
    optim.apply_gradients(grads_n_vars)

    costs[epoch] = cost.numpy()

fig = plt.figure()
plt.plot(costs)
plt.savefig('eprop2_costs.pdf')
plt.close()
