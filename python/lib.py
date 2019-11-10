#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.30.2019

def plot_res(y, inlist, path = "temp.pdf"):
    """
    y is a matrix giving one output in each row over time (time is represented by columns)
    inlist a list giving an array of firing times of the input neurons.
    """
    fig = plt.figure(figsize=[8,8])

    plt.subplot(2,1,1)
    plt.plot(y.T)
    plt.title("Output")

    plt.subplot(2,1,2)
    for p in range(P):
        plt.eventplot(inlist[p], color='r', linelengths = 0.5, lineoffsets=p+1)
    plt.title("Input Spikes")

    plt.savefig(path)
    plt.close()

def gen_problem(t_steps, t_eps = 0.01, min_seq_len = 1.0, hz = 10):
    """
    Generate a random problem from a simple class. Randomly have one of two inputs switch on for some amount of time. One input corresponds to a target of 1, and the other a taret of -1.

    T_EPS is the step size of our problem, and is assumed to be 0.01
    T_STEPS is the number of steps. Simulation time = T_EPS * T_STEPS
    min_seq_len is the length, in seconds, that has to ellapse before we allow the inputs to change.
    hz The frequency, in Hertz, of the signal, to be matched approximately.
    """

    assert t_eps == 0.01
    assert min_seq_len == 1.0
    assert hz == 10

    t_end = t_steps * t_eps

    seqs = int(np.ceil(T_STEPS * T_EPS))
    seqvals = np.random.choice(2,seqs)

    inlist = [[], []]
    for i,s in enumerate(seqvals):
        inlist[s].extend(np.arange(float(i), i+1.0, step = 1./hz))

    in_spikes = [[] for _ in range(T_STEPS+1)]
    for i,spiketrain in enumerate(inlist):
        for spike in spiketrain:
            in_spikes[int(spike/t_eps)] = [i]

    target = np.concatenate([((1-s) * -1. + s * 1.) * np.repeat(1.0, int(1/t_eps)) for s in seqvals])

    return in_spikes, inlist, target
