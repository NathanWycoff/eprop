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
