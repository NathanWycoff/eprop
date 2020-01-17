#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/mouse_lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.16.2020

def make_mouse_prob(t_eps, t_steps, signal_duration, spikes_per_signal, neur_per_group, cue_time):
    t_end = t_eps * t_steps

    # If it comes up heads (1), we want the mouse to go left, and otherwise right.
    coinflip = bool(np.random.choice(2))
    #coinflip = False

    # Signal for the correct side three times for a duration of 1 second, then 1 second off.
    major_times = [1.0, 5.0, 9.0]
    # Signal for the correct side twice, also with 1 second duration and 1 second in between.
    minor_times = [3.0, 7.0]

    # Sample major firing times.
    major_spikes = np.concatenate([np.random.uniform(low = mt, high = mt + signal_duration, size = [spikes_per_signal, neur_per_group]) for mt in major_times], axis = 0)
    minor_spikes = np.concatenate([np.random.uniform(low = mt, high = mt + signal_duration, size = [spikes_per_signal, neur_per_group]) for mt in minor_times], axis = 0)

    if coinflip:
        left_spikes = major_spikes
        right_spikes = minor_spikes
    else:
        right_spikes = major_spikes
        left_spikes = minor_spikes

    # Sample noise channel
    n_noise_spikes = int(t_end * spikes_per_signal)
    noise_spikes = np.random.uniform(low = 0., high = t_end, size = [n_noise_spikes, neur_per_group])

    # Sample cue channel
    cue_spikes = np.random.uniform(low = cue_time, high = cue_time + signal_duration, size = [spikes_per_signal, neur_per_group])
    inlist = [list(left_spikes.T), list(right_spikes.T), list(noise_spikes.T), list(cue_spikes.T)]

    # Convert to the format I've been using to do the actual simulation.
    inlist_flat = [item for sublist in inlist for item in sublist]
    in_spikes = [[] for _ in range(t_steps+1)]
    for i, spikesi in enumerate(inlist_flat):
        for spike in spikesi:
            in_spikes[int(spike/t_eps)].append(i)

    # Target is zero except for after the cue.
    target = np.zeros(t_steps)
    if coinflip:
        target[int(cue_time / t_eps):] = 10
    else:
        target[int(cue_time / t_eps):] = -10
    return coinflip, in_spikes, target

