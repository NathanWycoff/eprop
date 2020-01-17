#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.12.2020

class NeuralNetwork(object):
    """
    A very general neural network as defined in Bellec et al. 

    f - Latent to visible function
    g - State transition function
    get_xt - Input function

    R - Dimension of state space for a single neuron.
    H - Number of recurrent neurons.
    P - Number of input channels.
    Q - Number of output channels.

    net_params - Additional, fixed network parameters, such as membrane time constants. Should be a dictionary of arbitrary objects.
    trainable_params - Parameters to be learned via eprop, such as connection weights. Should be a dictionary of np.arrays.
    eprop_funcs - A dictionary of functions needed to do eprop.

    """

    #TODO: learn rate should be a full optmier.
    def __init__(self, f, g, get_xt, R, H, P, Q, net_params, trainable_params, eprop_funcs = None, learn_rate = 0.001, update_every = 1):
        self.f = f
        self.g = g
        self.get_xt = get_xt
        self.H = H
        self.P = P
        self.Q = Q
        self.R = R
        self.net_params = net_params
        self.trainable_params = trainable_params
        self.eprop_funcs = eprop_funcs
        self.learn_rate = learn_rate
        self.update_every = update_every
        self.n_runs = 0

        self.grad_in = np.zeros([self.H, self.P])
        self.grad_rec = np.zeros([self.H, self.H])
        self.grad_out = np.zeros([self.Q, self.H])

        assert Q == 1

        # Initialize state to zeros.
        self.st1 = np.zeros([self.R, self.H])
        self.zt1 = f(self, self.st1)

    def run(self, t_steps, ip, target = None, train = False, save_states = False, save_traces = False):
        """
        Run the simulation for the specified number of steps with input parameters given in the dictionary ip.

        if save_states is true, will return the values of the latent and observable states for all timesteps of the simulation.

        target - An indexible collection of length t_steps; gives the target at a given time point (1D for now)
        """

        if train:
            if target is None:
                raise ValueError("Need a target to do training; unsupervised training is not available.")
            if self.eprop_funcs is None:
                raise ValueError("Need to specify eprop_funcs to do training; audo-diff is not available.")

        if train:
            epst1 = np.zeros([self.H, self.H, self.R, 1])
            epst1_in = np.zeros([self.H, self.P, self.R, 1])
            filt_et = np.zeros([self.H, self.H])
            filt_et_in = np.zeros([self.H, self.P])
            dydout = np.zeros([self.Q,self.H])


        self.n_runs += 1

        if target is not None:
            cost = 0

        if save_states:
            S = np.zeros([self.R, self.H, t_steps])
            Z = np.zeros([self.H, t_steps])

        if save_traces:
            ET = np.zeros([self.H, self.H, t_steps])
            ET_in = np.zeros([self.H, self.P, t_steps])
            EPS = np.zeros([self.H, self.H, self.R, t_steps])
            EPS_in = np.zeros([self.H, self.P, self.R, t_steps])

        ys = np.zeros([self.Q, t_steps])
        yt1 = np.zeros([self.Q,1])

        st1 = self.st1
        zt1 = self.zt1
        for ti in range(t_steps):
            xt = self.get_xt(self, ip, ti)
            st = self.g(self, st1, zt1, xt)
            zt = self.f(self, st)

            yt = self.net_params['kappa'] * yt1 + self.trainable_params['out'] @ zt
            ys[:,ti] = yt

            if target is not None:
                cost += np.sum(np.square(yt - target[ti]))

            if train:

                # TODO: There is much code reuse between these next few blocks: we should write a function. The logic is very similar as it is, but I think there's a smart way to simultaneously get rid of tiling and make the code more similar.

                ## E-prop 1 for input connectivity.
                #TODO: The tiling is highly inefficient, I'm sure we can figure out how to do it without it.
                Dt = self.eprop_funcs['D'](self, st, st1)
                Dts = np.expand_dims(Dt, 1)
                Dts = np.tile(Dts, (1,self.P,1,1))
                dsdt = self.eprop_funcs['d_st_d_tp'](self, st, xt)
                epst_in = Dts @ epst1_in + dsdt
                dzds = self.eprop_funcs['dzds'](self, st)
                dzds = np.expand_dims(dzds, 1)
                dzds = np.expand_dims(dzds, 1)
                dzds = np.tile(dzds, (1,self.P,1,1))
                et_in = (dzds @ epst_in).reshape([self.H,self.P])

                filt_et_in = kappa * filt_et_in + et_in
                
                Lt = self.eprop_funcs['L'](self, yt, target[ti])

                # The commented out line seems to be equivalent (and hence, more efficient), although less readable, as array multiplication with different dimensions is unintuitive.
                #gradt_in = filt_et_in * Lt
                gradt_in = np.diag(Lt) @ filt_et_in
                self.grad_in += gradt_in

                # E-prop 1 for Recurrent connectivity.
                #TODO: The tiling is highly inefficient, I'm sure we can figure out how to do it without it.
                Dt = self.eprop_funcs['D'](self, st, st1)
                Dts = np.expand_dims(Dt, 1)
                Dts = np.tile(Dts, (1,self.H,1,1))
                dsdt = self.eprop_funcs['d_st_d_tp'](self, st, zt)
                epst = Dts @ epst1 + dsdt
                dzds = self.eprop_funcs['dzds'](self, st)
                dzds = np.expand_dims(dzds, 1)
                dzds = np.expand_dims(dzds, 1)
                dzds = np.tile(dzds, (1,self.H,1,1))
                et = (dzds @ epst).reshape([self.H,self.H])

                filt_et = kappa * filt_et + et
                
                Lt = self.eprop_funcs['L'](self, yt, target[ti])

                # The commented out line seems to be equivalent (and hence, more efficient)
                #gradt_rec = filt_et * Lt
                gradt_rec = np.diag(Lt) @ filt_et
                self.grad_rec += gradt_rec

                # Readout weights are easier.
                assert self.Q == 1
                delta = float(yt - target[ti]) 
                dydout = self.net_params['kappa'] * dydout + zt
                gradt_out = delta * dydout
                self.grad_out += gradt_out

                # Update vars
                epst1_in = epst_in
                epst1 = epst

                if save_traces:
                    ET[:,:,ti] = et
                    ET_in[:,:,ti] = et_in
                    EPS[:,:,:,ti] = epst.reshape([self.H,self.H,self.R])
                    EPS_in[:,:,:,ti] = epst_in.reshape([self.H,self.P,self.R])

            if save_states:
                S[:,:,ti] = st
                Z[:,ti] = zt

            st1 = st
            zt1 = zt
            yt1 = yt

        # Just gradient descent for now, and just on recurrent connections.
        if train and (self.n_runs % self.update_every == 0):
            # We choose to disallow self-connections, though it might be interesting to explore in the future.
            self.grad_rec = self.grad_rec - np.diag(np.diag(self.grad_rec))
            self.trainable_params['in'] = self.trainable_params['in'] - self.learn_rate * self.grad_in
            self.trainable_params['rec'] = self.trainable_params['rec'] - self.learn_rate * self.grad_rec
            self.trainable_params['out'] = self.trainable_params['out'] - self.learn_rate * self.grad_out
            self.grad_in = np.zeros([self.H, self.P])
            self.grad_rec = np.zeros([self.H, self.H])
            self.grad_out = np.zeros([self.Q, self.H])

        self.st1 = st1
        self.zt1 = zt1

        ret = {}
        ret['y'] = ys

        if save_states:
            ret['S'] = S
            ret['Z'] = Z
        if target is not None:
            ret['cost'] = cost
        if save_traces:
            ret['ET'] = ET
            ret['ET_in'] = ET_in
            ret['EPS'] = EPS
            ret['EPS_in'] = EPS_in

        return ret


class SpikingNeuralNetwork(NeuralNetwork):
    """
    Slightly more specific, will have weight initializers as well as the opportunity for additional parameters.

    """
    pass

class LifNeuralNetwork(SpikingNeuralNetwork):
    """
    Should be very quick to use and instantiate.

    """
    pass
