from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        alpha[:,0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for t in range(1,L):
            for s in range(S):
                alpha[s,t] = self.B[s, self.obs_dict[Osequence[t]]]*np.sum([alpha[i,t-1]*self.A[i,s] for i in range(S)])
        return alpha
    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        beta[:,L-1] = 1.0
        for t in range(L-1)[::-1]:
            for s in range(S):
                beta[s,t] = np.sum([self.B[i, self.obs_dict[Osequence[t+1]]]*beta[i,t+1]*self.A[s,i] for i in range(S)])
        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        prob = np.sum(self.forward(Osequence)[:,-1])        
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        forword = self.forward(Osequence)
        backward = self.backward(Osequence)
        prob = (forword*backward)/(np.sum(forword[:,-1]))
        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Edit here
        ###################################################
        S = len(self.pi)
        L = len(Osequence)

        paths = np.zeros([S, L])
        probs = np.zeros([S, L])
        probs[:,0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        for t in range(1,L):
            for s in range(S):
                prev_state = self.B[s, self.obs_dict[Osequence[t]]]*np.array([probs[i,t-1]*self.A[i,s] for i in range(S)])
                paths[s,t] = np.int(np.argmax(prev_state))
                probs[s,t] = np.max(prev_state)
        state = np.int(np.argmax(probs[:,-1]))
        for t in range(L)[::-1]:
            path.append(state)
            state = np.int(paths[state,t])

        state_path = []
        for state in path[::-1]:
            state_path.extend([k for k,v in self.state_dict.items() if v == state])
        return state_path