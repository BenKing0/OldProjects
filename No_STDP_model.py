import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rnd

class model:
    def __init__(self,Ne,Ni,v_threshold,weights,T,m=None,track=False,rnd_inp=False):
        self.Ne = int(Ne)
        self.Ni = int(Ni)
        self.N = int(Ne + Ni)
        self.v_threshold = v_threshold
        self.weights = weights
        self.T = T
        if m:
            self.m = m
        else:
            self.m = None
        v0,u0 = self.initialise()
        self.history = self.dynamics(v0,u0,rnd_inp)
        self.n_fired = self.overview(track)
        
    def initialise(self):
        '''Initial parameters for coupled ODEs'''
        self.a = np.concatenate((0.02*np.ones(self.Ne),
                                 0.02*np.ones(self.Ni)+0.08*rnd.rand(self.Ni)))
        self.b = np.concatenate((0.2*np.ones(self.Ne),
                                 0.25*np.ones(self.Ni)-0.05*rnd.rand(self.Ni)))
        self.c = np.concatenate((-65*np.ones(self.Ne)+15*rnd.rand(self.Ne),
                                 -65*np.ones(self.Ni)))
        self.d = np.concatenate((8*np.ones(self.Ne)-6*rnd.rand(self.Ne)**2,
                                 2*np.ones(self.Ni)))
        v0 = -65*np.ones(self.N)
        u0 = self.b*v0
        self.conductance = np.concatenate((self.weights[0]*rnd.rand(self.Ne,self.N),
                                   self.weights[1]*rnd.rand(self.Ni,self.N)),axis=0)
        for k in range(self.N):
            self.conductance[k][k] = 0
            if k >= self.Ne:
                self.conductance[k][self.Ne:self.N] = 0
            # Set (N-m) weights to 0 for excitatory neurons, 
            # and (Ne-m) weights to 0 for Inhibitory neurons
            remIndE = rnd.choice(self.N,(int(self.N-self.m),1),replace=False)
            remIndI = rnd.choice(self.Ne,(int(self.Ne-self.m),1),replace=False)
            if self.m:
                if k < self.Ne:
                    self.conductance[k][remIndE] = 0
                else:
                    self.conductance[k][remIndI] = 0
        return v0,u0
    
    def dynamics(self,v,u,rnd_inp):
        history = []
        vE = []
        vI = []
        R1 = int(self.Ne*rnd.rand(1))
        R2 = int(self.N-self.Ni*rnd.rand(1))
        # Time step to solve ODEs with - dictates numeric stability
        dt = 0.75
        # T*1000 ms = T second(s)
        if rnd_inp == True:
            val = rnd.randint(0,1000*self.T,size=1)
            timing = np.linspace(val,val+9,10)
        else:
            timing = np.linspace(np.nan,np.nan,10)
        for t in range(int(self.T*1000)):
            # 5 mV thalamic excitatory i/p, 2mV thalamic inhibatory i/p
            I = np.concatenate((5*rnd.rand(self.Ne),2*rnd.rand(self.Ni)))
            if t in timing:
                # inject near-threshold potential to all neurons to recreate surge
                I += rnd.randint(10,20,self.N)
            fired = np.where(v>=self.v_threshold)[0]
            vE.append(v[R1])
            vI.append(v[R2])
            history.append(np.concatenate((np.linspace(t,t,len(fired)),fired)))
            for i,j in enumerate(v):
                if i in fired:
                    v[i] = self.c[i]
                    u[i] += self.d[i]    
            for i in range(self.N):  
                # S_ij is from neuron i (pre) into neuron j (post). 
                # Therefore if neuron i fires, all j's of S_ij are triggered: [k][i]
                I[i] += np.sum([self.conductance[k][i] for k in fired])
            v,u = self.equations(v,u,dt,I)
        self.vE = np.where(np.array(vE)>self.v_threshold,self.v_threshold,np.array(vE))
        self.vI = np.where(np.array(vI)>self.v_threshold,self.v_threshold,np.array(vI))
        return history
      
    def equations(self,v,u,dt,I):
        v += (0.04*v**2+5*v+140-u+I)*dt
        u += self.a*(self.b*v-u)*dt
        return v,u
    
    def overview(self,track):
        # T*1000 ms = T second(s)
        n_fired = []
        n_E_fired = np.zeros(self.T*1000)
        n_I_fired = np.zeros(self.T*1000)
        for t,hist in enumerate(self.history):
            n_fired.append(0.5*len(hist))
            if track == True:
                n_E_fired[t] = len(np.where(hist[int(0.5*len(hist)):len(hist)]<self.Ne)[0])
                n_I_fired[t] = len(np.where(hist[int(0.5*len(hist)):len(hist)]>=self.Ne)[0])
        self.n_E_fired = n_E_fired
        self.n_I_fired = n_I_fired
        return np.array(n_fired)
    
    def results(self):
        '''The firing history for all neurons ([0]), and the total number fired ([1])'''
        # `self.history` is a list of neurons that fired (2nd half), and the 
        # time that they fired at (1st half).
        return self.history,self.n_fired
    
    def track(self):
        '''Tracks the number of excitatory neurons ([0]) and inhibitory neurons ([1]) that have fired'''
        return self.n_E_fired,self.n_I_fired
    
    def case_study(self):
        '''Follows the potential for an excitatory and an inhibitory neuron'''
        return self.vE,self.vI
    
#%%
'''This model is a simplified version of Izhikevich's polychronous model.
It has no delay parameter for an AP, and as such no STDP function.

In addition, it assumes `m` connections per neuron, and that inhibitory 
neurons cannot connect with other inhibitory neurons. 

It takes account of heterogenous firing patterns wiht randomised parameters.'''

# Add in:
    # STDP - needs a delay matrix, and an N x (1000T+D) matrix for a `future` view.