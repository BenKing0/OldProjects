import os
os.chdir('C:/Users/44790/Documents/University/ICL Project/Papers/Codes')
import No_STDP_model as model
import PE 
#%%

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rnd

Num = 1000 # number of Neurons
N_e = 0.8*Num # Excitatory neurons
N_i = 0.2*Num # Inhibitory neurons
v_threshold = 30 # mV
weights = [0.53,-1] # Synapse weights [excitatory,inhibitory], mV
T = 1 # Second

g = model.model(N_e,N_i,v_threshold,weights,T,m=3,track=True,rnd_inp=False)
# `y` contains the firing history for all neurons ([0]), and the total number fired ([1]) 
y = np.array(g.results())
# `cs` (case study) follows the potential for an excitatory and an inhibitory neuron
cs = g.case_study()
# `tracker` tracks the number of excitatory neurons ([0]) and inhibitory neurons ([1]) that have fired
tracker = g.track()

# Plot the map of neuron firings against time
fig1 = plt.figure()
fig1.add_axes((0,0,1.5,1))
for i,j in enumerate(y[0]):
    l = int(len(j)/2)
    plt.scatter(j[:l],j[-l:],c='k',linewidths=0.1,marker='.') 
plt.ylabel('Neuron number')
plt.xticks([])
fig1.add_axes((0,-0.6,1.5,0.55))
plt.plot(np.linspace(0,len(y[1]),len(y[1])),y[1],'k-',linewidth=0.4)
plt.xlabel('Time, ms') 
plt.yticks(np.linspace(0,int(max(y[1])),int(max(y[1])+1)))  
plt.show()

# Plot the membrane potential for the 2 excitatory and Inhibitory neurons
time = np.linspace(0,len(cs[0]),len(cs[0]))
plt.plot(time,cs[0],'r-',label='E')
plt.plot(time,cs[1],'b--',label='I')
plt.legend()
plt.xlabel('Time, ms')
plt.ylabel('Membrane potential, mV')
plt.show()

# Check the tracker works as expected
assert [tracker[0][i]+tracker[1][i]==y[1][i] for i in range(T*1000)]

# Compare the firing patterns of the excitatory and inhibitory neurons
fig2 = plt.figure()
fig2.add_axes((0,0,1.5,0.95))
plt.ylabel('Excitatory')
plt.xticks([])
plt.yticks(np.linspace(0,int(max(tracker[0])),int(max(tracker[0])+1)))  
plt.plot(np.linspace(0,len(tracker[0]),len(tracker[0])),tracker[0],'r-',linewidth=0.4,label='E')
plt.legend()
fig2.add_axes((0,-1,1.5,0.95))
plt.plot(np.linspace(0,len(tracker[1]),len(tracker[1])),tracker[1],'b--',linewidth=0.4,label='I')
plt.ylabel('Inhibitory')
plt.xlabel('Time, ms')
plt.yticks(np.linspace(0,int(max(tracker[1])),int(max(tracker[1])+1)))  
plt.legend()
plt.show()
