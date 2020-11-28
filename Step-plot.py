window = 0.5
dir = 'C:/Users/bened/Documents/University/ICL Project/Papers/Codes/Cache/PE window {}sec/'.format(window)
mods = 'C:/Users/bened/Documents/University/ICL Project/Papers/Codes/'

#%%
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import readers 
import numpy as np
import matplotlib.pyplot as plt
from Bonato import globalThreshold as gt
from tqdm import trange

noises = [-20,-10,-5,0,10,20]
seeds = [1234]

for seed in seeds:
    fig = plt.figure()
    counter = 0
    for noise in noises:
        counter += 1
        
        #seed = 4321
        np.random.seed(seed)
        args = [2,2,1000,2048]
        Xi = readers.GetSimData(args)[0].T
        ind = np.argmax(np.max(Xi,axis=1))
        SNR = 1000
        T = 8
        fs = 2048
        x = np.linspace(0,0,fs*T)
        #noise = -5
        
        def generate_series(x,SNR,reference=[]):
            mean = np.mean(x)
            stddev = (np.std(x)/(10**(SNR/20)))  
            if len(reference) != 0:
                mean = np.mean(reference)
                stddev = (np.std(reference)/(10**(SNR/20)))  
            noise = np.random.normal(mean,stddev,len(x))
            pattern = np.add(x,noise)
            return pattern;
            
        num1,num2 = 1,2
            
        ks = np.random.randint(0,50,size=num2)
        #ks = np.expand_dims(ks[0],axis=0)
        args = [T,num1,SNR,fs]
        X_one = readers.GetSimData(args,ks=ks[:num1])[0].T
        X_one = X_one[ind]
        N = generate_series(x,noise,reference=X_one)
        X_one[:int(len(X_one)/2)] = 0
        X_one += N
        
        args = [T,num2,SNR,fs]
        X_two = readers.GetSimData(args,ks=ks[:num2])[0].T
        X_two = X_two[ind]
        X_two[:int(len(X_two)/2)] = 0
        #N = generate_series(x,noise,reference=N)
        X_two += N
        
        X = np.concatenate((X_one,X_two))
        X = np.concatenate((X,N[:int(len(N)/2)]))
        X_plot = X.copy()
        
        ##%% - Find indices: Bonato
        
        PE_X = np.load(dir+'GF_PE_cache_{0}db_{1}.npy'.format(noise,seed))
        z_stat = (PE_X - np.mean(PE_X[:3*fs])) / (np.std(PE_X[:3*fs]))
        zeta = 5
        indices = gt(z_stat,lag=2,threshold=zeta,direction='above')()[0]
        assert isinstance(indices,np.ndarray)
        shift = len(X_plot) - len(PE_X)
        
        sensitivity = (fs//20 ) # (2 x sensitivity) / fs is sesnsitivity in seconds
        to_del = []
        count = 0
        for i in trange(len(indices)-(sensitivity+1)):
            start,stop = indices[i]+1,indices[i]+sensitivity+1
            bool_array = [val in indices for val in np.arange(start,stop)]
            ## IMPORTANT!: before `indices *= 2`, difference between adjacent indices is still 1, eventhough it represents 2 PE values.
            if sum(bool_array) < sensitivity: # `sum(bool_array)` is number of `True`s in array - ie. `any False's?`
                to_del.append(i)
                count += 1
            if PE_X[2*indices[i]] >= np.mean(PE_X[:3*fs]):
                to_del.append(i)
                count += 1
        for i in range(len(indices)-(sensitivity+1),len(indices)):
            if PE_X[2*indices[i]] >= np.mean(PE_X[:3*fs]):
                to_del.append(i)
                count += 1
        indices = np.delete(indices,to_del)
        
        indices *= 2
        indices += shift
        
        ##%% - make step function
                    
        xs = np.arange(len(X_plot))
        included = np.array([x in indices for x in xs])
        assert sum(included) == len(indices)
        for i in trange(len(included)):
            if included[i] == True and included[i+2] == True:
                included[i+1] = True
        step = np.where(included==True,0.5*max(X_plot),0)
        bias1 = -1000 * (np.where(step>0)[0][0] - 4*fs) / fs
        try:
            second_fire = list(step[10*fs:]).index(0.5*max(X_plot)) + 10*fs
            bias2 = -1000 * (second_fire - 12*fs) / fs
        except:
            bias2 = None
        fig.add_axes((((counter-1) % 3)*1.05,-1.1*((counter-1) > 2),0.9,0.9))      
        plt.plot(np.arange(len(X_plot))/fs,X_plot,'k-',linewidth=0.3)
        try:
            plt.plot(np.arange(len(step))/fs,step,'b-',linewidth=3,label=r'$\zeta$ = {2}; bias: {0:.0f} ms, {1:.0f} ms'.format(bias1,bias2,zeta))
        except TypeError:
            plt.plot(np.arange(len(step))/fs,step,'b-',linewidth=3,label=r'$\zeta$ = {1}; bias: {0:.0f} ms'.format(bias1,zeta))
        plt.ylabel('Amplitude (mV)')
        plt.xlabel('Time (sec)')
        plt.title(r'SNR = {} dB'.format(noise))
        plt.legend(loc='lower right')
        plt.yticks([])
    plt.savefig(dir+'GF_PE_stepplot_{1}-publish.png'.format(noise,seed))
    plt.show()
        
#%% Arrange plots:
from PIL import Image
    
noises = [-20,-10,-5,0,10,20]
seeds = [1234]
 
for seed in seeds:       
    figs = []
    for noise in noises:
        figs.append(plt.imread(dir+'GF_PE_stepplot_{0}db_{1}.png'.format(noise,seed)))
        
    
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5),
                            subplot_kw={'xticks': [], 'yticks': []})
    
    i = 0
    for ax, interp_method in zip(axs.flat, noises):
        ax.imshow(figs[i])
        ax.axis('off')
        i += 1
    
    plt.tight_layout()
    plt.savefig(dir+'concatenated_stepplot_{0}db_{1}.png'.format(noise,seed))