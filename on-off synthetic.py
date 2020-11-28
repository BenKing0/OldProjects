window = 0.5
dir = 'C:/Users/bened/Documents/University/ICL Project/Papers/Codes/Cache/PE window {}sec/'.format(window)
mods = 'C:/Users/bened/Documents/University/ICL Project/Papers/Codes/'

#%%
for seed in [1234,4321,2469,2580,1738]:
    print('\n'+str(seed))
    for noise in [-20,-10,-5,0,10,20]:
        import os
        import sys
        file_dir = os.path.dirname(__file__)
        sys.path.append(file_dir)
        sys.path.append(mods)
        import readers 
        import entropies
        import numpy as np
        import matplotlib.pyplot as plt
        #seed = 2468
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
        X_two += N
            
        X = np.concatenate((X_one,X_two))
        X = np.concatenate((X,N[:int(len(N)/2)]))
        X_plot = X.copy()
        
        plt.plot(np.arange(len(X_plot))/fs,X_plot)
        plt.title('Raw signal pre-filter. {} dB'.format(noise))
        plt.show()
        
        np.save(dir+'NF_raw_cache_{0}db_{1}.npy'.format(noise,seed),X)
        
        # Uncomment when raw signal saved
        ''' 
        import Denoise as dn
        X = dn.Gaussian(X_plot,sigma=3,kernel_width=9).results()
        plt.plot(np.arange(len(X))/fs,X)
        plt.show()
        
        
        PE_X = entropies.PE(X,bin_width=3,normalised=True,window_size=int(fs*window)).results()
        np.save(dir+'GF_PE_cache_{0}db_{1}.npy'.format(noise,seed),PE_X)

        
        import numpy as np
        PE_X = np.load(dir+'GF_PE_cache_{0}db_{1}.npy'.format(noise,seed))
        shift = len(X) - len(PE_X)
        xs = np.divide((np.arange(len(PE_X)) + shift),fs)
        fig = plt.figure()
        fig.add_axes((0,0,1.5,0.65))
        plt.title('On/Off signal - varying # of MUs. SNR = {} db'.format(noise))
        plt.plot(np.arange(len(X)),X)
        plt.ylabel('Signal (mV) - filtered')
        plt.xticks([])
        fig.add_axes((0,-0.9,1.5,0.85))
        plt.plot(xs,PE_X)
        plt.ylabel('PE (normalised)')
        plt.xlabel('Time (sec)')
        plt.xticks(np.arange(max(xs),step=3))
        plt.show()'''

#%% START CWT

import CWT
from CWT import HermiteRodriguez as HR
from CWT import MexicanHat as MH

mother,name = MH,'MH' 
widths = np.arange(5,45,step=5) * 1e-3
Y = CWT.CWT_filter_bank(X_plot, mother, widths, fs=fs, T=int(2.5*T))()
np.save(dir+'CWT_cache_{0}_{1}db_{2}.npy'.format(name,noise,seed),Y)

#%%

mother,name = MH,'MH' 
from Denoise import Gaussian
Y_ = np.load(dir+'CWT_cache_{1}_{0}db_{2}.npy'.format(noise,name,seed))
Y = Gaussian(Y_,sigma=1,kernel_width=9).results()
plt.plot(np.arange(len(Y))/fs,Y,'r--',label='After')
#plt.plot(np.arange(len(X_plot))/fs,X_plot,'b-',label='Before')
plt.legend()
plt.show()

#%%

PE_Y = entropies.PE(Y,bin_width=3,normalised=True,window_size=fs).results()
np.save(dir+'cache_synth_{1}_{0}db_{2}.npy'.format(noise,name,seed),PE_Y)

#%%

PE_Y = np.load(dir+'cache_synth_{1}_{0}db_{2}.npy'.format(noise,name,seed))

shift = len(Y) - len(PE_Y)
xs = np.divide((np.arange(len(PE_Y)) + shift),fs)
fig = plt.figure()
fig.add_axes((0,0,1.5,0.65))
plt.title('On/Off signal - varying # of MUs. SNR = {} db'.format(noise))
plt.plot(np.arange(len(Y)),Y)
plt.ylabel('Signal (mV) - filtered')
plt.xticks([])
fig.add_axes((0,-0.9,1.5,0.85))
plt.plot(xs,PE_Y)
plt.ylabel('PE (normalised)')
plt.xlabel('Time (sec)')
plt.xticks(np.arange(max(xs),step=3))
plt.show()

#%% Adaptive threshold:
'''
keep influence approx 0 to pick ALL (almost) values below threshold rather than just the start of below threshold,
and keep window_size smaller for greater sensitivity. Sensitivity approx 10% of window?
'''

import adaptive_threshold as at

pred_fire = np.array(at.below(PE_Y,gamma=2,window_size=1000,influence=0,disable=False))
pred_fire += len(Y)-len(PE_Y) # account for pred_fire indices starting from 0, when PE starts from a greater value
true_fire = int(len(X_one)/2)

sensitivity = 20 # number of next hits to test after each hit
to_del = []
count = 0
for i in range(len(pred_fire)-(sensitivity+1)):
    start,stop = pred_fire[i]+1,pred_fire[i]+sensitivity+1
    bool_array = [val in pred_fire for val in np.arange(start,stop)]
    if sum(bool_array) < sensitivity: # `sum(bool_array)` is number of `True`s in array - ie. `any False's?`
        to_del.append(i)
        count += 1
pred_fire = np.delete(pred_fire,to_del)

print(count)
bias = true_fire - min(pred_fire)
print('\nBias is {:.0f} samples'.format(bias))
print('Bias is {:.2f} sec'.format(bias/fs))

assert max(pred_fire) <= max(xs)*fs

shift = len(Y)-len(PE_Y)

xs = np.divide((np.arange(len(PE_Y)) + shift),fs)
fig = plt.figure()
fig.add_axes((0,0,1.5,0.65))
plt.title('On/Off signal - varying # of MUs. SNR = {} db'.format(noise))
plt.plot(np.arange(len(Y)),Y)
plt.ylabel('Signal (mV) - filtered')
plt.xticks([])
fig.add_axes((0,-0.9,1.5,0.85))
plt.plot(xs,PE_Y)
plt.ylabel('PE (normalised)')
plt.xlabel('Time (sec)')
plt.plot(pred_fire/fs,PE_Y[pred_fire-shift],'r.')
plt.xticks(np.arange(max(xs),step=3))
plt.show()

#%% Global threshold: (uses a reference pure Gaussian noise signal with 2 std)
    
shift = len(Y) - len(PE_Y)
sim_noise = generate_series(np.random.randn(fs//10), SNR=noise)
PE_ref = entropies.PE(sim_noise,bin_width=3,window_size=200,fixed=False).results()
PE_ref = PE_Y[:fs//10]
global_threshold = np.mean(PE_ref) - 2 * np.std(PE_ref)

print('\nGlobal Threshold: {:.3f}'.format(global_threshold))
pred_fire_global = np.array([ind for ind in np.arange(len(PE_Y)) if PE_Y[ind] < global_threshold])
pred_fire_global += shift

sensitivity = fs//10 # number of next hits to test after each hit
to_del = []
count = 0
from tqdm import trange
for i in trange(len(pred_fire)-(sensitivity+1)):
    start,stop = pred_fire[i]+1,pred_fire[i]+sensitivity+1
    bool_array = [val in pred_fire for val in np.arange(start,stop)]
    if sum(bool_array) < sensitivity: # `sum(bool_array)` is number of `True`s in array - ie. `any False's?`
        to_del.append(i)
        count += 1
pred_fire = np.delete(pred_fire,to_del)
print(count)

true_fire = int(len(X_one)/2)
bias = true_fire - min(pred_fire_global)
print('\nBias is {:.0f} samples'.format(bias))
print('Bias is {:.2f} sec'.format(bias/fs))

xs = np.divide((np.arange(len(PE_Y)) + shift),fs)
fig = plt.figure()
fig.add_axes((0,0,1.5,0.65))
plt.title('On/Off signal - varying # of MUs. SNR = {} db'.format(noise))
plt.plot(np.arange(len(Y)),Y)
plt.ylabel('Signal (mV) - filtered')
plt.xticks([])
fig.add_axes((0,-0.9,1.5,0.85))
plt.plot(xs,PE_Y)
plt.ylabel('PE (normalised)')
plt.xlabel('Time (sec)')
plt.plot(pred_fire_global/fs,PE_Y[pred_fire_global-shift],'r.')
plt.xticks(np.arange(max(xs),step=3))
plt.show()

#%% Bonato

from Bonato import sumSquares as ss
from Bonato import globalThreshold as gt

'''noisey = np.random.normal(size=3000)
gf_noisey = dn.Gaussian(noisey,kernel_width=3,sigma=9).results()
PE_noise = entropies.PE(gf_noisey,bin_width=3,normalised=True,window_size=2048).results()
mu = np.mean(PE_noise)
sigma = np.std(PE_noise)'''

type = 'CWT'
noise = 20
name = 'MH'

if type == 'CWT':
    Y = np.load(dir+'CWT_cache_{1}_{0}db_{2}.npy'.format(noise,name,seed))
    PE_X = Y
    X = Y
else:
    PE_X = np.load(dir+'GF_PE_cache_{0}db_{1}.npy'.format(noise,seed))
shift = len(X) - len(PE_X)

mu = np.mean(PE_X[:3*fs])
sigma = np.std(PE_X[:3*fs])

print('\nNoise mean: {0:.3e}, Noise standard dev: {1:.3e}'.format(mu,sigma))

z_stat_signal = (PE_X - mu) / sigma 
zeta = 6

#result = gt(z_stat_signal,lag=2,threshold=zeta,direction='above')() # `above`, as sum of squares v large for both v positive AND v negative z_statistics
#indices = result[0]
indices = np.where(z_stat_signal > zeta)[0]
print('\nNumber of Highlighted indices: {}'.format(len(indices)))
print('\nSum Square values: ')

sensitivity = fs//20 # number of next hits to test after each hit
to_del = []
count = 0
'''
from tqdm import trange
if type != 'CWT':
    for i in trange(len(indices)-(sensitivity+1)):
        start,stop = indices[i]+1,indices[i]+sensitivity+1
        bool_array = [val in indices for val in np.arange(start,stop)]
        if sum(bool_array) < sensitivity: # `sum(bool_array)` is number of `True`s in array - ie. `any False's?`
            to_del.append(i)
            count += 1
        if PE_X[2*indices[i]] >= mu:
            to_del.append(i)
            count += 1
    for i in range(len(indices)-(sensitivity+1),len(indices)):
        if PE_X[2*indices[i]] >= mu:
            to_del.append(i)
            count += 1
    
    indices = np.delete(indices,to_del)
print('\nNumber of removed indices: {}'.format(count))
indices = 2 * indices
'''

xs = np.divide((np.arange(len(PE_X)) + shift),fs)
fig = plt.figure()
fig.add_axes((0,0,1.5,0.65))
plt.title('On/Off signal - varying # of MUs. SNR = {} db'.format(noise))
plt.plot(np.arange(len(X)),X)
plt.ylabel('Signal (mV) - filtered')
plt.xticks([])
fig.add_axes((0,-0.9,1.5,0.85))
plt.plot(xs,PE_X)
plt.ylabel('PE (normalised)')
plt.xlabel('Time (sec)')
plt.plot((indices+shift)/fs,PE_X[indices],'r.')
plt.xticks(np.arange(max(xs),step=3))
plt.show()

pred_fire = indices[0]
true_fire = int(len(X_one)/2)
bias = true_fire - pred_fire - shift
print('\nBias is {:.0f} samples'.format(bias))
print('Bias is {:.2f} sec'.format(bias/fs))

#%% ROC curve - all 3 methods:
import os
os.chdir(mods)
from Bonato import globalThreshold as gt
import numpy as np
from tqdm import trange

#type = 'PE'
#noise = 20
name = 'MH'
seeds = [1234,2469,2580,4321,1738]

fs = 2048
true_pos = 8 * fs #+ (1 - fs) * 0.5
true_neg = 12 * fs #+ (1 - fs) * 0.5

sensitivity = fs//20 # number of next hits to test after each hit
PE_X = np.load(dir+'GF_PE_cache_{0}db_{1}.npy'.format(-20,1234))
X = np.load(dir+'NF_raw_cache_{0}db_{1}.npy'.format(-20,1234))
shift = len(X) -  len(PE_X)

multiplicative_factor = 2 #* ((len(PE_X) + 1 - fs) / len(PE_X))
print('Indice multiplicative factor: {}'.format(multiplicative_factor)) # the number of PE indices represented by each chi-squared highlighted indice?

for seed in seeds:
    print('\n{}'.format(seed))
    for types in ['raw']:#['PE','CWT']:
        for noise in [-20,-10,-5,0,10,20]:
            
            if types == 'CWT':
                Y = np.load(dir+'CWT_cache_{1}_{0}db_{2}.npy'.format(noise,name,seed))
                zetas = np.array([0,0.01,0.1,0.2,0.5,1,1.5,2,2.5,3,4,5,10,20],dtype=np.single)*5
                PE_X = Y
                X = Y
            elif types == 'PE':
                zetas = [0,0.01,0.1,0.2,0.5,1,1.5,2,2.5,5,10,20,100,200,500,1000] 
                PE_X = np.load(dir+'GF_PE_cache_{0}db_{1}.npy'.format(noise,seed))
            elif types == 'raw':
                zetas = [0,0.01,0.1,0.2,0.5,1,1.5,2,2.5,5,10,20,100,200,500,1000] 
                PE_X = np.load(dir+'NF_raw_cache_{0}db_{1}.npy'.format(noise,seed))
            mu = np.mean(PE_X[:3*fs])
            sigma = np.std(PE_X[:3*fs])
            print('\nNoise: {} dB'.format(noise))
            print('\nNoise mean: {0:.3e}, Noise standard dev: {1:.3e}'.format(mu,sigma))
            z_stat_signal = (PE_X - mu) / sigma 
            results = []
            
            for i in range(len(zetas)):
                # post-processing with sensitivity tolerance
                
                zeta = zetas[i]
                
                if types == 'CWT':
                    indices = [index for index in range(len(z_stat_signal)) if abs(z_stat_signal[index]) > zeta]
                    to_del = []
                    count = 0
                    '''
                    if len(indices) > sensitivity + 1:
                        for i in trange(len(indices)-(sensitivity+1)):
                            start,stop = indices[i]+1,indices[i]+sensitivity+1
                            bool_array = [val in indices for val in np.arange(start,stop)]
                            if sum(bool_array) < sensitivity: # `sum(bool_array)` is number of `True`s in array - ie. `any False's?`
                                to_del.append(i)
                                count += 1
                            if PE_X[2*indices[i]] >= mu:
                                to_del.append(i)
                                count +=  1
                        for i in range(len(indices)-(sensitivity+1),len(indices)):
                            if PE_X[2*indices[i]] >= mu:
                                to_del.append(i)
                                count += 1
                        indices = np.delete(indices,to_del)
                    '''
                
                elif types != 'CWT':
                    _ = gt(z_stat_signal,lag=2,threshold=zeta,direction='above')() # `above`, as sum of squares v large for both v positive AND v negative z_statistics
                    indices = _[0]
                    to_del = []
                    count = 0
                    if len(indices) > sensitivity + 1:
                        for i in trange(len(indices)-(sensitivity+1)):
                            start,stop = indices[i]+1,indices[i]+sensitivity+1
                            bool_array = [val in indices for val in np.arange(start,stop)]
                            if sum(bool_array) < sensitivity: # `sum(bool_array)` is number of `True`s in array - ie. `any False's?`
                                to_del.append(i)
                                count += 1
                            if PE_X[2*indices[i]] >= mu:
                                to_del.append(i)
                                count +=  1
                        for i in range(len(indices)-(sensitivity+1),len(indices)):
                            if PE_X[2*indices[i]] >= mu:
                                to_del.append(i)
                                count += 1
                        indices = np.delete(indices,to_del)
                    indices = 2 * np.array(indices,dtype=np.int)
                
                if types == 'CWT':
                    bool = [(indice > 4*fs and indice < 8*fs) or (indice > 12*fs and indice < 16*fs) for indice in indices]
                elif types == 'raw':
                    bool = [(indice > 4*fs and indice < 8*fs) or (indice > 12*fs and indice < 16*fs) for indice in indices]
                elif types == 'PE':
                    bool = [(indice > 4*fs-shift and indice < 8*fs-shift) or (indice > 12*fs-shift and indice < 16*fs-shift) for indice in indices]
                tp = (1 + (types=='PE' or types=='raw'))*sum(bool) # the `2x` if PE or raw is becasue the indices are half the number of the indices that they represent due to the summation in the chi-square calculation
                tpr = tp/true_pos
                fpr = ((1 + (types=='PE' or types=='raw'))*len(indices)-tp)/true_neg # see above reasoning for `2x`
                print(tpr,fpr)
                
                results.append([tpr,fpr])
            results = np.array(results).T
            
          
            np.save(dir+'ROC_results_{0}_{1}_{2}.npy'.format(str(noise),str(types),seed),results)

#%% all 3 comparison:
    
from scipy.optimize import curve_fit as fit
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1237)
def fit_func(x,a,b):
    if noise_ == -20:
        return np.tanh(a*x)
    else:
        return b * np.tanh(a*x)
    
seeds = [1234,4321,2580,2469,1738] 
method = 'BNT'   
    
fig = plt.figure()
for i,noise_ in enumerate([-20,-10,-5,0,10,20]):
    ys_PE_array,xs_PE_array,ys_CWT_array,xs_CWT_array,ys_raw_array,xs_raw_array = [],[],[],[],[],[]
    for seed_ in seeds:
        ys_PE_array.append(np.load(dir+'ROC_results_{0}_{1}_{2}.npy'.format(str(noise_),str('PE'),seed_))[0])
        xs_PE_array.append(np.load(dir+'ROC_results_{0}_{1}_{2}.npy'.format(str(noise_),str('PE'),seed_))[1])
        ys_CWT_array.append(np.load(dir+'ROC_results_{0}_{1}_{2}.npy'.format(str(noise_),str('CWT'),seed_))[0])
        xs_CWT_array.append(np.load(dir+'ROC_results_{0}_{1}_{2}.npy'.format(str(noise_),str('CWT'),seed_))[1])
        ys_raw_array.append(np.load(dir+'ROC_results_{0}_{1}_{2}.npy'.format(str(noise_),str('raw'),seed_))[0])
        xs_raw_array.append(np.load(dir+'ROC_results_{0}_{1}_{2}.npy'.format(str(noise_),str('raw'),seed_))[1])

    ys_PE,xs_PE = np.mean(np.array(ys_PE_array),axis=0),np.mean(np.array(xs_PE_array),axis=0)
    ys_CWT,xs_CWT = np.mean(np.array(ys_CWT_array),axis=0),np.mean(np.array(xs_CWT_array),axis=0)
    err_ys_PE,err_xs_PE = np.std(np.array(ys_PE_array),axis=0),np.std(np.array(xs_PE_array),axis=0)
    err_ys_CWT,err_xs_CWT = np.std(np.array(ys_CWT_array),axis=0),np.std(np.array(xs_CWT_array),axis=0)
    ys_raw,xs_raw = np.mean(np.array(ys_raw_array),axis=0),np.mean(np.array(xs_raw_array),axis=0)
    err_ys_raw,err_xs_raw = np.std(np.array(ys_raw_array),axis=0),np.std(np.array(xs_raw_array),axis=0)
    
    ys_PE = np.insert(ys_PE,0,1)
    xs_PE = np.insert(xs_PE,0,1)
    ys_CWT = np.insert(ys_CWT,0,1)
    xs_CWT = np.insert(xs_CWT,0,1)
    ys_raw = np.insert(ys_raw,0,1)
    xs_raw = np.insert(xs_raw,0,1)
    err_ys_PE = np.insert(err_ys_PE,0,0)
    err_xs_PE = np.insert(err_xs_PE,0,0)
    err_ys_CWT = np.insert(err_ys_CWT,0,0)
    err_xs_CWT = np.insert(err_xs_CWT,0,0)
    err_ys_raw = np.insert(err_ys_raw,0,0)
    err_xs_raw = np.insert(err_xs_raw,0,0)
    
    fig.add_axes(((i%3)*1.1,-1.2*(noise_ >= 0),0.95,0.95))
    plt.title('SNR = {} dB'.format(noise_))
    plt.errorbar(xs_PE,ys_PE,yerr=err_ys_PE,xerr=err_xs_PE,fmt='k.',elinewidth=0.4)
    plt.plot(xs_PE,ys_PE,'k-',linewidth=0.6,label=r'PE + $\chi^2$-threshold')
    
    plt.errorbar(xs_CWT,ys_CWT,yerr=err_ys_CWT,xerr=err_xs_CWT,fmt='r*',elinewidth=0.4)
    plt.plot(xs_CWT,ys_CWT,'r-',linewidth=0.6,label=r'CWT + $\sigma$-threshold')
    
    plt.errorbar(xs_raw,ys_raw,yerr=err_ys_raw,xerr=err_xs_raw,fmt='bo',elinewidth=0.4)
    plt.plot(xs_raw,ys_raw,'b-',linewidth=0.6,label=r'Unfiltered + $\chi^2$-threshold')
    
    plt.ylim(0,1.05)
    plt.xlim(0,1.05)
    plt.xlabel('False Positive Probability')
    plt.ylabel('True Positive Probability')
    plt.legend(loc='lower right')
fig.savefig(dir+'ROC_total_{0}_seeds_all3.png'.format(len(seeds)),bbox_inches='tight')
plt.show()

# Log plot?

#%% Single standard deviation based method vs chi-square method:
    
seeds = [2469]#[1234,4321,2580,2469,1738]    
    
fig = plt.figure()
for i,noise_ in enumerate([-20,-10,-5,0,10,20]):
    ys_PE_array,xs_PE_array,ys_SD_array,xs_SD_array = [],[],[],[]
    for seed_ in seeds:
        ys_PE_array.append(np.load(dir+'ROC_results_{0}_{1}_{2}_SD.npy'.format(str(noise_),str('PE'),seed_))[0])
        xs_PE_array.append(np.load(dir+'ROC_results_{0}_{1}_{2}_SD.npy'.format(str(noise_),str('PE'),seed_))[1])
        ys_SD_array.append(np.load(dir+'ROC_results_{0}_{1}_{2}_SD.npy'.format(str(noise_),str('CWT'),seed_))[0])
        xs_SD_array.append(np.load(dir+'ROC_results_{0}_{1}_{2}_SD.npy'.format(str(noise_),str('CWT'),seed_))[1])

    ys_PE,xs_PE = np.mean(np.array(ys_PE_array),axis=0),np.mean(np.array(xs_PE_array),axis=0)
    #ys_PE,xs_PE = np.insert(ys_PE,0,0),np.insert(xs_PE,0,0)
    #ys_PE,xs_PE = np.append(ys_PE,1),np.append(xs_PE,1)
    ys_SD,xs_SD = np.mean(np.array(ys_SD_array),axis=0),np.mean(np.array(xs_SD_array),axis=0)
    err_ys_PE,err_xs_PE = np.std(np.array(ys_PE_array),axis=0),np.std(np.array(xs_PE_array),axis=0)
    err_ys_SD,err_xs_SD = np.std(np.array(ys_SD_array),axis=0),np.std(np.array(xs_SD_array),axis=0)
    #err_ys_PE,err_xs_PE = np.insert(err_ys_PE,0,0),np.insert(err_xs_PE,0,0)
    #err_ys_PE,err_xs_PE = np.append(err_ys_PE,0),np.append(err_xs_PE,0)
    
    '''gradient = [1.]*len(xs_PE)
    #gradient = (np.linspace(0.01,0.99,len(xs))**0.5)
    params_PE, cov_PE = fit(fit_func,xs_PE,ys_PE,p0=np.array([1.,1.]),sigma=gradient)
    params_CWT, cov_CWT = fit(fit_func,xs_CWT,ys_CWT,p0=np.array([1.,1.]),sigma=gradient)
    
    UB_PE_a = params_PE[0] + np.sqrt(cov_PE.flatten()[0])
    LB_PE_a = params_PE[0] - np.sqrt(cov_PE.flatten()[0])
    UB_PE_b = params_PE[1] + np.sqrt(cov_PE.flatten()[-1])
    LB_PE_b = params_PE[1] - np.sqrt(cov_PE.flatten()[-1])
    UB_CWT_a = params_CWT[0] + np.sqrt(cov_CWT.flatten()[0])
    LB_CWT_a = params_CWT[0] - np.sqrt(cov_CWT.flatten()[0])
    UB_CWT_b = params_CWT[1] + np.sqrt(cov_CWT.flatten()[-1])
    LB_CWT_b = params_CWT[1] - np.sqrt(cov_CWT.flatten()[-1])'''
    
    fig.add_axes(((i%3)*1.1,-1.2*(noise_ >= 0),0.95,0.95))
    plt.title('SNR = {} dB'.format(noise_))
    plt.errorbar(xs_PE,ys_PE,yerr=err_ys_PE,xerr=err_xs_PE,fmt='k.',elinewidth=0.4)
    #plt.plot(np.linspace(0,1,100),fit_func(np.linspace(0,1,100),*params_PE),'k--',linewidth=0.6,label='PE (fitted to tanh curve)')
    #plt.plot(np.linspace(0,1,100),fit_func(np.linspace(0,1,100),UB_PE_a,UB_PE_b),'k--',linewidth=0.2)
    #plt.plot(np.linspace(0,1,100),fit_func(np.linspace(0,1,100),LB_PE_a,LB_PE_b),'k--',linewidth=0.2)
    plt.plot(xs_PE,ys_PE,'k--',linewidth=0.6,label=r'$\chi^2$ (no fitting)')
    
    plt.errorbar(xs_SD,ys_SD,yerr=err_ys_SD,xerr=err_xs_SD,fmt='r*',elinewidth=0.4)
    #plt.plot(np.linspace(0,1,100),fit_func(np.linspace(0,1,100),*params_CWT),'r--',linewidth=0.6,label='CWT (fitted to tanh curve)')
    #plt.plot(np.linspace(0,1,100),fit_func(np.linspace(0,1,100),UB_CWT_a,UB_CWT_b),'r--',linewidth=0.2)
    #plt.plot(np.linspace(0,1,100),fit_func(np.linspace(0,1,100),LB_CWT_a,LB_CWT_b),'r--',linewidth=0.2)
    plt.plot(xs_SD,ys_SD,'r--',linewidth=0.6,label='Standard Deviation (no fitting)')
    
    plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),'c-',linewidth=0.4,label='Baseline')
    plt.ylim(0,1.05)
    plt.xlim(0,1.05)
    plt.xlabel('False Positive Probability')
    plt.ylabel('True Positive Probability')
    plt.legend(loc='lower right')
fig.savefig(dir+'ROC_total_{}_seeds_SDvBon.png'.format(len(seeds)),bbox_inches='tight')
plt.show()
