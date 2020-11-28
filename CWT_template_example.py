import os
dir = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(dir)
import numpy as np
import matplotlib.pyplot as plt
import CWT
from CWT import HermiteRodriguez as HR
from CWT import MexicanHat as MH
from Denoise import Gaussian

fs = 1024
T = 2
widths = np.arange(0.05,0.25,step=0.05)

xs = np.arange(fs*T/2)/fs
y = MH(xs,t=0.5,a=0.1)
temp = y + 2.0 * np.random.normal(size=len(y))
signal = np.concatenate((temp,temp))
xs = np.arange(len(signal))/fs

filtered_ = CWT.CWT_filter_bank(signal,MH,widths,fs,T)()
filtered = Gaussian(filtered_,sigma=1,kernel_width=9).results()
xs2 = np.arange(len(filtered))/fs

plt.plot(xs,signal,'b--',label='Data',linewidth=0.5)
plt.plot(xs2,filtered,'r-',label='Filtered',linewidth=4)
plt.title('Mexican Hat mother wavelet CWT for Hermite Rodrigues')
plt.xlabel('Time (ms)')
plt.ylabel('A.U.')
plt.legend()
plt.show()

#%%

import simulation as sim

gen = sim.generateMUAPtrain(1,numMU=1,fs=2048) # 192 channels with 128 sample length
#gen.initSim()
temp = gen.return_templates()
single_channel = temp[0][np.argmax(np.sum(temp,axis=2))]

import matplotlib.pyplot as plt

plt.plot(np.arange(len(single_channel)),single_channel)
plt.title('Example Template')
plt.ylabel('Voltage (mV)')
plt.xlabel('Sample time (A.U.)')
plt.show()