import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pprint import pprint
import sys
sys.path.append('C:/Users/bened/Documents/University/ICL Project/Papers/Codes')
from entropies import PE
import adaptive_threshold as at
import numpy as np

name = ('aapl').upper()
ticker = yf.Ticker(name) 

hist = ticker.history('1d',interval='1m')
keys = hist.keys()
abs_error = hist['High']-hist['Low']
highs = hist['High']
lows = hist['Low']

gamma = 3

average = np.mean(np.array(list(zip(highs,lows))).T,axis=0)
PE_series = PE(average,bin_width=3,normalised=True,window_size=20).results()
indices = at.below(PE_series,gamma=gamma,window_size=10,influence=1)

fig = plt.figure()
fig.add_axes((0,0,2,1))
plt.title(r'Stock Value and Permutation Entropy of {0} Ticker'.format(name))
plt.plot(np.arange(len(highs))/60,highs,label='Highest value')
plt.plot(np.arange(len(lows))/60,lows,label='Lowest value')
plt.scatter(indices/60,highs[indices])
plt.scatter(indices/60,lows[indices])
plt.ylabel('Value ($)')
plt.legend(loc='lower right')
plt.xticks([])
fig.add_axes((0,-1.05,2,1))
plt.plot(np.arange(len(PE_series))/60,PE_series)
plt.scatter(indices/60,PE_series[indices],label='$\gamma$={}'.format(gamma))
plt.legend()
plt.ylabel('PE (normalised)')
plt.xlabel('Hours to present')
plt.show()

'''
import time

pause = 2 # seconds
name = ('aapl').upper()
ticker = yf.Ticker(name) 

while 1:
    hist = ticker.history('1d',interval='1m')
    keys = hist.keys()
    current_high = hist['High'][-1]
    current_low = hist['Low'][-1]
    abs_error = current_high - current_low
    print(current_high)
    time.sleep(pause)
    
'''    
'''
information = ticker.info

hist = ticker.history()

df_download = yf.download(name, start='YYYY-MM-DD')

'''
