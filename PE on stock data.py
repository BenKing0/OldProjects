import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pprint import pprint
import sys
sys.path.append('C:/Users/bened/Documents/University/ICL Project/Papers/Codes')
from entropies import PE
import adaptive_threshold as at
import numpy as np
from Bonato import sumSquares

name = ('aapl').upper()
ticker = yf.Ticker(name) 
days = 20
mins = 2

hist = ticker.history('{}d'.format(days),interval='{}m'.format(mins))
keys = hist.keys()
abs_error = hist['High']-hist['Low']
highs = hist['High']
lows = hist['Low']

gamma = 5
window = 200
lag = 2

average = np.mean(np.array(list(zip(highs,lows))).T,axis=0)
PE_series = PE(average,bin_width=3,normalised=True,window_size=window).results()
adjacent_squared = sumSquares(PE_series,lag=lag)()
indices = at.below(adjacent_squared,gamma=gamma,window_size=10,influence=1)
indices *= lag
indices = np.array(indices)
shift = len(average) - len(PE_series)

fig = plt.figure()
fig.add_axes((0,0,2,1))
plt.title(r'Stock Value and Permutation Entropy of {0} Ticker'.format(name))
plt.plot(np.arange(len(highs))/60*mins,highs,label='Highest value')
plt.plot(np.arange(len(lows))/60*mins,lows,label='Lowest value')
plt.scatter((indices+shift)/60*mins,highs[indices+shift],color='k')
plt.scatter((indices+shift)/60*mins,lows[indices+shift],color='k')
plt.ylabel('Value ($)')
plt.legend(loc='lower right')
plt.xlim(-1,1.05*max((indices+shift)/60*mins))
plt.xticks([])
fig.add_axes((0,-1.05,2,1))
plt.plot((np.arange(len(PE_series))+shift)/60*mins,PE_series,alpha=0.6)
plt.scatter((indices+shift)/60*mins,PE_series[indices],label='$\gamma$={}'.format(gamma),color='r',alpha=1)
plt.legend()
plt.xlim(-1,1.05*max((indices+shift)/60*mins))
plt.ylabel('PE (normalised)')
plt.xlabel('Hours to present')
plt.show()

#%%
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
information = ticker.info

hist = ticker.history()

df_download = yf.download(name, start='YYYY-MM-DD')

'''