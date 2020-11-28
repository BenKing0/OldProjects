import numpy as np
from tqdm import tqdm
import tensorflow as tf

class LempelZiv:
    def __init__(self,x,threshold=None,window_size=100,normalised=True,window_overlap=1,disable_bar=False):
        self.x,self.window_size,self.disable,self.normalised,self.window_overlap = np.array(x),window_size,disable_bar,normalised,window_overlap
        if threshold is None:
            threshold = np.median(self.x)
        self.normalisation_const = window_size/np.log2(window_size)
        binary_sequence = self.to_binary(threshold)
        self.complexity_per_win = self.complexity(binary_sequence)
        
    def to_binary(self,threshold):
        binary_x = np.where(self.x >= threshold,np.ones((len(self.x))),np.zeros((len(self.x))))
        return binary_x
        
    def complexity(self,binary_sequence):
        window = tf.data.Dataset.from_tensor_slices(binary_sequence)
        window = window.window(self.window_size,shift=self.window_overlap,drop_remainder=True)
        window = window.flat_map(lambda f: f.batch(self.window_size))
        complexity_per_win = []
        for vctr in tqdm(window,total=int((len(self.x)+1-self.window_size)/self.window_overlap),disable=self.disable):
            vctr = vctr.numpy().tolist()
            seen = [[vctr[0]]]
            i = 1
            repeated_end_value = False
            while i <= len(vctr)-1:
                k = 1
                while any([vctr[i:i+k] == item for item in seen]) == True:
                    if i+k > len(vctr):
                        repeated_end_value = True
                        break # breaks from the `repeated end values, infinite loop` problem.
                    else: 
                        k += 1
                if repeated_end_value != True:
                    seen.append(vctr[i:i+k])
                i += k
            complexity_per_win.append(len(seen))
        complexity_array = np.array(complexity_per_win)
        if self.normalised == True:
            complexity_array = complexity_array/self.normalisation_const
        return complexity_array
    
    def results(self):
        return self.complexity_per_win