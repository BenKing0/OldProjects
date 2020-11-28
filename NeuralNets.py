import numpy as np
from numpy import random as rnd
from tqdm import tqdm
from scipy import special
import random

class FNN:
    '''A simple deep feed-forward network with updating weights and biases.'''
    def __init__(self,layers,training,test=None,batch_size=10,epochs=100,params=[0.1,0.1],cost_function='sigmoid'):
        '''Layers of type [i/p,h1,h2,...,o/p]. Training data in form ([[x1],[y1]],[[x2],[y2]],...).
        Params of type [alpha,beta].'''
        self.layers = layers
        self.cost_function = cost_function
        self.alpha = params[0]
        self.beta = params[1]
        self.initialise()
        self.error = self.run_network(training,batch_size,epochs)
        if type(test) != type(None):
            test_data = test
            self.output = self.test(test_data)
        
    def initialise(self):
        if self.cost_function == 'sigmoid':
            initialisation = [np.sqrt(2/size) for size in self.layers[:-1]]
        if self.cost_function == 'tanh':
            initialisation = [np.sqrt(1/size) for size in self.layers[:-1]]
        self.weights = [self.f(rnd.normal(size=(self.layers[i+1],self.layers[i]))*initialisation[i]) for i in range(len(self.layers)-1)]
        activations,biases,deltas = [],[],[]
        for i,num in enumerate(self.layers):
            activations.append(np.zeros(num))
            biases.append(np.zeros(num))
            deltas.append(np.zeros(num))
        self.activations,self.biases,self.deltas = np.array(activations),np.array(biases),np.array(deltas)
        return 
    
    def run_network(self,training,batch_size,epochs):
        rnd.shuffle(training)
        # `zip(*tuple)` is the way of unnzipping the tuple to lists:
        training_xs,training_ys = zip(*training)
        t = tqdm(range(epochs),desc='Running network, epoch:')
        error = []
        self.correct = []
        count = 0
        for i in t: # `for each epoch`
            diff_vctr = np.zeros(self.layers[-1])
            for k in range(int(len(training_xs)/batch_size)): # `for each mini-batch result in the training set`  
                indices = random.sample(list(np.arange(len(training_xs))),batch_size)    
                current_batch = np.array([[training_xs[index],training_ys[index]] for index in indices]).T
                for j,input in enumerate(current_batch[0]): # `for each element in the mini-batch`
                    input = np.array(input,dtype='float64')
                    if abs(max(input)) > 1:
                        input = self.f(input)
                    self.feedforward(input)
                    target = current_batch[1][j]
                    diff_vctr += self.activations[-1] - target
                    self.activations[-1][np.argmax(self.activations[-1])] = 1.1
                    self.activations[-1] = np.where(self.activations[-1]==1.1,1,0)
                    if np.where(self.activations[-1]==1)[0][0]==np.where(target==1)[0][0]:
                        count += 1
                        self.correct.append(count)
                diff_vctr = diff_vctr / batch_size
                self.backpropagate(diff_vctr)
                error.append((1/len(self.activations[-1]))*np.sum(self.cost(diff_vctr)))
        return error
    
    def feedforward(self,input):
        n = len(self.layers)
        x = input
        self.activations[0] = x
        for i in range(1,n):
            z_layer = np.matmul(self.weights[i-1],self.activations[i-1]) + self.biases[i]
            self.activations[i] = self.f(z_layer)
        return 
    
    def backpropagate(self,diff_vctr):
        self.deltas[-1] = self.cost_prime(diff_vctr)
        for k in range(1,len(self.layers)):
            dotted = np.multiply(self.deltas[-k],self.f_prime(self.activations[-k]))
            self.deltas[-(k+1)] = np.matmul(self.weights[-k].T,dotted)
        weights_change = []
        for k in range(len(self.layers)):
            if k < len(self.layers) - 1:
                dotted = np.multiply(self.deltas[k+1],self.f_prime(self.activations[k+1]))
                self.activations[k] = np.expand_dims(self.activations[k],axis=1)
                dotted = np.expand_dims(dotted,axis=1)
                weights_change = (np.matmul(dotted,self.activations[k].T))
                self.weights[k] -= self.alpha*weights_change
            sig_prime_layer = np.squeeze(self.f_prime(self.activations[k]))
            biases_change = (np.multiply(self.deltas[k],sig_prime_layer))
            self.biases[k] -= self.beta*biases_change  
        return
    
    def training_results(self):
        return np.array(self.error),np.array(self.correct)
    
    def test(self,test_data):
        t = tqdm(test_data,desc='Test data, epoch:')
        result = []
        self.weights = tuple(self.weights)
        self.biases.flags.writeable = False
        for i,input in enumerate(t):
            input = np.array(input,dtype='float64')
            if abs(max(input)) > 1:
                input = self.f(input)
            self.feedforward(input)
            self.activations[-1][np.argmax(self.activations[-1])] = 1.1
            self.activations[-1] = np.where(self.activations[-1]==1.1,1,0)
            result.append(self.activations[-1])
        return np.array(result)
    
    def test_results(self):
        return self.output
    
    def f(self,x):
        if self.cost_function == 'sigmoid':
            return special.expit(x)
        if self.cost_function == 'tanh':
            return np.tanh(x)
        if self.cost_function == 'ReLU':
            return np.max(x,initial=0)
        else:
            return 'Cost function not programmed.'
    
    def f_prime(self,x):
        if self.cost_function == 'sigmoid':
            return special.expit(x)*(1-special.expit(x))
        if self.cost_function == 'tanh':
            return (1-np.tanh(x)**2)
        if self.cost_function == 'ReLU':
            relu_prime = np.zeros(len(x))
            for i,val in enumerate(x):
                if val > 0:
                    relu_prime[i] = 1
            return relu_prime
        else:
            return 'Cost function not programmed.'
    
    def cost(self,diff_vctr):
        # Mean Squared Error cost function:
        return (1/2)*(diff_vctr)**2
    
    def cost_prime(self,diff_vctr):
        return (diff_vctr)
    
class RNN:
    def __init__(self):
        return
