import numpy as np
import random
from scipy.signal import resample,welch
import os
import sys
file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

class generateMUAPtrain:
    #This class uses templates from Silvia's model to build emg from MUAP trains.
    #The convolution function used causes edge effects, so to avoid this the 
    #MUAP train simulated is longer than specified by sampleLen, and then
    #cut to the desired length. 
    
    def __init__(self,sampleTime,numMU,fs=2048,chosen_MUs=None): #number of samples of length sampleLen
        np.random.seed(4321)    
        self.sampleLen = int(sampleTime*fs) + 200    #length of sample (samp Freq 4026Hz)
        self.numMU = numMU                  #number of motor units in sim
        self.lambdas = np.random.randint(25,30,numMU)/fs #firing rate of motor units
        self.fs = fs # define sampling frequency (default 2048 Hz)
        #load simulated MUAP templates: - 1806 MUs Max.
        templates = np.load('templates.npy')[1000:,:,:,:]
        locations = np.load('fibreLocs.npy')[1000:,:]
        distance = np.sqrt(locations[:,0]**2 + locations[:,1]**2)
        templates = templates[np.argsort(distance),:,:,:]
        # Taken last 50 templates to ensure consistent amplitudes (only largest ones)
        #template_peaks = np.max(np.mean(np.sum(abs(templates),axis=3),axis=2),axis=1)
        #templates = templates[np.argsort(template_peaks),:,:,:]
        templates = templates[-50:,:,:,:]
        if chosen_MUs is None:
            print('Shuffling')
            # Shuffle the motor unit axis of `templates` to randomise MU selection (pick different templates):
            np.random.shuffle(templates)
            templates = templates[1:(numMU+1),:,:,:]
        else:
            print('Not shuffling')
            ks = chosen_MUs
            templates = np.array([templates[k] for k in ks])
        self.numSensors = np.shape(templates)[1]*np.shape(templates)[2]
        templates = np.transpose(templates,(0,2,1,3))
        reshapeTemplates = np.zeros([np.shape(templates)[0],self.numSensors,np.shape(templates)[3]])
        count = 0
        for i in range(np.shape(templates)[1]):
            for j in range(np.shape(templates)[2]):
                reshapeTemplates[:,count,:] = templates[:,i,j,:]
                count+=1
        # templates of shape: [N (of 50) MUs, 192 channels, resampled shape]
        self.templates = reshapeTemplates
        self.tempMax = np.trapz(np.abs(self.templates),axis=1)
            
    #nextSequence finds the timestamps of MU activations (using a poisson distribution)
    def nextSequence(self,lam):
        timeStamps = np.zeros(1,dtype=int)
        while timeStamps[-1] <= self.sampleLen:
            nextSpike = int(random.expovariate(lam))
            while nextSpike < 70 or nextSpike > 500: nextSpike = int(random.expovariate(lam))
            timeStamps = np.append(timeStamps,timeStamps[-1] + nextSpike)
        return timeStamps[1:-1]    
    
    #spikeTrain converts the timestamps into a binary time series MUAP train
    def spikeTrain(self):
        spikes = np.zeros((self.numMU,self.sampleLen))
        for i,j in enumerate(self.lambdas):
            spikes[i,self.nextSequence(j)-1] = 1
        return spikes
            
    #genSpikes uses the spikeTrain function to build an array of 
    #MU activation trains of size and shape specified on initialisation
    def genSpikes(self):
        MUtrain = np.zeros([self.sampleLen,self.numMU])
        MUtrain = np.transpose(self.spikeTrain())
        self.MUtrain = MUtrain
        
    #addNoise calculates and adds gaussian noise of desired SNR
    def addNoise(self,EMGsamples,SNR):
        data = np.transpose(EMGsamples)
        NoisyEMGsamples = np.zeros(np.shape(EMGsamples))
        for i in range(np.shape(data)[0]):
            sigWelch = welch(data[i,:],self.fs)
            # ... self.fs is the sampling frequency.
            sigPower = np.sum(sigWelch[1]) * (sigWelch[0][1] - sigWelch[0][0])
            # convert from decibels (SNR) to power ratio:
            noisePowerTarget = sigPower/(10**((SNR)/10))
            noise = np.random.normal(0,np.sqrt(noisePowerTarget),np.shape(data)[1])
            noiseWelch = welch(noise,self.fs)
            noisePower = np.sum(noiseWelch[1]) * (noiseWelch[0][1] - noiseWelch[0][0])
            realSNR = 10*np.log10(sigPower/noisePower)
            if (realSNR - SNR) > 1: 
                print("SNR warning")
                print(realSNR)
            NoisyEMGsamples[:,i] = EMGsamples[:,i] + np.transpose(noise)
        return NoisyEMGsamples
        
    #initSim builds the EMG from the MU activation trains by convolving
    #the binary MU trains with the templates 
    def initSim(self):
        self.genSpikes() 
        EMGsamples = np.zeros([self.sampleLen,self.numSensors])
        for k in range(self.numSensors):
                emg = np.zeros([self.numMU,self.sampleLen])
                for j in range(self.numMU):
                    train = np.squeeze(self.MUtrain[:,j])
                    template = np.squeeze(self.templates[j,k,:])
                    template = resample(template,128)
                    emg[j,:] = np.convolve(train,template,'same')
                emg = np.sum(emg,axis=0,keepdims=True)
                EMGsamples[:,k] = np.squeeze(np.transpose(emg))
        self.MUtrain = self.MUtrain[100:-100,:]
        self.EMGsamples = EMGsamples[100:-100,:]

    def returnOutput(self): return self.MUtrain
    
    def returnInput(self,SNR): return self.addNoise(self.EMGsamples,SNR)
    
    def return_templates(self):
        return self.templates
