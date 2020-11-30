import random
import numpy as np
from scipy import integrate
from tqdm import tqdm

Z = 2
e = (1.602e-19 * np.sqrt(1/(4*np.pi*8.854e-12))) # J
e_sq_raw = e**2
e_sq = (e**2 / 1.6e-19) * 1e10 #14.4 # eV A
hbar_raw = (6.63e-34 / (2*np.pi))
hbar = hbar_raw * 1e10 # A
pi = np.pi
m_e_raw = 9.11e-31 # kg 
m_e = m_e_raw * 1e10 # kg A
r0 = (hbar_raw**2 / (e_sq_raw*m_e_raw)) * 1e10 # 0.5297 # A, hbar**2/m_e*e**2

upper = 300 # change this when test-function requires a higher upper-bound

class VMC:
    def __init__(self,j,k,m,chis,Z=Z,bounds=[0,upper],num_samples=300):
        self.E_min = self.minimise(chis,j,k,m,Z,bounds,num_samples)
        
    def minimise(self,chis,j,k,m,Z,bounds,num_samples):
        E_Ts,var_E_Ls = [],[]
        for i,chi in enumerate(tqdm(chis)):
            self.norm = self.find_norm(j,k,m,chi,bounds,Z)
            E_Ts.append(self.MC_importance_sampling(self.local_energy,self.probability_density,j,k,m,num_samples,bounds,chi)[0])
            var_E_Ls.append(self.MC_importance_sampling(self.local_energy,self.probability_density,j,k,m,num_samples,bounds,chi)[1])
        self.E_Ts_for_plotting = E_Ts
        self.var_E_Ls_for_plotting = var_E_Ls
        E_minimum = min(E_Ts)
        return E_minimum
        
    def test_function(self,r1,r2,j,k,m,chi,Z): # !! find a better test function
        return np.exp(-chi*r1**2)

    def mod_sq_function(self,r1,r2,j,k,m,chi,Z): # !! find a better test function
        return np.exp(-2*chi*r1**2)

    def transformation_method(self,function,num_samples,bounds,j,k,m,chi,Z):
        sample_r1 = np.arange(bounds[0],bounds[1],(bounds[1]-bounds[0])/100)
        sample_r2 = np.arange(bounds[0],bounds[1],(bounds[1]-bounds[0])/100)
        M = max([function(self.mod_sq_function,el_r1,el_r2,j,k,m,chi,Z) for el_r1,el_r2 in zip(sample_r1,sample_r2)])
        print('\nMaximum distribution value: {:.2f}'.format(M))
        count,total = 0,0
        sample = []
        while count < num_samples:
            total += 1
            r = random.random()
            r1 = bounds[0] + (bounds[1] - bounds[0]) * random.random() # !! CHANGE FROM UNIFORM SAMPLING?
            r2 = bounds[0] + (bounds[1] - bounds[0]) * random.random()
            if r * M < function(self.mod_sq_function,r1,r2,j,k,m,chi,Z):
                count += 1
                sample.append([r1,r2,function(self.mod_sq_function,r1,r2,j,k,m,chi,Z)])
        hit = count/total
        print('\nPercentage hit samples: {:.2f}%'.format(hit))
        return np.array(sample)

    def find_norm(self,j,k,m,chi,bounds,Z):
        integrand = self.mod_sq_function
        def r1int(r2,j,k,m,int_max,chi):
            return integrate.quad(integrand,0,int_max,args=(r2,j,k,m,chi,Z))[0]
        norm,err = integrate.quad(lambda r2: r1int(r2,j,k,m,bounds[1],chi),bounds[0],bounds[1])
        #assert np.isclose(norm,integrate.nquad(integrand, [[0,bounds[1]],[0,bounds[1]]], args=(j,k,m,chi))[0])
        return norm
    
    def probability_density(self,mod_sq_function,r1,r2,j,k,m,chi,Z):
        return (1/self.norm) * mod_sq_function(r1,r2,j,k,m,chi,Z)

    def acting_hamiltonian(self,r1,r2,j,k,m,chi): # change for every structurally different test function, find analytically
        return # !! ... Do the maths, but find a better test function first

    def local_energy(self,acting_hamiltonian,test_function,r1,r2,j,k,m,chi):
        return acting_hamiltonian(r1,r2,j,k,m,chi) / test_function(r1,r2,j,k,m,chi,Z)

    def MC_importance_sampling(self,operator_func,dist_func,j,k,m,num_samples,bounds,chi):
        r1s,r2s = self.transformation_method(dist_func,num_samples,bounds,j,k,m,chi,Z).T[:2]
        self.r1s_for_plotting,self.r2s_for_plotting = r1s,r2s
        E_Ls = [self.local_energy(self.acting_hamiltonian,self.test_function,r1,r2,j,k,m,chi) for r1,r2 in zip(r1s,r2s)]
        E_test = (1/len(E_Ls)) * sum(E_Ls)
        var_E_Ls = (1/len(E_Ls)) * sum(map(lambda x: x**2, E_Ls)) - E_test**2 # change to numpy if speed-up required
        return E_test,var_E_Ls
    
    def __call__(self):
        return self.E_min
    
    def return_for_plotting(self):
        return self.E_Ts_for_plotting,self.var_E_Ls_for_plotting,self.r1s_for_plotting,self.r2s_for_plotting
  

#%% - Replacements to trial function with a simple harmonic oscillator - working as of 29/11/2020
'''
return (r1+r2)**j * (r1-r2)**k * np.exp(-(Z/(r0*chi))*(r1+r2))

return (r1+r2)**(2*j) * (r1-r2)**(2*k) * np.exp(-(2*Z/(r0*chi))*(r1+r2))

w = 1
return -(1/2) * 2 * chi * (2*chi*r1**2-1) * np.exp(-chi*r1**2) + (w * r1**2 / 2) * np.exp(-chi*r1**2)

r2 = 0
norm = integrate.quad(integrand,0,100,args=(r2,j,k,m,chi,Z))[0]
return norm * 2 # just for trial hamiltonian, as r1 can be negative
'''
#%%
results,variances = VMC(0,0,0,np.arange(0.1,1,0.1),1).return_for_plotting()[:2]
print(results)
print(np.arange(0.1,1,0.1)[results.index(min(results))])
#%%
print(variances)
#%%
import matplotlib.pyplot as plt
plt.plot(np.arange(0.1,1,0.1),results)
plt.show()
plt.plot(np.arange(0.1,1,0.1),variances)
plt.show()
