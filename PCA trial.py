import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

x = np.linspace(0,100,101)
y = x + np.random.normal(loc=0,scale=10,size=(len(x),))
data = np.stack((x,y),axis=1) # (#points, #sources)

# Centre the data:
mus = [np.mean(data.T[i]) for i in range(data.shape[1])]
data = np.array([(point-mus) for point in data]) # (#points, #sources)

# Calculate eigenvalues and eigenvectors of covariance matrix:- cov shape (#sources, #points)
cov = np.cov(data.T)
eigenvals, eigenvects = la.eig(cov)
eigenvals = np.flip(np.sort(eigenvals)) # sort eigenvalues into descending order
eigenvects = eigenvects.T # column is eigenvector - transpose to row
eigenvects = eigenvects[np.argsort(eigenvals)] # sort eigenvects to eigenvalue order
assert np.dot(eigenvects[0],eigenvects[1]) == 0, 'Error: Bases not orthogonal.'
assert eigenvals.all() > 0
if eigenvects[0].all() < 0:
    print('done')
    eigenvects = eigenvects * -1

# Turn eigenvectors into line gradients - for plotting:
m = [eigenvects[i][1]/eigenvects[i][0] for i in range(len(eigenvects))]
basis_ys = np.array([m[i]*data.T[0] for i in range(len(m))])

# Decide how many PCs below a threshold (= 0.95 here):
sum_eigenval = 0
included_PCs,included_eigenvals = [],[]
for n,eigenval in enumerate(eigenvals):
    if sum_eigenval < 0.95*(np.sum(eigenvals)):
        included_eigenvals.append(eigenval)
        included_PCs.append(eigenvects[n])
        sum_eigenval += eigenval
included_PCs,included_eigenvals = np.array(included_PCs),np.array(included_eigenvals)
print('Dimensionality reduced from {0} to {1} dimensions.'.format(len(eigenvects),len(included_PCs)))
    
# Projection of each point to the PCs:
eigenvects = eigenvects.T # change eigenvectors back into transformation matrix form
new_basis = eigenvects # define the new basis as the transformation matrix (transforming from cartesian)
rot_data = np.array([np.matmul(new_basis,dpoint) for dpoint in data]) # (#points, #sources)

# Remove redundant commponents:
reduced_rot_data = rot_data.copy()
for n,eigenval in enumerate(eigenvals):
    if eigenval not in included_eigenvals:
        reduced_rot_data[:,n] = 0
    
# Plot the results:
data = data.T # (#sources, #points)
fig = plt.figure()
fig.add_axes((0,0,1.2,1))
plt.title('Original dataset')
plt.plot(data[0],data[1],'k.')
plt.plot(np.linspace(min(data[0]),max(data[0]),100),0*np.linspace(min(data[0]),max(data[0]),100),'k-')
plt.plot(0*np.linspace(min(data[1]),max(data[1]),100),np.linspace(min(data[1]),max(data[1]),100),'k-')
fig.add_axes((1.3,0,1.2,1))
plt.title('PCs projected on-top')
plt.plot(data[0],data[1],'k.')
plt.plot(data[0],basis_ys[0],'r--',label=r'1$^{st}$ principal component')
plt.plot(data[0],basis_ys[1],'b--',label=r'2$^{nd}$ principal component')
plt.legend(loc='lower right')
fig.add_axes((0,-1.1,1.2,1))
rot_data = rot_data.T # (#sources, #points)
string1 = r'{0:.1f}% of $\sigma^2$'.format(100*(abs(eigenvals[0]))/np.sum(abs(eigenvals)))
string2 = r'{0:.1f}% of $\sigma^2$'.format(100*(abs(eigenvals[1]))/np.sum(abs(eigenvals)))
plt.plot(rot_data[0],rot_data[1],'k.')
plt.plot(np.linspace(min(rot_data[0]),max(rot_data[0]),100),0*np.linspace(min(rot_data[0]),max(rot_data[0]),100),'r--',label=string1)
plt.plot(0*np.linspace(min(rot_data[1]),max(rot_data[1]),100),np.linspace(min(rot_data[1]),max(rot_data[1]),100),'b--',label=string2)
plt.legend(loc='lower right')
plt.xlabel('Rotated to PC\'s basis')
fig.add_axes((1.3,-1.1,1.2,1))
reduced_rot_data = reduced_rot_data.T
plt.plot(reduced_rot_data[0],reduced_rot_data[1],'k.')
plt.plot(np.linspace(min(reduced_rot_data[0]),max(reduced_rot_data[0]),100),0*np.linspace(min(reduced_rot_data[0]),max(reduced_rot_data[0]),100),'r-')
plt.plot(0*np.linspace(min(reduced_rot_data[1]),max(reduced_rot_data[1]),100),np.linspace(min(reduced_rot_data[1]),max(reduced_rot_data[1]),100),'b-')
plt.xlabel('Reduced data projection')
plt.show()