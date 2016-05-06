import numpy as np

#forward propagation
nrows, ncols = np.shape(X)
ncols += 1
arc = 1
nodes = 2

np.zeros(nrows)

X = pd.DataFrame(X)

a1 = np.ones((nrows, ncols))
a1[:,1:ncols] = X
theta1 = np.random.randn(ncols, nodes)

z2 = np.dot(a1, theta1)
a2 = np.tanh(z2)

    


