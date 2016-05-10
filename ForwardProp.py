import numpy as np

#forward propagation
nrows, ncols = np.shape(X)
ncols += 1
arc = 1
nodes = 2
labels = 2

np.zeros(nrows)

X = pd.DataFrame(X)

a1 = np.ones((nrows, ncols))
a1[:,1:ncols] = X
theta1 = np.random.randn(ncols, nodes)
theta2 = np.random.randn(nodes+1, labels)


z2 = np.dot(a1, theta1)
a2 = np.ones((nrows, nodes+1))
a2[:,1:nodes+1] = np.tanh(z2)

z3 = np.dot(a2, theta2)
a3 = np.exp(z3)
probs = a3 / np.sum(a3, axis=1, keepdims=True)






#backward propagation



    


