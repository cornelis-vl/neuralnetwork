# MODULES AND PACKAGES
import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np
import pandas as pd

# FUNCTIONS

class ArtificialNeuralNetwork():
    """
    Author      : Cornelis Vletter
    Date        : 06 May 2016
    Version     : 0.1
    
    Description : A flexible multi-layer neural network model in Python.
    """
    
    def __init__(self, hid_lay=2, num_nodes=[3, 2]):
        self.hid_lay = hid_lay
        self.num_nodes = num_nodes
        self.dim_check = hid_lay == len(num_nodes)
        
        print "Artificial Neural Network initialized with " \
        "{0} hidden layers. Nodes for each layer are {1}," \
        " respectively.".format(hid_lay, num_nodes)
    
    def build_params(self, input_layer, target):
        obs, input_nodes = np.shape(input_layer)
        classes = np.unique(target)
        num_classes = len(classes)
        build_check = self.dim_check
        params = {}
        
        if build_check:
            for h in range(0,self.hid_lay+1):
                if h == 0:
                    in_nodes = input_nodes
                    out_nodes = self.num_nodes[h]

                elif h == self.hid_lay:
                    in_nodes = self.num_nodes[h-1]
                    out_nodes = num_classes

                else:
                    in_nodes = self.num_nodes[h-1]
                    out_nodes = self.num_nodes[h]

                exec "W_{0} = np.random.randn({1}, {2})".format(
                    h,
                    in_nodes,
                    out_nodes)

                exec "b_{0} = np.zeros({1})".format(
                    h,
                    out_nodes)

                exec "params['W_{0}'] = W_{0}".format(h)
                exec "params['b_{0}'] = b_{0}".format(h)
                
            return params
            
        else:
            print "The nodes for each layer are not defined." \
                  "I count {n_lay} layers, but nodes are only defined" \
                  "for {n_nodes} layer.".format(
                    n_lay=self.hid_lay,
                    n_nodes=len(self.num_nodes))


ann = ArtificialNeuralNetwork()
pars = ann.build_params(X,y)

#define labels and input layer size   
    
    
        
# Calculate the loss of your model
def calculate_loss(model):
    """Function to calculate the (log) loss of the in-sample predictions of neural network.
    
    model : dict. Contains the parameters of the model that will be used to make predictions.
          : Currently fit to a neural network with one hidden layer.
    """
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)    
    
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):
     
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
 
    # This is what we return at the end
    model = {}
     
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
 
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
 
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
         
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
         
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, calculate_loss(model))
     
    return model


# Plot decision boundary function
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title("Decision Boundary for hidden layer size 3")
    plt.show()    


# PROGRAM

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)


#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# Parameters
num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
 
# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength
nn_hdim = 3

# Build a model with a 3-dimensional hidden layer
model = build_model(20, print_loss=True)
 
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x), X, y)
 

plt.show()



