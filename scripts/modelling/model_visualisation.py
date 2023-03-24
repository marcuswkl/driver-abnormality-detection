from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import random
from sklearn import tree


# Visualise a random decision tree in a random forest
def visualise_dt(rfc, feature_names, target_names):
    # Prepare the figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    # Randomly select a tree
    random_tree_index = random.randint(0, len(rfc.estimators_)-1)
    
    # Plot the tree
    tree.plot_tree(rfc.estimators_[random_tree_index], feature_names=feature_names, class_names=target_names)

# Visualise an artificial neural network (ANN)
def visualise_ann(mlp):
    # Get number of neurons in each layer
    n_neurons = [len(layer) for layer in mlp.coefs_]
    n_neurons.append(mlp.n_outputs_)
    
    # Calculate the coordinates of each neuron on the graph
    y_range = [0, max(n_neurons)]
    x_range = [0, len(n_neurons)]
    loc_neurons = [[[l, (n+1)*(y_range[1]/(layer+1))] for n in range(layer)] for l,layer in enumerate(n_neurons)]
    x_neurons = [x for layer in loc_neurons for x,y in layer]
    y_neurons = [y for layer in loc_neurons for x,y in layer]
    
    # Identify the range of weights
    weight_range = [min([layer.min() for layer in mlp.coefs_]), max([layer.max() for layer in mlp.coefs_])]
    
    # Prepare the figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    # Draw the neurons
    ax.scatter(x_neurons, y_neurons, s=100, zorder=5)
    
    # Draw the connections with line width corresponds to the weight of the connection
    for l,layer in enumerate(mlp.coefs_):
      for i,neuron in enumerate(layer):
        for j,w in enumerate(neuron):
          ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'white', linewidth=((w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)*1.2)
          ax.plot([loc_neurons[l][i][0], loc_neurons[l+1][j][0]], [loc_neurons[l][i][1], loc_neurons[l+1][j][1]], 'grey', linewidth=(w-weight_range[0])/(weight_range[1]-weight_range[0])*5+0.2)