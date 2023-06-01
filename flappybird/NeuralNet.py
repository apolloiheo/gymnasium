import numpy as np

class NeuralNet:
    '''A simple neural network with a hidden layer. ReLU activation
    and classifies with softmax.'''
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 weight_scale=1e-3):
        '''Weight matrices are initialized from a normal distribution
        scaled by `weight_scale`. Biases are set to zero.
        
        `self.params` will store all the weights'''
        self.dims = input_dim, hidden_dim, output_dim

        self.params = {}
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, output_dim))
        self.params['b2'] = np.zeros(output_dim)