import numpy as np

class NeuralNet:
    '''A simple neural network with a hidden layer. ReLU activation
    and classifies with softmax.'''
    def __init__(self, W1, b1, W2, b2):
        '''`self.dims = (M, D, C)` stores the layers' dimensions
        `self.params` will store all the weights
        W1 = (M, D), b1 = (D)
        W2 = (D, C), b2 = (C)'''
        # input_dim, hidden_dim, output_dim
        self.dims = *W1.shape, *b2.shape
        self.params = {
            'W1': W1,   'b1': b1,
            'W2': W2,   'b2': b2
        }

    def create_random(input_dim: int, hidden_dim: int, output_dim: int,
                 weight_scale=1):
        '''Weight matrices are initialized from a normal distribution
        scaled by `weight_scale`. Biases are set to zero.'''
        W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)

        W2 = np.random.normal(0, weight_scale, (hidden_dim, output_dim))
        b2 = np.zeros(output_dim)
        return NeuralNet(W1, b1, W2, b2)

    def predict(self, X):
        '''Computes a forward pass of the neural net.
        Accepts `X` = (N, M) and returns a (N, C) np.array'''
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        h1 = X @ W1 + b1
        z1 = np.maximum(0, h1)
        h2 = z1 @ W2 + b2

        # binary classification if 1 output neuron
        if self.dims[2] == 1:
            return h2 > 0

        # softmax
        f = h2 - np.max(h2, axis=1)     # for precision + overflow error
        exp_f = np.exp(f)
        z2 = exp_f / np.sum(exp_f, axis=1)

        return z2


    ############################################
    ###     Genetic Algorithms               ###
    ############################################
    def reproduce(self, nn,
                babies_count: int, mutation_scale=1e-1):
        '''Inputs a `NeuralNet` model mate and reproduces'''
        nn1, nn2 = self, nn

        babies = []
        for _ in range(babies_count):
            baby_params = {}

            for p in nn1.params.keys():
                # crossover
                baby_params[p] = NeuralNet.crossover(nn1.params[p], nn2.params[p])

                # mutation
                baby_params[p] += np.random.normal(0, mutation_scale, baby_params[p].shape)
            
            babies.append(NeuralNet(**baby_params))
        return babies

    def crossover(a, b):
        '''performs naive half-vectorized crossover across two np.array of same size'''        
        idx = np.random.randint(0, 2, (*a.shape,))
        parents = np.stack((a, b), axis=-1)
        if len(a.shape) == 1:
            return parents[np.arange(a.shape[0]), idx]
        else:
            s1, s2 = a.shape
            return parents[
                np.tile(np.arange(s1).reshape(s1, 1), (1, s2)).flatten(),
                np.tile(np.arange(s2), (1, s1))[0],
                idx.flatten()
            ].reshape(s1, s2)
    