"""
@file   network.py
@author Hayat Rajani  [hayatrajani@gmail.com]

November 04, 2019
"""

import numpy as np
import math
import time
import copy


class Network:
    """
    Implements a multilayer perceptron.
    """

    def __init__(self, size, activations, lr=0.01, beta1=0.9, beta2=0.99, drop=None):
        """
        Initializes the network's parameters.

        Args:
            size: A list containing the number of neurons in each layer.
            activations: A list of activation functions for each layer except the input layer.
            lr: Learning rate.
            beta1: Exponential decay for first moment estimates.
            beta2: Exponential decay for second moment estimates.
            drop: Probability of dropping a neuron.
        """
        self.size = size
        self.layers = len(size)
        self.activations = activations
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, 1e-8
        self.drop = drop
        # He initialization for ReLU and Xavier for tanh
        np.random.seed(11)
        self.weights = [None]*(self.layers-1)
        for i in range(self.layers - 1):
            if self.activations[i].lower() == 'relu':
                self.weights[i] = np.random.randn(
                    self.size[i], self.size[i + 1]) * math.sqrt(2 / self.size[i])
            elif self.activations[i].lower() == 'tanh':
                self.weights[i] = np.random.randn(
                    self.size[i], self.size[i+1])*math.sqrt(1/self.size[i])
            else:
                self.weights[i] = np.random.randn(
                    self.size[i], self.size[i+1])*math.sqrt(2/(self.size[i]+self.size[i+1]))
        self.bias = [np.random.randn(m, 1) for m in size[1:]]

    def load_weights(self, path):
        """
        Reads weights from a file.

        Args:
            path: Path of the file to read weights from.
        """
        i = 0
        n = 0
        j = 0
        for line in open(path):
            if line[0] == '#':
                self.size = [int(x) for x in line[1:].split()]
                self.weights = [np.zeros((n, m)) for n, m in zip(
                    self.size[:-1], self.size[1:])]
                [None] * (self.layers - 1)
            else:
                if j != -1:
                    self.bias[j] = np.array(
                        line.split(), dtype=np.float).reshape(self.size[j+1], 1)
                    j += 1
                    if j == self.layers-1:
                        j = -1
                else:
                    self.weights[i][n, :] = np.array(
                        line.split(), dtype=np.float)
                    n += 1
                    if i < self.layers-1 and n == self.size[i]:
                        i += 1
                        n = 0
        print('Weights loaded successfully!')

    def save_weights(self):
        """
        Saves weights to a file named 'weights.dat' in the current directory.
        """
        fout = open('weights.dat', 'w')
        fout.write('# ' + ' '.join(str(x) for x in self.size) + '\n')
        for b in self.bias:
            fout.write(np.array2string(b.T, max_line_width=np.inf)
                       [2:-2].strip() + '\n')
        for s, w in zip(self.size[:-1], self.weights):
            for n in range(s):
                fout.write(np.array2string(
                    w[n, :], max_line_width=np.inf)[1:-1].strip()+'\n')
        fout.close()
        print("Weights saved to file: 'weights.dat'")

    def make_mini_batch(self, P, inputs, _outputs, B=4):
        """
        Creates random mini batches for training.

        Args:
            P: Number of training patterns.
            inputs: A NxP matrix of input per pattern.
            _outputs: A MxP matrix of expected (correct) output per pattern.
            B: Number of training patterns per mini batch.

        Returns:
            A list of tuples of the form (input_batch, _output_batch)
            where,
                input_batch is an NxB matrix of inputs
                _output_batch is an MxB matrix of corresponding expected (correct) outputs
        """
        P_ = list(np.random.permutation(P))
        inputs = inputs[:, P_]
        _outputs = _outputs[:, P_]
        batches = math.floor(P/B)
        mini_batches = [None]*batches
        for batch in range(batches):
            mini_batches[batch] = (
                inputs[:, batch*B:(batch+1)*B], _outputs[:, batch*B:(batch+1)*B])
        if P % B != 0:
            mini_batches.append(
                (inputs[:, batches*B:], _outputs[:, batches*B:]))
        return mini_batches

    def predict(self, input, backprop=False):
        """
        Predicts an output for the supplied pattern(s).

        Args:
            input: An NxP matrix; P == #patterns

        Returns:
            input: An MxP matrix; P == #patterns
        """
        if len(input.shape) == 1:
            input = input.reshape(self.size[0], 1)
        if not backprop:
            for fkt, w, b in zip(self.activations, self.weights, self.bias):
                input = activate(fkt, np.matmul(w.T, input)+b)
            return input
        else:
            outputs = [None]*self.layers
            outputs[0] = input
            for i, fkt, w, b in zip(range(1, self.layers), self.activations, self.weights, self.bias):
                outputs[i] = activate(fkt, np.matmul(w.T, outputs[i-1])+b)
            return outputs[1:]

    def train(self, P, inputs, _outputs, epochs=1000):
        """
        Learns the MLP's weights using Backpropagation.

        Args:
            P: Number of training patterns.
            inputs: A NxP matrix of input per pattern.
            _outputs: A MxP matrix of expected (correct) output per pattern.
        """
        print('Training...')
        since = time.time()
        error = np.zeros(epochs)
        best_wt, best_bi = None, None
        best_error = np.inf
        no_improve = 0
        threshold = 1e-6
        batch_size = 4
        m_w = [np.zeros((n, m)) for n, m in zip(self.size[:-1], self.size[1:])]
        m_b = [np.zeros((m, 1)) for m in self.size[1:]]
        v_w = [np.zeros((n, m)) for n, m in zip(self.size[:-1], self.size[1:])]
        v_b = [np.zeros((m, 1)) for m in self.size[1:]]
        for epoch in range(epochs):
            print('Epoch', epoch + 1)
            for mini_batch in self.make_mini_batch(P, inputs, _outputs, batch_size):
                input_batch, _output_batch = mini_batch
                outputs = self.predict(input_batch, True)
                delta = [None] * (self.layers - 1)
                if self.drop:
                    drop_mask = [np.random.binomial(
                        1, 1-self.drop, (m, input_batch.shape[1]))/(1-self.drop) for m in self.size[1:-1]]
                    for i in range(self.layers - 2):
                        outputs[i] *= drop_mask[i]
                for i, fkt in zip(range(self.layers-2, -1, -1), self.activations[-1::-1]):
                    # for output layer; -2 since zero-indexed
                    if i == self.layers-2:
                        delta[i] = (_output_batch-outputs[i]) * \
                            activate_(fkt, outputs[i])
                    # for hidden layers
                    else:
                        delta[i] = np.matmul(
                            self.weights[i+1], delta[i+1])*activate_(fkt, outputs[i])
                for i, d in zip(range(self.layers-1), delta):
                    # for first hidden layer
                    if i == 0:
                        m_w[i] = self.beta1 * m_w[i] + (1-self.beta1)*np.sum(
                            np.einsum('np,mp->pnm', input_batch, d), axis=0)
                        v_w[i] = self.beta2 * v_w[i] + (1-self.beta2)*np.square(
                            np.sum(np.einsum('np,mp->pnm', input_batch, d), axis=0))
                    # for all subsequent layers
                    else:
                        m_w[i] = self.beta1 * m_w[i] + (1-self.beta1)*np.sum(
                            np.einsum('np,mp->pnm', outputs[i-1], d), axis=0)
                        v_w[i] = self.beta2 * v_w[i] + (1-self.beta2)*np.square(
                            np.sum(np.einsum('np,mp->pnm', outputs[i-1], d), axis=0))
                    m_w_ = m_w[i]/(1-self.beta1**(i+1))
                    v_w_ = v_w[i]/(1-self.beta2**(i+1))
                    m_b[i] = self.beta1 * m_b[i] + \
                        (1-self.beta1)*np.sum(d, axis=1, keepdims=True)
                    v_b[i] = self.beta2 * v_b[i] + \
                        (1-self.beta2)*np.square(np.sum(d, axis=1, keepdims=True))
                    m_b_ = m_b[i]/(1-self.beta1**(i+1))
                    v_b_ = v_b[i]/(1-self.beta2**(i+1))
                    self.weights[i] += self.lr*m_w_/(np.sqrt(v_w_)+self.eps)
                    self.bias[i] += self.lr*m_b_/(np.sqrt(v_b_)+self.eps)
                # average error per epoch
                error[epoch] += 0.5 * \
                    np.sum((_output_batch-outputs[-1])**2)/batch_size
            elapsed_time = time.time()-since
            print('\tError:', abs(best_error-error[epoch]))
            print('\tTime Elapsed: {:.0f}m {:.0f}s'.format(
                elapsed_time//60, elapsed_time % 60))
            if abs(best_error-error[epoch]) <= threshold:
                no_improve += 1
                if no_improve == 10:
                    self.bias = copy.deepcopy(best_bi)
                    self.weights = copy.deepcopy(best_wt)
                    print('Training Complete!')
                    self.save_weights()
                    return error[:epoch]
            else:
                best_error = error[epoch]
                best_bi = copy.deepcopy(self.bias)
                best_wt = copy.deepcopy(self.weights)
        self.save_weights()
        return error

    def evaluate(self, P, inputs, outputs, threshold):
        """
        Evaluates the performance of the MLP.

        Args:
            P: Number of training patterns.
            inputs: A NxP matrix of input per pattern.
            outputs: A MxP matrix of expected (correct) output per pattern.
        """
        count = 0
        for p in range(P):
            prediction = self.predict(inputs[:, p])
            count += 1 if (abs(outputs[:, p]-prediction)
                           <= threshold).all() else 0
        score = count/P
        print('Score:', score)


def activate(fkt, z):
    """
    Implements the activation funtion.

    Args:
        fkt: The selected activation function.
        z: The result of the weighted sum.
    """
    if fkt.lower() == 'relu':
        return np.maximum(0, z)
    elif fkt.lower() == 'tanh':
        return np.tanh(z)
    elif fkt.lower() == 'logistic':
        return 1/(1+np.exp(-z))
    else:
        return 1/(1+np.exp(-z))


def activate_(fkt, x):
    """
    Implements the gradient of the activation funtion.

    Args:
        fkt: The selected activation function.
        x: The output of the activation function.
    """
    if fkt.lower() == 'relu':
        return 1*(x > 0)
    elif fkt.lower() == 'tanh':
        return 1-np.square(x)
    elif fkt.lower() == 'logistic':
        return x*(1-x)
    else:
        return x*(1-x)
