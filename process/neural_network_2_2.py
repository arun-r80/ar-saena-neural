from process import neural_network_2_1 as neural
import numpy as np
import random


def vectorize_y(y):
    """
    Create vectorized training output for a given training
    :return: vectorized training output
    """
    vector_base = np.zeros(10)
    vector_base[y] = 1
    return vector_base


class neural_2_2(neural.neural_2_1):

    def __init__(self, training_data, no_of_neural_layers, no_of_training_set_members=50000,
                 no_of_validation_data_members=10000, eta=0.25, l_regularize=0.15, m=9000):
        """
                Initialize class with size as input.
                :param no_of_neural_layers: a list which contains no of neurons for each layer.So, len(size) will provide total
                no of layers in this neural schema, including input(which contains features or "X" values)
                and output layers.
        """
        super().__init__(training_data, no_of_neural_layers, no_of_training_set_members=no_of_training_set_members,
                         no_of_validation_data_members=no_of_validation_data_members, eta=eta,
                         l_regularize=l_regularize, m=m)
        self.training_data_transposed = list(zip(self.X.T, self.Y.T))

    def _regularize_w(self, eta, lmbda, m, batch_size, w_network, nabla_w):
        """
        A convenience method to employ different reqularization techniques. A convenience wrapper to
        seperate regularization logic from backward propagation.
        :param eta: learning rate
        :param lmbda: regularization parameter
        :param m: training set size
        :param batch_size: batch size
        :param w_network: weights for the network
        :param nabla_w: difference in weights
        :return: regularized weight based on hyper-parameters and regularization logic
        """
        return [(1 - eta * (lmbda / m)) * w - (eta / batch_size) * nw for w, nw in
                zip(w_network, nabla_w)]

    def _backward_propagate__(self, a, y):
        """
        Update weights and biases for the epoch, using backward propagation.
        :return:
        """
        nabla_w = [np.zeros(w.shape) for w in self.W]
        nabla_b = [np.zeros(b.shape) for b in self.B]
        delta = self._moment_lossonoutput__(a, y)  # delta => n[l],b where b is batch size
        db = np.sum(delta, axis=1, keepdims=True)  # db => n[l],1
        dw = np.dot(delta, self.A[-2].T)  # delta => n[l], b; A[-2].T => b,n[l-1]; dw => n[l], n[l-2]
        nabla_w[-1] = dw
        nabla_b[-1] = delta
        for layer in range(len(self.W) - 1, 0, -1):
            delta = np.dot(self.W[layer].T, delta) * self._moment_of_activation_function_on_weighted_output__(layer)

            # np.dot(W[layer].T => (n[l-1],n[l], delta => (n[l],b)) => n[l-1],b
            # delta = n[l-1],b
            db_population = delta
            dw = np.dot(db_population, self.A[layer - 1].T)
            nabla_w[layer - 1] = dw
            # nabla_w[layer - 1] = self.W[layer - 1] * (1 - self.eta * (self.lmbda / self.m)) - (
            #             self.eta / self.batch_size) * dw
            nabla_b[layer - 1] = db_population
        return nabla_w, nabla_b

    def _run_minibatch(self, minibatch):
        """
        A wrapper function to evaluate minibatch one by one
        :param minibatch: one minibatch from stochastically re-shuffled training data
        :return:
        """
        nabla_w = [np.zeros(w.shape) for w in self.W]
        nabla_b = [np.zeros(b.shape) for b in self.B]
        for x, y in minibatch:
            self._prepare_epoch__(np.reshape(x, (784, 1)))
            self._propagate_forward__()
            delta_nabla_w, delta_nabla_b = self._backward_propagate__(self.A[-1], np.reshape(y, (self.size[-1], 1)))
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.W = self._regularize_w(self.eta, self.lmbda, self.m, self.batch_size, self.W, nabla_w)
        # self.W = [ (1 - self.eta*(self.lmbda/self.m))*w - (self.eta/self.batch_size) * nw  for w,nw in zip(self.W, nabla_w)]
        # self.W = (1 - self.eta*(self.lmbda/self.m))*self.W - (self.eta/self.batch_size) * nabla_w
        self.B = [b - (self.eta / self.batch_size) * nb for b, nb in zip(self.B, nabla_b)]
        # self.B -= (self.eta/self.batch_size) * nabla_b

    def train(self, epochs=10):
        """ This is the externally exposed class, which is just a wrapper
                on forward and backward propagation functions.
                epochs: No of epochs to train the data
            """
        self.epochs = epochs
        self.cost_function = []
        self.success_rate = []
        for e in range(epochs):
            print("Epoch ", e, end=" ")
            random.shuffle(self.training_data_transposed)
            minibatches = [self.training_data_transposed[k: k + self.batch_size] for k in
                           range(0, self.m, self.batch_size)]

            for minibatch in minibatches:
                self._run_minibatch(minibatch)
            self._calculate_loss__(self._process_feedforward(self.X), self.Y, self.lmbda, batchsize=50000)
            self.cost_function.append(self.J)
            rate = self._evaluate(neural.devectorize(self._process_feedforward(self.Validation_Data)),
                                  self.RAW_VALIDATION_Y, self.length_validation_data)
            self.success_rate.append(rate)
            print("Cost = ", self.J, "Success rate %f %", rate)
