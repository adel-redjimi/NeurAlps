import numpy as np

def xavier_uniform_weights(size_in, size_out):
    """ Initialize a weight matrix following the Xavier uniform initialization.
        size_in: number of input units.
        size_out: number of output units."""
    a = np.sqrt(6 / (size_in + size_out))
    return np.random.uniform(-a, a, (size_out, size_in))

class DifferentiableOperation:
    """ Abstract base class for differentiable operations.
        A differentiable operation is a node in the computational graph."""
    def forward(self, x, train=False):
        """ Function for computing the forward pass.
            It means computing the output given inputs.
            
            x: the input tensor, must be of shape (number of samples, number of features).
            train: tells the function to operate either on train mode or test mode. For most differentiable operations it doesn't have a utility, but for the dropout differentiable operation, it matter whether to compute the forward in train or test mode.
            
            We included the train parameter in the base class and all other differentiable operations classes to have a consistent interface of using the function forward()"""
        pass
    
    def backward(self, dy):
        """ Function for computing the backward pass.
            It means computing the derivatives w.r.t. learnable parameters (if there are any) and return the derivative w.r.t. the inputs to be passed in back-propagation.
            
            dy: derivative of the loss w.r.t. the output of this differentiable operation."""
        pass

    def sgd_update(self, learning_rate):
        """ Performs SGD update.

            learning_rate: the learning rate for SGD.

            Note: differentiable operations that do not have learnable parameters do not implement this function."""
        pass

    def rms_update(self, learning_rate, decay_rate):
        """ Performs RMSprop update.

            learning_rate: the learning rate for RMSprop.
            decay_rate: decay rate for RMSprop.

            Note: differentiable operations that do not have learnable parameters do not implement this function."""
        pass

    def params(self):
        """ This returns the parameters of the differentiable operation. Used when saving models."""
        return tuple()

class Linear(DifferentiableOperation):
    def __init__(self, size_in, size_out, W=None, b=None):
        """ size_in: input features size.
            size_out: output features size.
            W: matrix of weights (if loading a model).
            b: vector of biases (if loading a model)."""
        self.size_in = size_in
        self.size_out = size_out

        if W is None: 
            self.W = xavier_uniform_weights(size_in, size_out)
        else:
            self.W = W

        if b is None:
            self.b = np.zeros(size_out)
        else:
            self.b = b

        # did_rms is set to True when a differentiable operation performs an RMSprop update
        self.did_rms = False

    def forward(self, x, train=False):
        """ Computes the linear transformation Wx+b for a batch of inputs.
        
            x: input tensor, must be of shape (number of samples, size_in). """
        self.last_x = x
        return np.dot(x, self.W.T) + self.b

    def backward(self, dy):
        """ Computes the gradient w.r.t. W and b. 
            Returns gradient w.r.t. the inputs. """
        self.db = np.mean(dy, axis=0)

        x = self.last_x
        self.dW = np.dot(x.T, dy).T / x.shape[0]

        return np.dot(dy, self.W)

    def clear_gradients(self):
        self.db = np.zeros_like(self.b)
        self.dW = np.zeros_like(self.W)

    def acc_backward(self, dy):
        self.db += np.mean(dy, axis=0)

        x = self.last_x
        self.dW += np.dot(x.T, dy).T / x.shape[0]

    def sgd_update(self, learning_rate):
        """ Computes the SGD update for W and b. """
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def rms_update(self, learning_rate, decay_rate):
        """ Comptues the RMSprop update for W and p. """
        if not self.did_rms:
            # initialize squared gradients moving average with 0's
            self.rms_W = np.zeros_like(self.W)
            self.rms_b = np.zeros_like(self.b)
            self.did_rms = True

        # computing squared gradients moving average
        self.rms_W = decay_rate * self.rms_W + (1-decay_rate) * self.dW**2
        self.rms_b = decay_rate * self.rms_b + (1-decay_rate) * self.db**2

        self.W -= (learning_rate / (np.sqrt(self.rms_W) + 1e-4)) * self.dW
        self.b -= (learning_rate / (np.sqrt(self.rms_b) + 1e-4)) * self.db

    def params(self):
        return (self.size_in, self.size_out,
                self.W, self.b)


class ReLU(DifferentiableOperation):
    def forward(self, x, train=False):
        """ Computes ReLU activation for inputs max(x, 0). """
        self.last_x = x
        return np.maximum(0, x)

    def backward(self, dy):
        """ Returns gradient of the loss w.r.t. the inputs. """
        return dy * (self.last_x > 0)


class Sigmoid(DifferentiableOperation):
    def forward(self, x, train=False):
        """ Computes Sigmoid activation for inputs. """
        self.last_x = x

        # last_s: saving last activation to use it when computing the gradient
        self.last_s = 1 / (1 + np.exp(-x))

        return self.last_s

    def backward(self, dy):
        """ Returns gradient of the loss w.r.t. the inputs. """
        s = self.last_s
        return dy * (s * (1 - s))


class Softmax(DifferentiableOperation):
    """ Note: there is no backward pass implemented for the softmax activation because it is always used with the cross-entropy loss, therefore we can compute directly the gradient w.r.t. inputs of the softmax during backpropagation """
    def forward(self, x, **kwargs):
        exps = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
        return exps / np.sum(exps, axis=1)[:, np.newaxis]


class DropOut(DifferentiableOperation):
    def __init__(self, prob):
        """ prob: probability of keeping units ON. """
        self.prob = prob

    def forward(self, x, train=True):
        """ Applies dropout to inputs. """
        if train:
            # in train mode, we only keep some units ON with probability prob
            # divide by prob to preserve the same magnitide as when we don't apply dropout.
            self.mask = np.random.binomial(1, self.prob, x.shape) / self.prob
            return x * self.mask
        else:
            # we don't divide by prob because all units are ON
            return x

    def backward(self, dy):
        """ Returns gradient of the loss w.r.t. the inputs. """
        return dy * self.mask

    def params(self):
        return (self.prob,)


class FullForward(DifferentiableOperation):
    def __init__(self, size_in, size_out, activation):
        self.size_in = size_in
        self.size_out = size_out

        self.linear = Linear(size_in, size_out)
        self.activation = activation

    def forward(self, x, train=False):
        """ Returns [x, activation(Wx+b)] """
        self.last_x = x
        #self.linear.clear_gradients()
        
        a = self.linear.forward(x)
        y = self.activation.forward(a)

        return np.hstack((x, y))

    def backward(self, d):
        """ d: partial w.r.t. [x  , activation(Wx+b)]"""
        wrt_lin = self.activation.backward(d[:,-self.size_out:])
        wrt_x = self.linear.backward(wrt_lin)

        return d[:, :self.size_in] + wrt_x