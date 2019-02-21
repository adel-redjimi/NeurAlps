import numpy as np

# epsilon used for numerical stability to avoid log(0)
eps = 1e-5

class BinaryCrossEntropy:
    """ Binary Cross Entropy loss"""

    @staticmethod
    def evaluate(y_true, y_pred):
        """ Evaluates the binary cross entropy loss.
        
        y_true: correct labels.
        y_pred: predicted labels."""
        # clip y_preds to [eps, 1-eps] to avoid falling into log(0)
        y_pred_safe = np.minimum(np.maximum(y_pred, eps), 1-eps)
        return -np.mean(y_true * np.log(y_pred_safe)
                        + (1-y_true) * np.log(1-y_pred_safe))

    @staticmethod
    def gradient(y_true, y_pred):
        """ Evaluates the gradient of binary cross entropy w.r.t. the output of the network."""
        return (y_pred - y_true) / ((1 - y_pred) * y_pred)

    @staticmethod
    def gradient_skip(y_true, y_pred):
        """ Evaluates the gradient of binary cross entropy w.r.t. the preactivations of the sigmoid unit.

            Because the the gradient of the sigmoid output w.r.t. its preactivations = the denominator of the gradient of the binary cross entropy loss. Thus, it cancels out when applying chain rule.

            This function is implemented to avoid dividing then multiplying by ((1-y_pred) * y_pred)."""
        return (y_pred - y_true)


class CategoricalCrossEntropy:
    """ Categorical Cross Entropy loss"""

    @staticmethod
    def evaluate(y_true, y_pred):
        """ Evaluates the categorical cross entropy loss.
        
        y_true: correct labels.
        y_pred: predicted labels."""
        # clip y_preds to [eps, 1-eps] to avoid falling into log(0)
        y_pred_safe = np.minimum(np.maximum(y_pred, eps), 1-eps)
        return -np.mean(y_true * np.log(y_pred_safe))

    @staticmethod
    def gradient(y_true, y_pred):
        """ Evaluates the gradient of categorical cross entropy w.r.t. the output of the network."""
        return (y_pred - y_true) / ((1 - y_pred) * y_pred)

    @staticmethod
    def gradient_skip(y_true, y_pred):
        """ Evaluates the gradient of categorical cross entropy w.r.t. the preactivations of the softmax unit.

            Because the the gradient of the sigmoid output w.r.t. its preactivations = the denominator of the gradient of the binary cross entropy loss. Thus, it cancels out when applying chain rule.

            This function is implemented to avoid dividing then multiplying by ((1-y_pred) * y_pred)."""
        return (y_pred - y_true)


class MeanSquaredError:

    @staticmethod
    def evaluate(y_true, y_pred):
        """ Evaluates the mean squared error (MSE)."""
        return 0.5 * np.mean(np.square(y_pred - y_true))

    @staticmethod
    def gradient(y_true, y_pred):
        """ Evaluates the gradient of the mean squared error w.r.t. the output of the network."""
        return (y_pred - y_true)

