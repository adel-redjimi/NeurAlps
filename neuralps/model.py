import numpy as np
from functools import reduce
import neuralps
import neuralps.diffops as diffops
import neuralps.loss as loss
import neuralps.metrics

class Model:
    """ A wrapper class to build, train, save, load, use a differentiable model. """
    def __init__(self, *ops):
        """ *ops: the differentiable operations that make this differentiable model.
        
            e.g:
            model = Model(
                Linear(10, 1),
                Sigmoid()
            )
            
            this example is a a logistic regression. """
        self.ops = list(ops)

    def predict(self, X, train=False):
        """ Predicts by propagating the signal from the first differentiable operation to the output differentiable operation.

        X: input tensor, shape (number of samples, number of features).
        train: train or test mode. """
        return reduce(
            lambda x, op: op.forward(x, train=train),
            self.ops,
            X
        )

    def backprop(self, loss_grad, skip_last=False):
        """ Backpropagate the error signal from the output differentiable operation to the first differentiable operation.

        loss_grad: the gradient of the loss w.r.t. output of the network.
        skip_last: when we have cross entropy and sigmoid, loss_grad contains the gradient w.r.t. the preactivations of the sigmoid, so we skip computing the gradient at the sigmoid unit. """ 
        return reduce(
            lambda d, op: op.backward(d),
            reversed(self.ops[:-1 if skip_last else None]),
            loss_grad
        )

    def sgd_update(self, learning_rate):
        """ Updates the parameters of each differentiable operation using SGD rule.
            Note: differentiable operations without learnable parameters do nothing when .sgd_update() is invoked. """
        for op in self.ops:
            op.sgd_update(learning_rate)

    def rms_update(self, learning_rate, decay_rate):
        """ Updates the parameters of each differentiable operation using RMSprop rule.
            Note: differentiable operations without learnable parameters do nothing when .rms_update() is invoked. """
        for op in self.ops:
            op.rms_update(learning_rate, decay_rate)

    def train(self, X_train, y_train, loss_function,
                    batch_size=50, epochs=5, verbose=False,
                    X_val=None, y_val=None, algorithm='sgd',
                    learning_rate=.1, decay_rate=.99, save_best_model=None,
                    metric=None):
        """ Training the differentiable model.

        X_train: tensor of shape (number of samples, number of features).
        y_train: tensor of shape (number of samples, 1)

        loss_function: class corresponding to the loss function to be minimized (BinaryCrossEntropy or MeanSquaredError)

        batch_size: number of samples per mini-batch.
        epochs: number of passage over training data.

        verbose: if True, progress is printed on the screen.

        X_val, y_val: validation data.

        algorithm: 'sgd' or 'rmsprop'.

        learning_rate: parameter for SGD and RMSprop.
        decay_rate: parameter for RMSprop.

        metric: function to be used as metric. e.g, binary accuracy, mutli-class accuracy.

        save_best_model: (string) filename to save the best model reached during training (highest validation accuracy). """
        # highest_val_acc: keeps track of the highest validation accuracy during training.
        highest_val_acc = 0

        # n: size of training data.
        n = X_train.shape[0]

        # idx: hold indices and shuffled indices.
        idx = np.arange(n)

        # batch_edges: indices indicating where batches start and stop.
        batch_edges = list(range(0, n, batch_size)) + [None]

        for epoch in range(epochs):
            # shuffling indices
            np.random.shuffle(idx)

            for start, end in zip(batch_edges[:-1], batch_edges[1:]):
                # X_batch, y_batch: batch to be used for training.
                idx_batch = idx[start: end]
                X_batch = X_train[idx_batch]
                y_batch = y_train[idx_batch]

                y_pred = self.predict(X_batch, train=True)

                # skip_last: True if loss is binary cross entropy AND final differentiable operation is sigmoid
                skip_last = (
                    isinstance(self.ops[-1], (diffops.Sigmoid, diffops.Softmax))
                    and (loss_function == loss.BinaryCrossEntropy 
                    or loss_function == loss.CategoricalCrossEntropy)
                )

                # loss_grad: gradient of loss wrt outputs or preactivations.
                loss_grad = (
                    loss_function.gradient(y_batch, y_pred) if not skip_last
                    else loss_function.gradient_skip(y_batch, y_pred)
                )

                # computing gradients using backprop
                self.backprop(loss_grad, skip_last)

                # updating according the algorithm
                if algorithm is 'sgd':
                    self.sgd_update(learning_rate)
                elif algorithm is 'rmsprop':
                    self.rms_update(learning_rate, decay_rate)

            if verbose:
                print('EPOCH %d' % epoch)
                y_train_pred = self.predict(X_train, train=False)

                train_loss = loss_function.evaluate(y_train, y_train_pred)
                train_acc = metric(y_train, y_train_pred)

                print(' - train loss: %.4f     acc: %.4f' % (train_loss, train_acc))
                
            if X_val is not None:
                y_val_pred = self.predict(X_val, train=False)

                val_loss = loss_function.evaluate(y_val, y_val_pred)
                val_acc = metric(y_val, y_val_pred)

                if verbose:
                    print(' - val  loss: %.4f     acc: %.4f' % (val_loss, val_acc))

                if save_best_model is not None:
                    # if save_best_model is set with a filename
                    if val_acc > highest_val_acc:
                        # if current validation accuracy is the highest
                        highest_val_acc = val_acc

                        # save the model on the disk
                        self.save(save_best_model)

     
    def save(self, filename):
        """ Saves the model on the disk. """
        with open(filename, 'wb') as outfile:
            to_save = {}
            for i, op in enumerate(self.ops):
                class_name = op.__class__.__name__
                full_class_name = op.__class__.__module__ + '.' + class_name
                to_save['op%04d' % i] = full_class_name
                for j, param in enumerate(op.params()):
                    to_save['op%04d_%04d' % (i, j)] = param

            np.savez(outfile, **to_save)

    @staticmethod
    def load(filename):
        """ Loads model from disk. """
        ops = []
        model_data = np.load(filename)
        current_class = None
        current_params = []
        for key in sorted(model_data):
            if len(key) == 6:
                if current_class:
                    ops.append(
                        eval(current_class)(*current_params)
                    )
                current_class = str(model_data[key])
                current_params = []
            else:
                current_params.append(model_data[key])
        
        ops.append(
            eval(current_class)(*current_params)
        )

        return Model(*ops)