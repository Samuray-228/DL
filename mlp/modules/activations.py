import numpy as np
from .base import Module
import scipy.special as sp


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        # return super().compute_output(input)
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        # return super().compute_grad_input(input, grad_output)
        return grad_output * np.where(input > 0, 1, 0)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        # return super().compute_output(input)
        return sp.expit(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        # return super().compute_grad_input(input, grad_output)
        x = self.compute_output(input)
        return x * (1 - x) * grad_output


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        # return super().compute_output(input)
        x = np.exp(input - np.amax(input, axis=-1, keepdims=True))
        return x / np.sum(x, axis=-1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (b yatch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        # return super().compute_grad_input(input, grad_output)
        x = self.compute_output(input)
        return x * (grad_output - np.sum(grad_output * x, axis=-1, keepdims=True))


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return input - np.amax(input, axis=-1, keepdims=True) - np.log(np.sum(np.exp(input - np.amax(input, axis=-1, keepdims=True)), axis=-1, keepdims=True))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        x = np.exp(input - np.amax(input, axis=-1, keepdims=True))
        softmax = x / np.sum(x, axis=-1, keepdims=True)
        diag = np.expand_dims(softmax, axis=-1) * np.expand_dims(np.eye(input.shape[1], input.shape[1]), axis=0)
        return np.einsum('ijk,ik->ij', diag - np.einsum('ij,ik->ijk', softmax, softmax), grad_output / softmax)
