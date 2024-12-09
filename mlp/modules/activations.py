import numpy as np
from .base import Module
from scipy.special import log_softmax, softmax 


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * ((input > 0).astype(input.dtype))


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.divide(1, 1 + np.exp(-input))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        res = self.compute_output(input)
        return grad_output * res * (1 - res)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return softmax(input, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        res = self.compute_output(input)
        return res * (grad_output - np.sum(grad_output * res, axis=1, keepdims=True))


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension.
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        softmax_v = softmax(input, axis=1)
        eye_diff = np.eye(softmax_v.shape[1]) - np.einsum('ij,ik->ijk', np.ones_like(softmax_v), softmax_v)
        return np.einsum("ij,ijk->ik", grad_output, eye_diff)
