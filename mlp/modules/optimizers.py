import numpy as np
from typing import Tuple
from .base import Module, Optimizer


class SGD(Optimizer):
    """
    Optimizer implementing stochastic gradient descent with momentum
    """
    def __init__(self, module: Module, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param momentum: momentum coefficient (alpha)
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]

        for param, grad, m in zip(parameters, gradients, self.state['m']):
          np.add(grad, self.weight_decay * param, out=grad)
          np.add(m * self.momentum, grad, out=m)
          np.subtract(param, self.lr * m, out=param)


class Adam(Optimizer):
    """
    Optimizer implementing Adam
    """
    def __init__(self, module: Module, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param betas: Adam beta1 and beta2
        :param eps: Adam eps
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]
            self.state['v'] = [np.zeros_like(param) for param in parameters]
            self.state['t'] = 0

        self.state['t'] += 1
        t = self.state['t']
        for param, grad, m, v in zip(parameters, gradients, self.state['m'], self.state['v']):
            gt = np.add(grad, self.weight_decay * param)
            np.multiply(m, self.beta1, out=m)
            np.add(m, (1 - self.beta1) * gt, out=m)
            np.multiply(v, self.beta2, out=v)
            np.add(v, (1 - self.beta2) * gt**2, out=v)
            m_est = m / (1 - self.beta1**t)
            v_est = v / (1 - self.beta2**t)
            np.subtract(param, self.lr * m_est / (np.sqrt(v_est) + self.eps), out=param)

