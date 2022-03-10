import numpy as np


class Operation:
    """
    Base class represents an operation in a neural network.
    """

    def __init__(self):
        self.input_ = None
        self.output = None
        self.input_grad = None

    def forward(self, input_: np.ndarray):
        """
        Feed the input to the operation and return the output.
        :param input_: a ndarray representing the input of the operation
        :return: output of the operation
        """
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Given the gradient of the next operation with respect to the output of this operation,
        compute the gradient of this operation with respect to the input and accumulate those gradients.
        :param output_grad: a ndarray representing the gradient of the next operation
        :return: the gradient of the accumulated gradient up until this operation
        """
        # the gradient must be of the same shape of its corresponding input
        assert output_grad.shape[0] == self.output.shape[0]

        self.input_grad = self._input_grad(output_grad)
        assert self.input_grad.shape[0] == self.input_.shape[0]
        return self.input_grad

    def _output(self) -> np.ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ParamOperation(Operation):
    """
    A class represents an operation with a parameter.
    """

    def __init__(self, param: np.ndarray):
        super().__init__()
        self.param = param
        self.param_grad = None

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Given the gradient of the next operation with respect to the output of this operation,
        compute the gradient of this operation with respect to the input
        and the gradient of this operation with respect to the parameters.
        Then, accumulate the gradient of the next operation with this input gradient to form the _input_grad;
        accumulate the gradient of the next operation with this param gradient to form the _param_grad.
        :param output_grad: the gradient of the next operation
        :return: the accumulated input gradient up until this operation
        """
        assert output_grad.shape[0] == self.output.shape[0]

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert self.input_grad.shape[0] == self.input_.shape[0]
        assert self.param_grad.shape[0] == self.param.shape[0]

        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class WeightMultiply(ParamOperation):
    """
    A class represents the weight multiplication operation for a neural network.
    """

    def __init__(self, weight: np.ndarray):
        super().__init__(weight)

    def output(self) -> np.ndarray:
        """
        Multiply the input and the weights.
        """
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of this operation with respect to the input.
        Then, accumulate this gradient and the output_grad according to the Chain of Rule.
        """
        # input(mxn) dot param(nxk) = output(mxk)
        # input_grad will be of shape (mxn) by multiplying output(mxk) and paramT(kxn)
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of this operation with respect to the parameter.
        Then, accumulate this gradient and the output_grad according to the Chain of Rule.
        """
        # input(mxn) dot param(nxk) = output(mxk)
        # param_grad = nxk = nxm dot mxk
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    """
    A class represents the bias addition.
    """

    def __init__(self, bias: np.ndarray):
        # check if bias is a vector
        assert bias.shape[0] == 1
        super().__init__(bias)

    def _output(self) -> np.ndarray:
        """
        Perform elementwise-addition between the input and the bias.
        Consider each sample, each of their features will be added by a biased value.
        """
        return self.input_ + self.param

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # because the input and output are of the same shape
        # we just need to multiply each element of the output_grad with 1 (which is the derivative of this addition).
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        param_grad = np.ones_like(self.param) * output_grad
        # because a single bias influences a specific feature of all samples,
        # a small change of bias leads to the same amount of change of the output
        # of that specific feature of all samples.
        # therefore, we can sum all the derivatives of the specific feature of all samples
        # to make the gradient more concise.
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Sigmoid(Operation):
    """
    A class represents the sigmoid activation function.
    """

    def __init__(self):
        super().__init__()

    def _output(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # sigmoid derivative
        sigmoid_backward = self.output * (1.0 - self.output)
        # sigmoid performs elementwise operation
        # so the gradient is simply elementwise multiplication between the sigmoid derivative and the output gradient
        input_grad = sigmoid_backward * output_grad
        return input_grad
