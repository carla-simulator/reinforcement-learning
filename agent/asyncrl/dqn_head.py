import chainer
from chainer import functions as F
from chainer import links as L
from . import nonlinearity


class NatureDQNHead(chainer.ChainList):
    """DQN's head (Nature version)"""

#TODO Alexey: bias init used to be 0.1 - does it matter?
    def __init__(self, n_input_channels=None, n_output_channels=512,
                 nonlinearity_str=None, bias=None):
        self.n_input_channels = n_input_channels
        self.nonlinearity = nonlinearity.get_from_str(nonlinearity_str)
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4, bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, bias=bias),
            L.Linear(3136, n_output_channels, bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.nonlinearity(layer(h))
        return h


class NIPSDQNHead(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=None, n_output_channels=256,
                 nonlinearity_str=None, bias=None):
        self.n_input_channels = n_input_channels
        self.nonlinearity = nonlinearity.get_from_str(nonlinearity_str)
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 16, 8, stride=4, bias=bias),
            L.Convolution2D(16, 32, 4, stride=2, bias=bias),
            L.Linear(2592, n_output_channels, bias=bias),
        ]

        super(NIPSDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.nonlinearity(layer(h))
        return h
