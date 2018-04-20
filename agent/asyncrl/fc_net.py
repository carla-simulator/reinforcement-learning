import chainer
from chainer import functions as F
from chainer import links as L
from . import nonlinearity

class FCNet(chainer.ChainList):

    def __init__(self, n_channels_list=[], last_nonlinearity=None, nonlinearity_str=None):
        # n_channels is a list of channel sizes. The first entry in n_channels_list should be the number of input channels, the last - the number of output channels
        #assert len(n_channels_list) >= 2, "The first entry in n_channels_list should be the number of input channels, the last - the number of output channels"
        self.last_nonlinearity = last_nonlinearity
        self.nonlinearity = nonlinearity.get_from_str(nonlinearity_str)

        layers = []
        for nlayer in range(len(n_channels_list)-1):
            layers.append(L.Linear(n_channels_list[nlayer], n_channels_list[nlayer+1]))
        super().__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self[:-1]:
            h = self.nonlinearity(layer(h))
        if len(self) > 0:
            if self.last_nonlinearity is True:
                h = self.nonlinearity(self[-1](h))
            else:
                h = self[-1](h)
        return h
