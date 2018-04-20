from chainer import links as L
import numpy as np

def init_with_str(link, init_str = ""):
    if init_str == "xavier":
        xavier(link)
    elif init_str.startswith("msra"):
        # string should be in format "msra_0.2" , where the number is the negative slope
        relu_neg_slope = float(init_str.split('_')[1])
        msra(link, relu_neg_slope=relu_neg_slope)
    else:
        raise Exception('Unknown initialization method', init_str)

def xavier(link):
    # Mimic torch's default parameter initialization
    # TODO(muupan): Use chainer's initializers when it is merged
    for l in link.links():
        if isinstance(l, L.Linear):
            out_channels, in_channels = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)
            if l.b is not None:
                l.b.data[:] = np.random.uniform(-stdv, stdv,
                                                size=l.b.data.shape)
        elif isinstance(l, L.Convolution2D):
            out_channels, in_channels, kh, kw = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels * kh * kw)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)
            if l.b is not None:
                l.b.data[:] = np.random.uniform(-stdv, stdv,
                                                size=l.b.data.shape)
                
def msra(link, relu_neg_slope=0):
    # Mimic torch's default parameter initialization
    # TODO(muupan): Use chainer's initializers when it is merged
    for l in link.links():
        if isinstance(l, L.Linear):
            out_channels, in_channels = l.W.data.shape
            in_dim = in_channels          
        elif isinstance(l, L.Convolution2D):
            out_channels, in_channels, kh, kw = l.W.data.shape
            in_dim = in_channels * kh * kw
        else:
            return
            
        stdv = 2 / np.sqrt(in_dim * (1 + relu_neg_slope**2))
            
        l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)
        if l.b is not None:
            l.b.data[:] = np.random.uniform(-stdv, stdv,
                                            size=l.b.data.shape)
