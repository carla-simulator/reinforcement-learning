from chainer import functions as F
import numpy as np

def get_from_str(nonlinearity_str):
    if nonlinearity_str == "relu":
        return F.relu
    elif nonlinearity_str.startswith("lrelu"):
        # string should be in format "lrelu_0.2" , where the number is the negative slope
        relu_neg_slope = float(nonlinearity_str.split('_')[1])
        return lambda w: F.leaky_relu(w, slope=relu_neg_slope)
    if nonlinearity_str.startswith("elu"):
        elu_alpha = float(nonlinearity_str.split('_')[1])
        return lambda w: F.elu(w, alpha=elu_alpha)
    else:
        raise Exception('Unknown nonlinearity', nonlinearity_str)


