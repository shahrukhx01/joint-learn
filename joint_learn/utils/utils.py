from torch import nn


def init_weights(m):
    """
    this method initializes the weights using the normal distribution
    or sets the pretrained weights of a model

    Argument:
    m -- pretrained weights of the model
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)
