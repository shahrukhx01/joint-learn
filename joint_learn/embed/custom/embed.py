from torch import nn


class Word2Vec(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        this constructor instantiates the nn.Linear modules
        """
        super().__init__()
        self.layer1 = nn.Linear(
            in_features=input_size, out_features=hidden_size, bias=False
        )
        self.layer2 = nn.Linear(
            in_features=hidden_size, out_features=input_size, bias=False
        )
        # self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, one_hot):
        """
        this method does the forward pass
        """
        x = self.layer1(one_hot)
        x = self.layer2(x)
        return x


def init_weights(m):
    """
    this method initializes the weights using the normal distribution
    or sets the pretrained weights of a model

    Argument:
    m -- pretrained weights of the model
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)
