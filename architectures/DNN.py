import torch
import torch.nn as nn
from collections import OrderedDict
import itertools

def build_DNN(input_dim, hidden_dim, num_hidden, embedding_dim=1, vocab_size=20,output_dim=1 ,activation_func=nn.Sigmoid):

    """ Function that automates the generation of a DNN by providing a template for
    pytorch's nn.Sequential class

    Parameters
    ----------
    input_dim : int

        Number of dimensions of input vector

    hidden_dim : int

        Number of dimensions for each hidden layer

    num_hidden : int

        Number of hidden layers to construct

    output_dim : int, default=1

        Number of output (label) dimensions

    activation_func : nn.Function

        Activation function applied to all but the penultimate layer

    return nn.Module

        The feedforward network as a PyTorch model
    """

    embed  = OrderedDict([("Embedding", nn.Embedding(vocab_size,embedding_dim))])
    input   = OrderedDict([("Input", nn.Linear(input_dim,hidden_dim)),("Sig1", activation_func())])
    hidden_structure = [[('Hidden{}'.format(i), nn.Linear(hidden_dim,hidden_dim)),
                            ('Sig{}'.format(i+1), nn.Sigmoid())] for i in range(1,num_hidden+1)]
    hiddens = OrderedDict(list(itertools.chain.from_iterable(hidden_structure)))
    output = OrderedDict([("Output", nn.Linear(hidden_dim,output_dim))])

    return nn.Sequential(OrderedDict(**embed, **input, **hiddens, **output))

class DNN(nn.Module):
    def __init__(self,input_dim, hidden_dim, embedding_dim=1, vocab_size=20,output_dim=1 ,activation_func=torch.sigmoid):
        super(DNN, self).__init__()
        self.embed       = nn.Embedding(vocab_size,embedding_dim)
        self.input_layer = nn.Linear(input_dim,hidden_dim)
        self.hidden1     = nn.Linear(hidden_dim,hidden_dim)
        self.hidden2     = nn.Linear(hidden_dim,hidden_dim)
        self.hidden3     = nn.Linear(hidden_dim,hidden_dim)
        self.output_layer = nn.Linear(hidden_dim,output_dim)
        self.activation_func = activation_func

    def forward(self,x):
        emb = self.embed(x).squeeze(-1)
        hid = self.activation_func(self.input_layer(emb))
        hid1 = self.activation_func(self.hidden1(hid))
        hid2 = self.activation_func(self.hidden2(hid1))
        hid3 = self.activation_func(self.hidden3(hid2))
        return self.output_layer(hid3)


if __name__ == "__main__":
    model = build_DNN(10,20,3)
    print(model)
