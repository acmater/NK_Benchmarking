import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    Custom coded RNN class in PyTorch
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size,output_dim=1):
        super(RNN, self).__init__()
        self.hidden_dim         = hidden_dim
        self.embedding_dim      = embedding_dim
        self.embed              = nn.Embedding(vocab_size,embedding_dim)
        self.internallayer      = nn.Linear(embedding_dim+hidden_dim,hidden_dim)
        self.outputlayer        = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        hidden = torch.zeros(x.shape[0],self.hidden_dim).cuda()
        hidden = hidden.double()
        for char in x.T:
            emb = self.embed(char)
            emb = emb.double()
            pre = torch.cat((hidden,emb),axis=1)
            hidden = torch.tanh(self.internallayer(pre))
        return self.outputlayer(hidden)
