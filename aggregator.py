import torch
import torch.nn as nn
class Aggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, aggreation = "mean"):
        super(Aggregator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggregation = aggreation
        nn.init.xavier_uniform_(self.weight)

    def forward(self, nei_feat):
        if self.aggregation == "mean":
            aggr_neighbor = nei_feat.mean(dim =1)
        elif self.aggregation =="sum":
            aggr_neighbor = nei_feat.sum(dim = 1)
        elif self.aggregation == "max":
            aggr_neighbor = nei_feat.max(dim =1)
        else:
            raise ValueError("Unknown Aggr Type!")

        nei_hidden = torch.matmul(aggr_neighbor, self.weight)      
        return nei_hidden