import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from aggregator import *
class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, 
                 activation = F.relu, 
                 aggr_nei = "mean",
                 aggr_hidden = "concat"
                 ):
        super(GraphSAGELayer, self).__init__()
        assert aggr_nei in ["mean", "sum", "max"]
        assert aggr_hidden in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_nei = aggr_nei
        self.aggr_hidden = aggr_hidden
        self.activation = activation
        self.aggregator = Aggregator(input_dim, hidden_dim, aggreation = aggr_nei)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
    
    def forward(self, src_node_feat, nei_node_feat):
        nei_hidden = self.aggregator(nei_node_feat)
        self_hidden = torch.matmul(src_node_feat, self.weight)

        if self.aggr_hidden =="sum":
            hidden = self_hidden + nei_hidden
        elif self.aggr_hidden =="concat":
            hidden = torch.concat([self_hidden, nei_hidden], dim = 1)
        else:
            raise ValueError("Hidden aggregation method not supported!")
        
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

class GraphSage(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim):
        super(GraphSage, self).__init__()
        self.num_layers = num_layers
        self.conv = nn.ModuleList()
        self.conv.append(GraphSAGELayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.conv.append(GraphSAGELayer(hidden_dim * 2, hidden_dim))
    def forward(self, node_feat_list):
        hidden = node_feat_list
        num_nei_list = [len(layer) for layer in node_feat_list]
        for l in range(self.num_layers):
            next_hidden = []
            conv = self.conv[l]
            for hop in range(self.num_layers - l):
                src_node_feat = hidden[hop]
                src_node_num = len(src_node_feat)
                nei_node_feat = hidden[hop+1].view((src_node_num, int(num_nei_list[hop +1]/src_node_num), -1))
                h = conv(src_node_feat, nei_node_feat)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

class Classifier(nn.Module):
    # 给GraphSage结构接一个线性层进行分类任务
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.encoder = GraphSage(num_layers, input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim *2, output_dim)
    
    def forward(self, node_feat_list):
        encoded = self.encoder(node_feat_list)
        logits = F.softmax(self.fc(encoded), dim = 1)   
        return logits