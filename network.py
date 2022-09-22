############################# LIBRARIES ######################################
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from math import sqrt

"""=================================================================================================================="""


def get_network(opt):
    """
    Selection function for available networks.
    """
    if '2plus1d' in opt.network:
        network = models.video.r2plus1d_18
    else:
        raise Exception('Network {} not available!'.format(opt.network))
    return ResNet18(network, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)


"""=================================================================================================================="""


class ResNet18(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, network, fixconvs=False, nopretrained=True):
        super(ResNet18, self).__init__()
        self.model = network(pretrained=nopretrained)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False

        for param in self.model.parameters():
                param.requires_grad = False


        self.regressor = nn.Linear(self.model.fc.in_features, 300)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)
        x = self.model(x)
        x = x.view(bs*nc, -1)
        x = x.reshape(bs, nc, -1)
        x = torch.mean(x, 1)
        x = self.dropout(x)
        x = self.regressor(x)
        x = F.normalize(x)
        
        return x


"""=================================================================================================================="""

class cross_attention(nn.Module):
    """
    k q v
    """

    def __init__(self, dim):
        super(cross_attention, self).__init__()
    
        
    
        self.W_K1 = nn.Linear(dim, dim, bias=False)
        self.W_Q1 = nn.Linear(dim, dim, bias=False)
        self.W_V1 = nn.Linear(dim,dim, bias=False)

        self.W_k2 = nn.Linear(dim, dim, bias=False)
        self.W_q2 = nn.Linear(dim, dim, bias=False)
        self.W_v2 = nn.Linear(dim,dim, bias=False)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.05)

     
    def forward(self, V, T):
        
        T = F.normalize(T)
        V = V.transpose(-2,-1)
        T = T.transpose(-2,-1)

        query_1 = self.relu(self.W_Q1(V))
        key_1 = self.relu(self.W_K1(T))
        value_1 = self.relu(self.W_V1(T)) #(300,20)

        query_2 = self.relu(self.W_q2(T))
        key_2 = self.relu(self.W_k2(V))
        value_2 = self.relu(self.W_v2(V))

        d_k_1 = query_1.size(-1)
        scores_1=torch.matmul(query_1, key_1.transpose(-2, -1)) / sqrt(d_k_1)
        p_attn_1 = F.softmax(scores_1, dim=1)
        weight_T_1 = torch.matmul(p_attn_1, value_1)
        weight_T_1 = weight_T_1.transpose(-2,-1)

        cross_attention_V = torch.add(V.transpose(-2,-1),weight_T_1)
        cross_attention_V = F.normalize(cross_attention_V)

        d_k_2 = query_2.size(-1)
        scores_2=torch.matmul(query_2, key_2.transpose(-2, -1)) / sqrt(d_k_2)
        p_attn_2 = F.softmax(scores_2, dim=1)
        weight_T_2 = torch.matmul(p_attn_2, value_2)
        weight_T_2 = weight_T_2.transpose(-2,-1)

        cross_attention_T = torch.add(T.transpose(-2,-1),weight_T_2)
        cross_attention_T = F.normalize(cross_attention_T)

        Aggregation = torch.add(cross_attention_V, cross_attention_T)
        Aggregation = F.normalize(Aggregation)
        return Aggregation  

class LUPI(nn.Module):
    """
    
    """

    def __init__(self):
        super(LUPI, self).__init__()
    
        self.FC1 = nn.Linear(300, 512)
        self.FC2 = nn.Linear(512, 512)
        self.FC3 = nn.Linear(512, 300)

        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.05)
    def forward(self, x):
        
        h1 = self.relu(self.FC1(x))
        h1 = self.dropout(h1)
        h2 = self.relu(self.FC2(h1))
        h2 = self.dropout(h2)
        h3 = self.relu(self.FC3(h2))
        return h3



