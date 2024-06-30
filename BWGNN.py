import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, ChebConv, GATConv, HeteroGraphConv


class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


class PolyConvBatch(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConvBatch, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, block, feat):
        def unnLaplacian(feat, D_invsqrt, block):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            block.srcdata['h'] = feat * D_invsqrt
            block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - block.srcdata.pop('h') * D_invsqrt

        with block.local_scope():
            D_invsqrt = torch.pow(block.out_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, block)
                h += self._theta[k]*feat
        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas



def calculate_theta2_hf(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(1, d):
        f = sympy.poly((0.3 - 1/2 + x/2) ** i)
        coeff = f.all_coeffs()
        coeff = [0] * (d - len(coeff)) + coeff
        inv_coeff = []
        for i in range(d):
            inv_coeff.append(float(coeff[d - 1 - i]))
        thetas.append(inv_coeff)
    return thetas


def calculate_theta2_lf(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(1, d):
        f = sympy.poly((0.3+ 1/2 - x/2) ** i)
        coeff = f.all_coeffs()
        coeff = [0] * (d - len(coeff)) + coeff
        inv_coeff = []
        for i in range(d):
            inv_coeff.append(float(coeff[d - 1 - i]))
        thetas.append(inv_coeff)
    return thetas




# heterogeneous graph
class BWGNN_Hetero(nn.Module):
    def __init__(self, in_feats_voc, in_feats_sms,in_feats_personal,h_feats, num_classes, graph, d=2):
        super(BWGNN_Hetero, self).__init__()
        self.g = graph
        self.attn_fn = nn.Tanh()
        self.lstm_voc = nn.LSTM(in_feats_voc,in_feats_voc,batch_first=True)
        self.lstm_sms = nn.LSTM(in_feats_sms, in_feats_sms,batch_first=True)
        self.thetas = calculate_theta2(d=d)
        self.thetas_hf = calculate_theta2_hf(d=d)
        self.thetas_lf = calculate_theta2_lf(d=d)
        self.h_feats = h_feats
        self.conv = [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]

        self.conv_hf = [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas_hf]
        self.conv_lf = [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas_lf]

        self.W_f = [nn.Sequential(nn.Linear(h_feats, h_feats),self.attn_fn,nn.Linear(h_feats, 1,bias=False)) for i in range(len(self.g))]

        self.linear = nn.Linear(in_feats_voc, h_feats)
        self.linear1 = nn.Linear(in_feats_sms, h_feats)
        self.linear2 = nn.Linear(2*h_feats, h_feats)
        self.linear3 = nn.Linear(2*h_feats, h_feats)
        self.linear4 = nn.Linear(4*h_feats, h_feats)
        self.linear5 = [nn.Linear(h_feats, h_feats) for i in range(len(self.g))]

        self.linear_personal = nn.Linear(in_feats_personal, h_feats)
        self.linear6 = nn.Linear(6*h_feats, num_classes)
        self.act = nn.LeakyReLU()
        # print(self.thetas)
        for param in self.parameters():
            print(type(param), param.size())

    def forward(self, voc_features,sms_features,personal_feature):
        x_in_voc, _ =  self.lstm_voc(voc_features)
        x_in_sms, _ = self.lstm_sms(sms_features)

        x_in_voc = self.linear(x_in_voc[:,-1])
        x_in_sms = self.linear1(x_in_sms[:, -1])
        x_in_personal = self.linear_personal(personal_feature)


        x_in_voc = self.act(torch.cat((x_in_voc,x_in_personal),dim=1))
        x_in_sms = self.act(torch.cat((x_in_sms,x_in_personal),dim=1))
        x_in_voc_sms=torch.cat((x_in_voc,x_in_sms),dim=1)

        x_in_voc= self.linear2(x_in_voc)
        x_in_sms = self.linear3(x_in_sms)
        x_in_voc_sms=self.linear4(x_in_voc_sms)

        x_in_voc = self.act(x_in_voc)
        x_in_sms = self.act(x_in_sms)
        x_in_voc_sms=self.act(x_in_voc_sms)

        x_in=[]
        x_in.append(x_in_voc)
        x_in.append(x_in_sms)
        x_in.append(x_in_voc_sms)


        h_all = []

        for relation in range(len(self.g)):
            h_final = []

            for conv in self.conv_lf:
                h0 = conv(self.g[relation], x_in[relation])
                h_final.append(h0)

            for conv in self.conv:
                h0 = conv(self.g[relation], x_in[relation])
                h_final.append(h0)

            for conv in self.conv_hf:
                h0 = conv(self.g[relation], x_in[relation])
                h_final.append(h0)

            h_filters = torch.stack(h_final, dim=1)
            h_filters_proj = self.W_f[relation](h_filters)
            soft_score = F.softmax(h_filters_proj, dim=1)
            # soft_score=soft_score.expand((h_filters.shape[0],)+soft_score.shape)

            res=(soft_score*h_filters).sum(1)

            h = self.linear5[relation](res)
            h_all.append(h)

        h_all = torch.cat((h_all[0], h_all[1],h_all[2]), dim=1)
        h_all = self.act(h_all)
        h_all=torch.cat((h_all, x_in[0],x_in[1],x_in[2]), dim=1)
        h_all = self.linear6(h_all)
        return h_all
