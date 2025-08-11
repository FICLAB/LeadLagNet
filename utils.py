import math
import pandas as pd
import random
import scipy.sparse as sp
from typing import Dict, Optional
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def dict_mean(dict_list):
    mean_dict = {}
    std_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = np.mean([d[key] for d in dict_list])
        std_dict[key] = np.std([d[key] for d in dict_list])
    return mean_dict, std_dict


def hard_sigm(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output


def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])

def write_log(log_file, string, mode = "a"):
    with open(log_file, mode) as fo:
        fo.write(string + "\n")

def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_adjacency(adj):
    """ Normalizes the adjacency matrix according to the
        paper by Kipf et al.
        https://arxiv.org/pdf/1609.02907.pdf
    """
    adj = sp.coo_matrix(adj)
    #     adj = adj + sparse.eye(adj.shape[0])

    node_degrees = np.array(adj.sum(1))
    node_degrees = np.power(node_degrees, -0.5).flatten()
    node_degrees[np.isinf(node_degrees)] = 0.0
    node_degrees[np.isnan(node_degrees)] = 0.0

    degree_matrix = sp.diags(node_degrees, dtype=np.float32)

    adj = degree_matrix @ adj @ degree_matrix

    return adj.todense()

def create_src_lengths_mask(
        batch_size: int, src_lengths: Tensor, max_src_len: Optional[int] = None
):
    """
    Generate boolean mask to prevent attention beyond the end of source
    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths
      max_src_len: Optionally override max_src_len for the mask
    Outputs:
      [batch_size, max_src_len]
    """
    if max_src_len is None:
        max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()


def masked_softmax(scores, src_lengths, src_length_masking=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
    if src_length_masking:
        bsz, max_src_len = scores.size()
        # compute masks
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        # Fill pad positions with -inf
        scores = scores.masked_fill(src_mask == 0, -np.inf)

    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)


def read_data_Feng(task= None, period = None):

    data_path = "/home/chengrui/workspace/AAAI2022/data/FengNASDAQ20130101to20171208:1026Stock:1244date.csv"
    market_data = pd.read_csv(data_path)
    num_stock = len(market_data.Stock.unique())
    num_timestep = len(market_data.date.unique())
    # ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open_pct',
    #  'High_pct', 'Low_pct', 'Close_pct', 'Volume_pct', 'SMA5', 'SMA10',
    #  'SMA20', 'SMA30', 'SMA5_pct', 'SMA10_pct', 'SMA20_pct', 'SMA30_pct',
    #  'BBANDS_upper', 'BBANDS_middle', 'BBANDS_lowever', 'MACD',
    #  'MACD_signal', 'MACD_hist', 'RSI', 'MOM', 'ADX', 'AD', 'OBV', 'ATR',
    #  'y', 'Stock']
    x_col = ['SMA5', 'SMA10', 'SMA20', 'SMA30']
    x_col = ['Open_pct',
     'High_pct', 'Low_pct', 'Close_pct', 'Volume_pct', 'SMA5', 'SMA10',
     'SMA20', 'SMA30', 'SMA5_pct', 'SMA10_pct', 'SMA20_pct', 'SMA30_pct',
     'BBANDS_upper', 'BBANDS_middle', 'BBANDS_lowever', 'MACD',
     'MACD_signal', 'MACD_hist', 'RSI', 'MOM', 'ADX', 'AD', 'OBV', 'ATR']
    x_ = market_data[x_col].to_numpy().reshape(num_stock, num_timestep, -1).transpose(1, 0, 2)
    y_ = market_data["y"].to_numpy().reshape(num_stock, num_timestep).transpose(1, 0)

    return x_, y_

class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensure input tensors have the same shape
#         assert y_pred.shape == y_true.shape, "Input tensors must have the same shape"
        
#         # Reshape the tensors for pairwise comparisons
#         y_pred = y_pred.view(y_pred.shape[0], 1, -1)
#         y_true = y_true.view(y_true.shape[0], 1, -1)
        
#         # Calculate the return difference for each pair of stocks
#         diff = y_pred - y_pred.transpose(0, 1)
#         true_diff = y_true - y_true.transpose(0, 1)
        
#         # Create a binary mask for revenue-generating stocks (1 if true_diff > 0, 0 otherwise)
#         mask = (torch.sign(diff) != torch.sign(true_diff)).type(torch.float32)
        
#         # Calculate the rank loss: mean of masked differences between predicted returns
#         rank_loss = torch.mean(mask * torch.exp(-diff))
        reg_out= y_pred
        batch_reg_y = y_true
        rank_loss = torch.relu(-(reg_out.view(-1,1)-reg_out.view(1,-1)) * (batch_reg_y.view(-1,1)-batch_reg_y.view(1,-1))).mean()
        
        return rank_loss