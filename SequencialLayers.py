from utils import *
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.autograd import Variable
from RotaryTransformer import *
import torch
from torch import nn
import torch.nn.functional as F


class MRS(nn.Module):
    def __init__(self, ipt_dim, hid_dim, lr_num=2):
        super(MRS, self).__init__()
        self.lstm = nn.LSTM(input_size=ipt_dim, hidden_size=hid_dim, num_layers=lr_num, bidirectional=True,
                            batch_first=True)
        self.lstm_sim = nn.LSTM(input_size=ipt_dim * 2, hidden_size=hid_dim, num_layers=lr_num, bidirectional=True,
                                batch_first=True)
        self.Leakyrelu = torch.nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, stock, market):
        opts, _ = self.lstm.forward(stock)
        stock_market_cat = torch.cat([stock, market], axis=-1)
        opts_sim, _ = self.lstm_sim.forward(stock_market_cat)
        return opts - opts_sim, opts_sim


class SCD_cell(nn.Module):
    def __init__(self, bottom_size, hidden_size, top_size, alpha, last_layer):
        super(SCD_cell, self).__init__()
        self.bottom_size = bottom_size
        self.hidden_size = hidden_size
        self.top_size = top_size
        self.alpha = alpha
        self.last_layer = last_layer
        '''
        U_11 means the state transition parameters from layer l (current layer) to layer l
        U_21 means the state transition parameters from layer l+1 (top layer) to layer l
        W_01 means the state transition parameters from layer l-1 (bottom layer) to layer l
        '''
        self.U_11 = nn.Parameter(torch.zeros(4 * self.hidden_size + 1, self.hidden_size))
        if not self.last_layer:
            self.U_21 = nn.Parameter(torch.zeros(4 * self.hidden_size + 1, self.top_size))
        self.W_01 = nn.Parameter(torch.zeros(4 * self.hidden_size + 1, self.bottom_size))
        self.bias = nn.Parameter(torch.zeros(4 * self.hidden_size + 1))
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, c, h_bottom, h, h_top, z, z_bottom):
        # h_bottom.size = bottom_size * batch_size
        s_recur_ = torch.mm(self.U_11, h)
        s_recur = (1 - z.expand_as(s_recur_)) * s_recur_
        #         s_recur = s_recur_

        if not self.last_layer:
            s_topdown_ = torch.mm(self.U_21, h_top)
            s_topdown = z.expand_as(s_topdown_) * s_topdown_
        else:
            s_topdown = Variable(torch.zeros(s_recur.size(), device=c.device), requires_grad=False)
        s_bottomup_ = torch.mm(self.W_01, h_bottom)
        s_bottomup = z_bottom.expand_as(s_bottomup_) * s_bottomup_

        f_s = s_recur + s_topdown + s_bottomup + self.bias.unsqueeze(1).expand_as(s_recur)
        # f_s.size = (4 * hidden_size + 1) * batch_size
        f = torch.sigmoid(f_s[0:self.hidden_size, :])  # hidden_size * batch_size
        i = torch.sigmoid(f_s[self.hidden_size:self.hidden_size * 2, :])
        o = torch.sigmoid(f_s[self.hidden_size * 2:self.hidden_size * 3, :])
        g = torch.tanh(f_s[self.hidden_size * 3:self.hidden_size * 4, :])
        z_hat = hard_sigm(self.alpha, f_s[self.hidden_size * 4:self.hidden_size * 4 + 1, :])

        one = Variable(torch.ones(f.size(), device=c.device), requires_grad=False)
        z = z.expand_as(f)
        z_bottom = z_bottom.expand_as(f)

        c_new = z * (i * g) + (one - z) * (one - z_bottom) * c + (one - z) * z_bottom * (f * c + i * g)
        h_new = z * o * torch.tanh(c_new) + (one - z) * (one - z_bottom) * h + (one - z) * z_bottom * o * torch.tanh(
            c_new)
        z_new = z_hat
        return h_new, c_new, z_new


class BiLSTMATT(nn.Module):

    def __init__(self, ipt_dim, hid_dim, lr_num, nheads=4, dropout_p=0.2, bi_directional=True):
        super(BiLSTMATT, self).__init__()
        self.bilstm = nn.LSTM(input_size=ipt_dim, hidden_size=hid_dim, num_layers=lr_num, bidirectional=bi_directional,
                              batch_first=True)
        self.encoder_layer = TransformerEncoderLayer(d_model=2 * hid_dim, nhead=nheads, dim_feedforward=hid_dim,
                                                     dropout=dropout_p, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, lr_num)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x):
        opt, _ = self.bilstm(x)
        opt = self.transformer_encoder(opt)
        return opt


class SCD(nn.Module):

    def __init__(self, alpha, input_size, size_list, lr_num=2, nheads=4, dropout_p=0.5):
        super(SCD, self).__init__()
        self.alpha = alpha
        self.input_size = input_size
        self.size_list = size_list
        self.dropout_p = dropout_p
        self.cell_1 = SCD_cell(self.input_size, self.size_list[0], self.size_list[1], self.alpha, False)
        self.cell_2 = SCD_cell(self.size_list[0], self.size_list[1], None, self.alpha, True)
        self.drop = torch.nn.Dropout(p=self.dropout_p)

        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, inputs, hidden=None):
        """
        input shape: [batch, time, feature]
        output shape:  [batch, time, feature]
        """
        time_steps = inputs.size(1)
        batch_size = inputs.size(0)

        if hidden == None:
            h_t1 = Variable(torch.zeros(self.size_list[0], batch_size, dtype=inputs.dtype, device=inputs.device),
                            requires_grad=False)
            c_t1 = Variable(torch.zeros(self.size_list[0], batch_size, dtype=inputs.dtype, device=inputs.device),
                            requires_grad=False)
            z_t1 = Variable(torch.zeros(1, batch_size, dtype=inputs.dtype, device=inputs.device), requires_grad=False)
            h_t2 = Variable(torch.zeros(self.size_list[1], batch_size, dtype=inputs.dtype, device=inputs.device),
                            requires_grad=False)
            c_t2 = Variable(torch.zeros(self.size_list[1], batch_size, dtype=inputs.dtype, device=inputs.device),
                            requires_grad=False)
            z_t2 = Variable(torch.zeros(1, batch_size, dtype=inputs.dtype, device=inputs.device), requires_grad=False)
        else:
            (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2) = hidden
        z_one = Variable(torch.ones(1, batch_size, dtype=inputs.dtype, device=inputs.device), requires_grad=False)

        h_1 = []
        h_2 = []
        z_1 = []
        z_2 = []
        for t in range(time_steps):
            h_t1, c_t1, z_t1 = self.cell_1(c=c_t1, h_bottom=inputs[:, t, :].t(), h=h_t1, h_top=h_t2, z=z_t1,
                                           z_bottom=z_one)
            h_t2, c_t2, z_t2 = self.cell_2(c=c_t2, h_bottom=h_t1, h=h_t2, h_top=None, z=z_t2,
                                           z_bottom=z_t1)
            h_1 += [h_t1.t()]
            h_2 += [h_t2.t()]
            z_1 += [z_t1.t()]
            z_2 += [z_t2.t()]

        hidden = (h_t1, c_t1, z_t1, h_t2, c_t2, z_t2)

        h_1, h_2, z_1, z_2, hidden = torch.stack(h_1, dim=1), torch.stack(h_2, dim=1), torch.stack(z_1,
                                                                                                   dim=1), torch.stack(
            z_2, dim=1), hidden

        opts = self.drop(h_2 * z_2)

        return opts


class BiLSTMATT(nn.Module):

    def __init__(self, ipt_dim, hid_dim, lr_num, nheads = 4, dropout_p =0.2, bi_directional = True):
        super(BiLSTMATT, self).__init__()
        self.bilstm = nn.LSTM(input_size=ipt_dim, hidden_size=hid_dim, num_layers=lr_num, bidirectional=bi_directional,
                       batch_first=True)
        self.encoder_layer = TransformerEncoderLayer(d_model=2 * hid_dim, nhead=nheads, dim_feedforward=hid_dim,
                                                     dropout=dropout_p, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, lr_num)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x):
        opt, _ = self.bilstm(x)
        opt = self.transformer_encoder(opt)
        return opt[:,-1,:]


class SequentialModule(nn.Module):
    name = "SequentialModule"
    """
    Input Shape = [Batch, T, ipt_dim]
    Output Shape = [Batch, 2 * hid_dim]
    """
    def __init__(self, ipt_dim, hid_dim, alpha, seq_len, lr_num=2, nheads=4, dropout_p=0.5):
        super(SequentialModule, self).__init__()
        self.use_MRS = True
        self.use_SCD = True
        self.seq_len = seq_len
        self.lr_num = lr_num
        self.dropout_p = dropout_p
        self.hid_dim = hid_dim
        # MRS
        self.mrs = MRS(ipt_dim=ipt_dim, hid_dim=hid_dim)
        # SCD
        self.dropout = torch.nn.Dropout(p=self.dropout_p)

        self.SubsequenceDetector = SCD(alpha, hid_dim * 2, [hid_dim * 2,hid_dim * 2], dropout_p = self.dropout_p)
        self.SubsequenceDetector_market = SCD(alpha, hid_dim * 2, [hid_dim * 2,hid_dim * 2], dropout_p = self.dropout_p)
        # RotaryTransformer
        self.SubsequenceDependencyEncoder = TransformerRotatry(dim = hid_dim * 2, n_heads = nheads, seq_len = self.seq_len, norm_eps = 1e-4, n_layers = 2)
        self.SubsequenceDependencyEncoder_market = TransformerRotatry(dim = hid_dim * 2, n_heads = nheads, seq_len = self.seq_len , norm_eps = 1e-4, n_layers = 2)

        self.out_dim = 2 * hid_dim

    def forward(self, x):
        x_market = x.mean(dim=0, keepdim=True)
        # lengths = torch.tensor([x.size(1)] * x.size(0)).to(x.device)
        stock_specific, market_induced = self.mrs(x, x_market.expand(x.shape[0], -1, -1))
        h_stock_specific = self.SubsequenceDetector(stock_specific)
        h_stock_market_induced = self.SubsequenceDetector_market(market_induced)
        v_stock_specific = self.SubsequenceDependencyEncoder(h_stock_specific)
        v_stock_market_induced  = self.SubsequenceDependencyEncoder_market(h_stock_market_induced)

        return v_stock_specific, v_stock_market_induced
