from utils import *
from RelationalModule import *
from SequencialLayers import *


class DRSN(nn.Module):
    name = "DRSN"

    def __init__(self, ipt_dim, hid_dim, t, alpha = 1, lr_num=2, nheads=4, dropout_p=0.2, tau_init=1, sqmode = 0):
        super(DRSN, self).__init__()

        self.hid_dim = hid_dim
        self.sqmode = sqmode
        self.dropout_p = dropout_p
        self.alpha = alpha

        self.sequential_module = SequentialModule(ipt_dim,
                                                  hid_dim,
                                                  seq_len= t,
                                                  alpha = alpha,
                                                  lr_num=lr_num,
                                                  nheads=nheads,
                                                  dropout_p = self.dropout_p)

        self.use_MRS = self.sequential_module.use_MRS

        self.spillover_module = DecoupledGraphAttentionNetwork(d_market = t * ipt_dim ,
                                                               d_sequentialEmbedding = self.sequential_module.out_dim,
                                                               d_hidden = self.sequential_module.out_dim,
                                                               dropout= self.dropout_p,
                                                               alpha=0.2,
                                                               concat=False)

        self.spillover_sequential_module = BiLSTMATT(ipt_dim=ipt_dim, hid_dim=hid_dim, lr_num = lr_num)

        # self.fc = Graph_Linear(num_nodes=self.num_nodes, ipt_dim= (2 + self.use_MRS) * self.seqential_module.out_dim, hid_dim=1)
        self.fc = nn.Linear(in_features= (2 + self.use_MRS) * self.sequential_module.out_dim,
                            out_features= 1,
                            bias=True)

        self.dropout = nn.Dropout(self.dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def get_spillover_process(self, x, relation_static = None):
        t = x.size()[0]
        num_stock = x.size()[1]
        if self.use_MRS:
            stock_specific_embedding, market_induced_embedding = self.sequential_module(x.transpose(0, 1))
        else:
            stock_specific_embedding = self.sequential_module(x.transpose(0, 1))

        _, (coef, gate) = self.spillover_module(x.transpose(0,1).reshape(num_stock,-1), stock_specific_embedding)

        return coef, gate

    def update_alpha(self, cur_epoch):
        if self.sqmode == 0:
            self.sequential_module.scd.cell_1.alpha = min(5.0, self.alpha + 0.1 * cur_epoch)
            self.sequential_module.scd.cell_2.alpha = min(5.0, self.alpha + 0.1 * cur_epoch)
            self.sequential_module.scd_market.cell_1.alpha = min(5.0, self.alpha + 0.1 * cur_epoch)
            self.sequential_module.scd_market.cell_2.alpha = min(5.0, self.alpha + 0.1 * cur_epoch)

    def forward(self, x,  relation_static = None, cur_epoch=0):
        """
        Input Shape: T * num_stock *  ipt_dim
        OutPut Shape: num_stock * 1
        """
        t = x.size()[0]
        num_stock = x.size()[1]
        x = x.transpose(0, 1) # T * num_stock *  ipt_dim ->  num_stock * T * ipt_dim

        # num_stock * T * ipt_dim ->  num_stock * 2 hid_dim
        if self.use_MRS:
            stock_specific_embedding, market_induced_embedding = self.sequential_module(x)
        else:
            stock_specific_embedding = self.sequential_module(x)
        # num_stock, T * ipt_dim,  num_stock * 2 hid_dim ->  num_stock T * ipt_dim
        spillover, _ = self.spillover_module(x.reshape(num_stock,-1), stock_specific_embedding)
        # num_stock T * ipt_dim ->  num_stock T * 2 hid_dim
        f_spillover = self.spillover_sequential_module(spillover.reshape(num_stock, t ,-1))
        # num_stock T * 6 hid_dim  ->  num_stock T * 1
        if self.use_MRS:
            output_layer_input = torch.cat([stock_specific_embedding, market_induced_embedding, f_spillover], dim=-1)
        else:
            output_layer_input= torch.cat([stock_specific_embedding, f_spillover], dim=-1)
        output_layer_input = self.dropout(output_layer_input)
        output = self.fc(output_layer_input)
        return output