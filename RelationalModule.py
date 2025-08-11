from utils import *
class DecoupledGraphAttentionNetwork(nn.Module):
    #two-times leakyrelu in the element-wise risk propagation
    def __init__(self, d_market, d_sequentialEmbedding, d_hidden, dropout=0.2, alpha=0.2, concat=True):
        super(DecoupledGraphAttentionNetwork, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.d_market = d_market
        self.d_hidden = d_hidden
        self.d_sequentialEmbedding = d_sequentialEmbedding
        self.concat = concat
        # implicit relation
        self.seq_transformation_r = nn.Parameter(torch.empty(size=(d_sequentialEmbedding, d_hidden)))
        self.f = nn.Parameter(torch.empty(size=(2 * d_hidden, int(d_hidden / 2))))
        self.a = nn.Parameter(torch.empty(size=(int(d_hidden / 2), 1)))
        # nn.init.xavier_uniform_(self.a, gain=1.414)
        # gate
        self.seq_transformation_g = nn.Parameter(torch.empty(size=(d_sequentialEmbedding, d_hidden)))
        self.w = nn.Parameter(torch.empty(size=(2 * d_hidden, d_hidden)))
        self.g = nn.Parameter(torch.empty(size=(d_hidden, d_market)))
        # static relation
        self.num_stock = None
        self.coef_revise = False
        self.W_static = None

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def get_gate(self, seq_s):
        """
        seq_s: shape n * f
        gate: shape n_x * n_y * d
        the gate from y to x
        """
        seq_s = self.leakyrelu(torch.matmul(seq_s, self.seq_transformation_g))
        transform_1 = self.leakyrelu(torch.matmul(seq_s, self.w[:self.d_hidden, :]))
        transform_2 = self.leakyrelu(torch.matmul(seq_s, self.w[self.d_hidden:, :]))
        transform = transform_1.unsqueeze(1) * transform_2
        gate = torch.matmul(transform, self.g)
        gate = torch.tanh(gate)
        gate = self.dropout(gate)
        return gate

    def forward(self, input_s, input_r, relation_static=None):
        """
        input_s: orignal market signal t * n * d1
        input_r: subsequence-based embedding n * d2
        """
        # infer relation
        coefs = None
        # infer gate
        gate = self.get_gate(input_r)
        # feature-level decrimination
        message_gated = input_s * gate
        # node leval descrimation
        spillovers = message_gated.sum(dim = 1)
        return spillovers, (coefs, gate)