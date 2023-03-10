import math
import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from einops import rearrange

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        a = self.fn(x, **kwargs)
        return a + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) -> b h d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h d -> b (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LayerNormalize(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                LayerNormalize(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class LossNet(nn.Module):
    def __init__(self, feature_sizes=[28, 14, 7, 3], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()
        
        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)
    
    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)        
        #x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return torch.sigmoid(x), feat, torch.cat((feat,x),1)

class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)        
        #x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x, feat, torch.cat((feat,x),1)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, f_filt=4):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*2*2)),                                 # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024*4*4),                           # B, 1024*8*8
            View((-1, 1024, 4, 4)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, self.f_filt, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class DSN(nn.Module):
	"""Deep Summarization Network"""
	def __init__(self,in_dim =19,hid_dim=256, num_layers=1):
		super(DSN, self).__init__()
		# in_dim = in_dim*in_dim # for semantic segementation
		in_dim = in_dim 
		self.lstm = nn.LSTM(in_dim, hid_dim, num_layers=1, bidirectional=True, batch_first=True)
		self.fc2 = nn.Linear(hid_dim*2,hid_dim*2)
		self.fc1 = nn.Linear(hid_dim*2,1)
		

	def forward(self, x,in_dim=19):
		
		# x=torch.reshape(x,(x.shape[0],x.shape[1],in_dim*in_dim)) # for semantic segmentation
		x = x.float()
		x,_=self.lstm(x)
		x= self.fc2(x)
		x= self.fc1(x)
		p = torch.sigmoid(x)
		return p

class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)

class Discriminator1(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator1, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1),
            # nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)

class Discriminator2(nn.Module):
    """Discriminator"""
    def __init__(self, z_dim=10):
        super(Discriminator2, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class GradReverse(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class Trans_disc(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Trans_disc, self).__init__()
        self.z_dim = z_dim
        self.temperature = 0.05
        # self.net = nn.Sequential(
        #                     Transformer(z_dim+1, 2, 2, 64, 128, 0.),
        #                     nn.Linear(z_dim+1, 512),
        #                     nn.ReLU(True)
        #             )
        self.fc2 = nn.Sequential(
                                nn.Linear(512, 512),
                                nn.ReLU(True))
        self.fc3 = nn.Sequential(
                                nn.Linear(512, 1),
                                nn.Sigmoid()
                        )
        self.weight_init()
        self.pos_embedding = nn.Parameter(torch.randn(128, z_dim + 1))
        self.cls_token = nn.Parameter(torch.randn(128, 1))
        self.dropout = nn.Dropout(0.)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    # def forward(self, x):
    #     b, n= x.shape
    #     cls_tokens = self.cls_token
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x += self.pos_embedding[:, :(n + 1)]
    #     x = self.dropout(x)
    #     return self.net(x)

    def forward(self, x, reverse=False, eta=0.1):
        # x = self.classifier(x)
        # b, n= x.shape
        # cls_tokens = self.cls_token
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)

        # x = self.net(x)
        if reverse:
            x = grad_reverse(x, eta)
        # x = F.normalize(x)
        x_out = self.fc2(x) / self.temperature
        x_out = self.fc3(x_out)
        # x_out = nn.Sigmoid(x_out)
        return x_out

class Discriminator3(nn.Module):
    def __init__(self, feature_sizes=[28, 14, 7, 3], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(Discriminator3, self).__init__()
        
        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 512)
        self.output = nn.Linear(512, 1)
        self.weight_init()
    
    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        pre_out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        out     = self.output(pre_out)
        out     = torch.sigmoid(out)
        return out

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)