'''ResNet & VGG in PyTorch.

'''
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from torch import einsum
import numpy as np
from torch.distributions import Normal
import sys
sys.path.append(".")
#from config import BATCH
from models.query_models import GraphConvolution
import math
from functools import partial

def convnxn(in_channels, out_channels, kernel_size, stride=1, groups=1, padding=0):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)

def conv1x1(in_channels, out_channels, stride=1, groups=1):
    return convnxn(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, p, **kwargs):
        super().__init__()

        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)

        return x

    def extra_repr(self):
        return "p=%s" % repr(self.p)

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.dropout(F.relu(self.bn1(self.conv1(x))), p=0.3, training=True)
        out = F.dropout(self.bn2(self.conv2(out)), p=0.3, training=True)
        out +=  self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, avg_pool=4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.avg_pool = avg_pool

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear = nn.Linear(512, num_classes)
        # self.linear2 = nn.Linear(1000, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = nn.AdaptiveAvgPool2d(out4, self.avg_pool) #F.avg_pool2d(out4, self.avg_pool)
        outf = out.view(out.size(0), -1)
        # outl = self.linear(outf)
        out = self.linear(outf)
        return out, outf, [out1, out2, out3, out4]

class ResNetfm(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetfm, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.linear2 = nn.Linear(1000, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        outf = out.view(out.size(0), -1)
        # outl = self.linear(outf)
        out = self.linear(outf)
        return out, outf, [out1, out2, out3, out4]


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


class MLP_Block3(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.ReLU() #GELU()
        self.nn2 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)

        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.nn2(x)
        
        return x

class Attention3(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)
        self.to_qk = nn.Linear(dim, dim * 2, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        self.nn1 = nn.Linear(1024, 512)
        torch.nn.init.xavier_uniform_(self.to_qk.weight)
        torch.nn.init.zeros_(self.to_qk.bias)    
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        n, _, h = *x.shape, self.heads
        # qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        qk =  self.to_qk(x)
        # q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions
        q, k = rearrange(qk, 'n (qk h d) -> qk h n d', qk = 2, h = h)
        dots = torch.einsum('hid,hjd->hij', q, k) #* self.scale
        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = rearrange(attn, 'h n d -> n (h d) ') #concat heads into one matrix, ready for next encoder block  
        # out =  self.nn1(out)
        
        out = torch.einsum('ij,jk->ik', out, x) #* self.scale
        # out = self.do1(out)
        return out


class Transformer3(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention3(dim, heads = heads, dropout = dropout)),
                Residual(MLP_Block3(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            att_out = attention(x, mask = mask) # go to attention
            x = mlp(att_out) #go to MLP_Block
        return x


class LocalAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 window_size=7, k=1,
                 heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        self.attn = Attention2d(dim_in, dim_out,
                                heads=heads, dim_head=dim_head, dropout=dropout, k=k)
        self.window_size = window_size

        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p

        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]

        x = rearrange(x, "b c (n1 p1) (n2 p2) -> (b n1 n2) c p1 p2", p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, "(b n1 n2) c p1 p2 -> b c (n1 p1) (n2 p2)", n1=n1, n2=n2, p1=p, p2=p)

        return x, attn

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        d = i[None, :, :] - i[:, None, :]

        return d

class Attention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, k=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)

        out = self.to_out(out)

        return out, attn



class AttentionBlock(nn.Module):
    # Attention block with pre-activation.
    expansion = 1

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, sd=0.0,
                 stride=1, window_size=7, k=1, norm=nn.BatchNorm2d, activation=nn.GELU,
                 **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion

        self.shortcut = []
        if stride != 1 or dim_in != dim_out * self.expansion:
            self.shortcut.append(conv1x1(dim_in, dim_out * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.norm1 = norm(dim_in)
        self.relu = activation()

        self.conv = nn.Conv2d(dim_in, width, kernel_size=1, bias=False)
        self.norm2 = norm(width)
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.norm1(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.conv(x)
        x = self.norm2(x)
        x, attn = self.attn(x)

        x = self.sd(x) + skip

        return x

def ResNet18(num_classes = 10, avg_pool = 4):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, avg_pool)

def ResNet18fm(num_classes = 10):
    return ResNetfm(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes = 10, avg_pool = 4):
    return ResNet(BasicBlock, [3,4,6,3], num_classes, avg_pool)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, classes):
        super(VGG, self).__init__()
        self.features = features
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(512, classes),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        dropout = 0.5
        self.classifier = nn.Sequential(
                                        nn.Linear(512 * 7 * 7, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(4096, classes),
                                    )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        feat = []
        y = x
        for i, model in enumerate(self.features):
            y = model(y)
            if i in {3,5,15,20}:
            # if i in {3,5,10,15}:
                feat.append(y)#(y.view(y.size(0),-1))

        x = self.features(x)
        out4 = x.view(x.size(0), -1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, out4, [feat[0], feat[1], feat[2], feat[3]]


class VGGf(nn.Module):
    '''
    VGG feature extractor model 
    '''
    def __init__(self, features):
        super(VGGf, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        feat = []
        y = x
        for i, model in enumerate(self.features):
            y = model(y)
            if i in {3,5,15,20}:
            # if i in {3,5,10,15}:
                feat.append(y)#(y.view(y.size(0),-1))

        x = self.features(x)
        # x = self.avgpool(x)
        out4 = x.view(x.size(0), -1)

        return out4 #, [feat[0], feat[1], feat[2], feat[3]]

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

class VGGc(nn.Module):
    '''
    VGG Classifier model 
    '''
    def __init__(self, in_feat, classes):
        super(VGGc, self).__init__()
        # self.features = features
        dropout = 0.5
        self.fc1 = nn.Linear(in_feat, 512)
        self.fc2 = nn.Linear(512, classes, bias=False)
        self.temperature = 0.05
        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(512, classes),
        # )
        # self.classifier = nn.Sequential(
        #                                 nn.Linear(in_feat * 7 * 7, 4096),
        #                                 nn.ReLU(True),
        #                                 nn.Dropout(p=dropout),
        #                                 nn.Linear(4096, 4096),
        #                                 nn.ReLU(True),
        #                                 nn.Dropout(p=dropout),
        #                                 nn.Linear(4096, classes),
                                    # )
    #     self._initialize_weights()

    # def _initialize_weights(self) -> None:
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x, reverse=False, eta=0.1):
        # x = self.classifier(x)
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temperature
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(classes):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), classes)

def vgg16(classes):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), classes)

def vgg16f():
    """VGG 16-layer model (configuration "D")"""
    return VGGf(make_layers(cfg['D']))



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        feat = out.view(out.size(0), -1)
        out = self.linear(feat)

        return out, feat, feat

def Wide_ResNet28(num_classes = 10):
    return Wide_ResNet(28, 10, 0.3, num_classes)

class MLPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(MLPBlock, self).__init__()

        self.dense1 = nn.Linear(in_features, 4096, True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(4096, 4096, True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense3 = nn.Linear(4096, num_classes, True)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size()[0], -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x

class TanhBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=1e1, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(TanhBlurBlock, self).__init__()

        self.temp = temp
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.blur = blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.temp * self.tanh(x / self.temp)
        x = self.relu(x)
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


def blur(in_filters, sfilter=(1, 1), pad_mode="constant"):
    if tuple(sfilter) == (1, 1) and pad_mode in ["constant", "zero"]:
        layer = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
    else:
        layer = Blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)
    return layer

class Blur(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode="replicate", **kwargs):
        super(Blur, self).__init__()

        filter_size = len(sfilter)
        self.pad = SamePad(filter_size, pad_mode=pad_mode)

        self.filter_proto = torch.tensor(sfilter, dtype=torch.float, requires_grad=False)
        self.filter = torch.tensordot(self.filter_proto, self.filter_proto, dims=0)
        self.filter = self.filter / torch.sum(self.filter)
        self.filter = self.filter.repeat([in_filters, 1, 1, 1])
        self.filter = torch.nn.Parameter(self.filter, requires_grad=False)

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, self.filter, groups=x.size()[1])

        return x

    def extra_repr(self):
        return "pad=%s, filter_proto=%s" % (self.pad, self.filter_proto.tolist())

class SamePad(nn.Module):

    def __init__(self, filter_size, pad_mode="constant", **kwargs):
        super(SamePad, self).__init__()

        self.pad_size = [
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
        ]
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, self.pad_size, mode=self.pad_mode)

        return x

    def extra_repr(self):
        return "pad_size=%s, pad_mode=%s" % (self.pad_size, self.pad_mode)

class FeedForward(nn.Module):

    def __init__(self, dim_in, hidden_dim, dim_out=None, *,
                 dropout=0.0,
                 f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.net = nn.Sequential(
            f(dim_in, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            f(hidden_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Attention1d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn

class VGGNet2(nn.Module):

    def __init__(self, block, num_blocks, filter=3,
                 sblock=TanhBlurBlock, num_sblocks=(0, 0, 0, 0, 0),
                 cblock=MLPBlock,
                 num_classes=10, name="vgg", **block_kwargs):
        super(VGGNet2, self).__init__()

        self.name = name

        self.layer0 = self._make_layer(block, filter, 64, num_blocks[0], pool=False, **block_kwargs)
        self.layer1 = self._make_layer(block, 64, 128, num_blocks[1], pool=True, **block_kwargs)
        self.layer2 = self._make_layer(block, 128, 256, num_blocks[2], pool=True, **block_kwargs)
        self.layer3 = self._make_layer(block, 256, 512, num_blocks[3], pool=True, **block_kwargs)
        self.layer4 = self._make_layer(block, 512, 512, num_blocks[4], pool=True, **block_kwargs)

        self.smooth0 = self._make_smooth_layer(sblock, 64, num_sblocks[0], **block_kwargs)
        self.smooth1 = self._make_smooth_layer(sblock, 128, num_sblocks[1], **block_kwargs)
        self.smooth2 = self._make_smooth_layer(sblock, 256, num_sblocks[2], **block_kwargs)
        self.smooth3 = self._make_smooth_layer(sblock, 512, num_sblocks[3], **block_kwargs)
        self.smooth4 = self._make_smooth_layer(sblock, 512, num_sblocks[4], **block_kwargs)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = []
        if cblock is MLPBlock:
            # self.classifier.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.classifier.append(cblock(7 * 7 * 512, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(512, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)

    @staticmethod
    def _make_layer(block, in_channels, out_channels, num_blocks, pool, **block_kwargs):
        layers, channels = [], in_channels
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for _ in range(num_blocks):
            layers.append(block(channels, out_channels, **block_kwargs))
            channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _make_smooth_layer(sblock, in_filters, num_blocks, **block_kwargs):
        layers = []
        for _ in range(num_blocks):
            layers.append(sblock(in_filters=in_filters, **block_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        out1 = self.smooth0(x)

        x = self.layer1(out1)
        out2 = self.smooth1(x)

        x = self.layer2(out2)
        out3 = self.smooth2(x)

        x = self.layer3(out3)
        out4 = self.smooth3(x)

        x = self.layer4(out4)
        features = self.smooth4(x)


        features = self.maxpool(features)
        x = self.classifier(out4)

        return x, features, [out1, out2, out3, out4]

class vggBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **block_kwargs):
        super(vggBasicBlock, self).__init__()

        self.conv = conv3x3(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VGGNet2C(nn.Module):
    def __init__(self, in_channels, num_classes, cblock=MLPBlock, **block_kwargs):
        super(VGGNet2C, self).__init__()

        self.classifier = []
        if cblock is MLPBlock:
            # self.classifier.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.classifier.append(cblock(7 * 7 * in_channels, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(in_channels, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)
    
    def forward(self, x):
        x = self.classifier(x)
        return x

def dnn_16(num_classes=10, filter=3, name="vgg_dnn_16", **block_kwargs):
    return VGGNet2(vggBasicBlock, [2, 2, 3, 3, 3], filter,
                  num_classes=num_classes, name=name, **block_kwargs)

def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return convnxn(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1)

class Transformer(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dim_mlp=1024, dropout=0.0, sd=0.0,
                 attn=Attention1d, norm=nn.LayerNorm,
                 f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))
            self.shortcut.append(nn.Linear(dim_in, dim_out))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.norm1 = norm(dim_in)
        self.attn = attn(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd1 = DropPath(sd) if sd > 0.0 else nn.Identity()

        self.norm2 = norm(dim_out)
        self.ff = FeedForward(dim_out, dim_mlp, dim_out, dropout=dropout, f=f, activation=activation)
        self.sd2 = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        skip = self.shortcut(x)
        x = self.norm1(x)
        x, attn = self.attn(x, mask=mask)
        x = self.sd1(x) + skip

        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.sd2(x) + skip

        return x



class VTVGGNet2(nn.Module):

    def __init__(self, block, num_blocks, filter=3,
                 sblock=TanhBlurBlock, num_sblocks=(0, 0, 0, 0, 0),
                 cblock=MLPBlock,
                 num_classes=10, name="vgg", **block_kwargs):
        super(VTVGGNet2, self).__init__()

        self.name = name

        self.layer0 = self._make_layer(block, filter, 64, num_blocks[0], pool=False, **block_kwargs)
        self.layer1 = self._make_layer(block, 64, 128, num_blocks[1], pool=True, **block_kwargs)
        self.layer2 = self._make_layer(block, 128, 256, num_blocks[2], pool=True, **block_kwargs)
        self.layer3 = self._make_layer(block, 256, 512, num_blocks[3], pool=True, **block_kwargs)
        self.layer4 = self._make_layer(block, 512, 512, num_blocks[4], pool=True, **block_kwargs)

        self.smooth0 = self._make_smooth_layer(sblock, 64, num_sblocks[0], **block_kwargs)
        self.smooth1 = self._make_smooth_layer(sblock, 128, num_sblocks[1], **block_kwargs)
        self.smooth2 = self._make_smooth_layer(sblock, 256, num_sblocks[2], **block_kwargs)
        self.smooth3 = self._make_smooth_layer(sblock, 512, num_sblocks[3], **block_kwargs)
        self.smooth4 = self._make_smooth_layer(sblock, 512, num_sblocks[4], **block_kwargs)

        # self.transformer = LocalAttention(512, 512, window_size=4, heads=8)
        self.transformer = Transformer(512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = []
        if cblock is MLPBlock:
            # self.classifier.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.classifier.append(cblock(7 * 7 * 512, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(512, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)

    @staticmethod
    def _make_layer(block, in_channels, out_channels, num_blocks, pool, **block_kwargs):
        layers, channels = [], in_channels
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for _ in range(num_blocks):
            layers.append(block(channels, out_channels, **block_kwargs))
            channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _make_smooth_layer(sblock, in_filters, num_blocks, **block_kwargs):
        layers = []
        for _ in range(num_blocks):
            layers.append(sblock(in_filters=in_filters, **block_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        out1 = self.smooth0(x)

        x = self.layer1(out1)
        out2 = self.smooth1(x)

        x = self.layer2(out2)
        out3 = self.smooth2(x)

        x = self.layer3(out3)
        out4 = self.smooth3(x)

        x = self.layer4(out4)
        x = self.smooth4(x)
        x = rearrange(x, 'b h w c -> b (w c) h')
        x = self.transformer(x)
        x = rearrange(x, 'b (w c) h -> b h w c', w=2, c=2)
        features = self.maxpool(x)
        # x = self.classifier(features)

        return x, features, [out1, out2, out3, out4]

def vtdnn_16(num_classes=10, filter=3, name="vgg_dnn_16", **block_kwargs):
    return VTVGGNet2(vggBasicBlock, [2, 2, 3, 3, 2], filter,
                  num_classes=num_classes, name=name, **block_kwargs)


