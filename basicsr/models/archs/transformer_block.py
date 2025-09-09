import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x -> b n c
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

##########################################################################
class DualGatedFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DualGatedFeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)

        self.proj_1 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.proj_2 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        proj_1 = self.proj_1(x)
        proj_2 = self.proj_2(x)
        gated_1 = proj_1 * F.sigmoid(proj_2)
        gated_2 = proj_2 * F.gelu(proj_1)
        x = self.project_out(gated_1 + gated_2)
        return x

##########################################################################
class IlluminationGuideAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(IlluminationGuideAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.svp_q_dwconv = nn.Conv2d(3, 3*self.num_heads, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim+3*self.num_heads, dim, kernel_size=1, bias=bias)
        
    def forward(self, x, svp_fea):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        svp_q = self.svp_q_dwconv(svp_fea)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        svp_q = rearrange(svp_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        svp_q = torch.nn.functional.normalize(svp_q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        q = torch.concat((q, svp_q), dim=2)
        attn = (q @ k.transpose(-2, -1)) * self.temperature     # b head c+3 c
        attn = attn.softmax(dim=-1)

        out = (attn @ v)    # b heads c+3 hw
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
class SAIGTransformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(SAIGTransformer, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = IlluminationGuideAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = DualGatedFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input: tuple):
        x, svp_fea = input

        x = x + self.attn(self.norm1(x), svp_fea)
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



