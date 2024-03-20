import math
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .curope import cuRoPE2D
    RoPE2D = cuRoPE2D
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')

    class RoPE2D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq 
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D,seq_len,device,dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos() # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D,seq_len,device,dtype] = (cos,sin)
            return self.cache[D,seq_len,device,dtype]
            
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim==2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
            x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens


class PositionEncodingSine1D(nn.Module):
    def __init__(self, d_model, max_seq_len=20_000):
        super(PositionEncodingSine1D, self).__init__()
        self.d_model = d_model
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # 1 x max_seq_len x d_model
    
    def forward(self, x):
        """
        x: BxLxC
        """
        x = x + self.pe[:, :x.size(1)]
        return x
    

# Position encoding for query image
class PositionEncodingSine2D(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(1280, 960)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        max_shape = tuple(max_shape)

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float()
            * (-math.log(10000.0) / d_model // 2)
        )
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class PositionEncodingSine3D(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 3-dimensional cubes
    """

    def __init__(self, d_model, max_shape=(128, 128, 128)):
        """
        Args:
            max_shape (tuple): for DxHxW cube
        """
        super().__init__()

        assert(d_model % 6 == 0), "d_model must be divisible by 6 for 3D sinusoidal position encoding"
        max_shape = tuple(max_shape)

        pe = torch.zeros((d_model, *max_shape)) # CxDxHxW
        z_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        y_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(2).float().unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 3).float()
            * (-math.log(10000.0) / d_model // 2)
        )
        div_term = div_term[:, None, None, None]  # [C//6, 1, 1, 1]

        pe[0::6, :, :, :] = torch.sin(x_position * div_term)
        pe[1::6, :, :, :] = torch.cos(x_position * div_term)
        pe[2::6, :, :, :] = torch.sin(y_position * div_term)
        pe[3::6, :, :, :] = torch.cos(y_position * div_term)
        pe[4::6, :, :, :] = torch.sin(z_position * div_term)
        pe[5::6, :, :, :] = torch.cos(z_position * div_term)
        
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, D, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, D, H, W]
        """
        assert(x.dim() == 5), "Input must be 5-dimensional"
        return x + self.pe[:, :, :x.size(-3), :x.size(-2), :x.size(-1)]

# Position encoding for 3D points
class PositionEncodingLinear3D(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs """

    def __init__(self, inp_dim, feature_dim, layers, norm_method="batchnorm"):
        super().__init__()
        self.encoder = self.MLP([inp_dim] + list(layers) + [feature_dim], norm_method)

        self.encoder
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, descriptors):
        """
        kpts: B*L*3 or B*L*4
        descriptors: B*C*L
        """
        # inputs = kpts  # B*L*3

        return descriptors + self.encoder(kpts).transpose(2, 1).expand_as(descriptors)  # B*C*L

    def MLP(self, channels: list, norm_method="batchnorm"):
        """ Multi-layer perceptron"""
        n = len(channels)
        layers = []
        for i in range(1, n):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias=True))
            if i < n - 1:
                if norm_method == "batchnorm":
                    layers.append(nn.BatchNorm1d(channels[i]))
                elif norm_method == "layernorm":
                    layers.append(nn.LayerNorm(channels[i]))
                elif norm_method == "instancenorm":
                    layers.append(nn.InstanceNorm1d(channels[i]))
                else:
                    raise NotImplementedError
                    # layers.append(nn.GroupNorm(channels[i], channels[i])) # group norm
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)


# Position encoding for 3D points
class KeypointEncoding_linear(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs """

    def __init__(self, inp_dim, feature_dim, layers, norm_method="batchnorm"):
        super().__init__()
        self.encoder = self.MLP([inp_dim] + list(layers) + [feature_dim], norm_method)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, descriptors):
        """
        kpts: B*L*3 or B*L*4
        descriptors: B*C*L
        """
        # inputs = kpts  # B*L*3
        return descriptors + self.encoder(kpts).transpose(2, 1)  # B*C*L

    def MLP(self, channels: list, norm_method="batchnorm"):
        """ Multi-layer perceptron"""
        n = len(channels)
        layers = []
        for i in range(1, n):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias=True))
            if i < n - 1:
                if norm_method == "batchnorm":
                    layers.append(nn.BatchNorm1d(channels[i]))
                elif norm_method == "layernorm":
                    layers.append(nn.LayerNorm(channels[i]))
                elif norm_method == "instancenorm":
                    layers.append(nn.InstanceNorm1d(channels[i]))
                else:
                    raise NotImplementedError
                    # layers.append(nn.GroupNorm(channels[i], channels[i])) # group norm
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)