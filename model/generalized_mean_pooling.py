import torch
from torch import nn
from torch.nn import functional as F

class GeM3D(nn.Module):
    def __init__(self, p=1, eps=1e-6):
        super(GeM3D,self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x.float(), p=self.p, eps=self.eps)
        
    def gem(self, x, p=1, eps=1e-6):
        return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class GeM2D(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM2D,self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x.float(), p=self.p, eps=self.eps)
        
    def gem(self, x, p=1, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class GeM1D(nn.Module):
    def __init__(self, p=3, channel_last=True, eps=1e-6):
        super(GeM1D, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.channel_last = channel_last

    def forward(self, x):
        return self.gem(x.float(), p=self.p, eps=self.eps)
        
    def gem(self, x, p=1, eps=1e-6):
        """
        x: (B, C, L) -> (B, C)
        """
        if self.channel_last:
            x = x.permute(0, 2, 1)
        
        # return F.avg_pool1d(x.clamp(min=eps), x.size(-1))
        return F.avg_pool1d(x.clamp(min=eps).pow(p), x.size(-1)).pow(1./p)
            
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

