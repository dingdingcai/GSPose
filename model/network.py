import os
import sys
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

# insert the project root of this project to the python path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_ROOT)

from .blocks import PositionGetter
from .position_encoding import RoPE2D, PositionEncodingSine2D
from .generalized_mean_pooling import GeM2D as GeMeanPool2d
from .dino_layers import (MemEffSelfAttention,
                                MemEffCrossAttention, 
                                MemEffEncoderLayer, 
                                MemEffDecoderLayer)


class model_arch(nn.Module):
    def __init__(self, cfg=None):
        super(model_arch, self).__init__()
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        if cfg is None:
            cfg = dict()
        self.img_size = cfg.get('img_size', 224)   # image size
        self.batch_size = cfg.get('batch_size', 1) # batch size        
        self.temperature = cfg.get('temperature', 0.1) # softmax temperature
        self.pose_feat_dim = cfg.get('pose_feat_dim', 256)
        
        self.pos_embed = cfg.get('pos_embed', 'RoPE100') # ['cosine', 'RoPE100']
        self.dino_patch_size = cfg.get('dino_patch_size', 14)
        
        self.coseg_feat_dim = cfg.get('coseg_feat_dim', 256)
        self.coseg_num_heads = cfg.get('coseg_num_heads', 8)
        self.coseg_mlp_ratio = cfg.get('coseg_mlp_ratio', 4)
        self.coseg_num_layers = cfg.get('coseg_num_layers', 4)

        self.backbone_feat_dim = 768
        self.dino_block_indices = [2, 5, 8, 11] # the 3nd, 6th, 9th, 12th blocks
        self.dino_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
        for param in self.dino_backbone.parameters():
            param.requires_grad = False
        
        self.pixelshuffle_x2 = nn.PixelShuffle(2)
        self.pixelshuffle_x7 = nn.PixelShuffle(7)
        self.dino_backbone_transition = nn.ModuleDict()
        for blk_idx in self.dino_block_indices:
            self.dino_backbone_transition[str(blk_idx)] = nn.Sequential(
                nn.Linear(self.backbone_feat_dim, self.backbone_feat_dim),
                nn.LayerNorm(self.backbone_feat_dim),
                nn.GELU(),
            )
        
        self.coseg_aware_projection = nn.Sequential(
            nn.Conv2d(self.backbone_feat_dim, self.coseg_feat_dim, 1, 1, 0),
            nn.GroupNorm(1, self.coseg_feat_dim),
            nn.LeakyReLU(inplace=True),
        )
        
        if self.pos_embed == 'cosine':
            self.coseg_position_encoding = PositionEncodingSine2D(d_model=self.coseg_feat_dim)
            self.rope = None # nothing for cosine 
        elif self.pos_embed.startswith('RoPE'): # eg RoPE100 
            self.coseg_position_encoding = None # nothing to add in the encoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it")
            freq = float(self.pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError('Unknown pos_embed ' + self.pos_embed)

        self.coseg_refer_LayerNorm = nn.LayerNorm(self.coseg_feat_dim)
        self.coseg_query_LayerNorm = nn.LayerNorm(self.coseg_feat_dim)

        self.coseg_refer_selfAttn_groupwise = nn.ModuleList(
            [
                MemEffEncoderLayer(
                    attn_class=MemEffSelfAttention,
                    rope=self.rope,
                    dim=self.coseg_feat_dim,
                    num_heads=self.coseg_num_heads,
                    mlp_ratio=self.coseg_mlp_ratio,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                    )
                for _ in range(self.coseg_num_layers)
        ])

        self.coseg_refer_selfAttn_framewise = nn.ModuleList(
            [
                MemEffEncoderLayer(
                    attn_class=MemEffSelfAttention,
                    rope=self.rope,
                    dim=self.coseg_feat_dim,
                    num_heads=self.coseg_num_heads,
                    mlp_ratio=self.coseg_mlp_ratio,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                    )
                for _ in range(self.coseg_num_layers)
        ])

        self.coseg_refer_crossAttn = nn.ModuleList(
            [
                MemEffDecoderLayer(
                    attn_class=MemEffCrossAttention,
                    rope=self.rope,
                    dim=self.coseg_feat_dim,
                    num_heads=self.coseg_num_heads,
                    mlp_ratio=self.coseg_mlp_ratio,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                    )
                for _ in range(self.coseg_num_layers)
        ])

        self.coseg_query_selfAttn = nn.ModuleList(
            [
                MemEffEncoderLayer(
                    attn_class=MemEffSelfAttention,
                    rope=self.rope,
                    dim=self.coseg_feat_dim,
                    num_heads=self.coseg_num_heads,
                    mlp_ratio=self.coseg_mlp_ratio,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                    )
                for _ in range(self.coseg_num_layers)
        ])
        self.coseg_query_crossAttn = nn.ModuleList(
            [
                MemEffDecoderLayer(
                    attn_class=MemEffCrossAttention,
                    rope=self.rope,
                    dim=self.coseg_feat_dim,
                    num_heads=self.coseg_num_heads,
                    mlp_ratio=self.coseg_mlp_ratio,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                    )
                for _ in range(self.coseg_num_layers)
        ])

        self.norm = nn.GroupNorm(1, self.coseg_feat_dim)
        self.coseg_prediction_head = nn.Sequential(
            nn.Conv2d(self.coseg_feat_dim, self.coseg_feat_dim, 3, 1, 1),
            nn.GroupNorm(1, self.coseg_feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.coseg_feat_dim, 49, 3, 1, 1),
        )

        self.pose_aware_projection = nn.Sequential(
            nn.Conv2d(self.backbone_feat_dim, self.pose_feat_dim, 1, 1, 0),
            nn.GroupNorm(1, self.pose_feat_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.rotation_embedding_head = nn.Sequential(
            nn.Conv2d(self.pose_feat_dim + 1, 128, 3, 1, 1), # Cx32x32 -> 128x32x32
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), # 128x32x32 -> 256x16x16
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 128, 3, 1, 1), # 256x16x16 -> 128x16x16
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), # 128x16x16 -> 256x8x8
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(inplace=True),

            GeMeanPool2d(1), # 128x8x8 -> 128x1x1
            nn.Flatten(1),

            nn.Linear(256, 64),
        )
    
    def extract_DINOv2_feature(self, x, return_last_dino_feat=False):
        """
        input: x: [BV, C, H, W]
        output: feat_maps: [BV, C, H/8, W/8]
        """
        dim_B, _, dim_H, dim_W = x.shape
        dim_h = dim_H // self.dino_patch_size
        dim_w = dim_W // self.dino_patch_size

        xs = list()
        x = self.dino_backbone.prepare_tokens_with_masks(x) #  Bx3xHxW -> Bx(1+HW)xC
        for blk_idx in range(len(self.dino_backbone.blocks)):
            x = self.dino_backbone.blocks[blk_idx](x) # Bx(1+HW)xC -> Bx(1+L)xC
            if blk_idx in self.dino_block_indices:
                new_x = x[:, 1:, :]
                new_x = self.dino_backbone_transition[str(blk_idx)](new_x) # BxLxC -> BxLxC
                xs.append(new_x)
        xs = torch.cat(xs, dim=-1) # list of [BxLxC]  -> BxLx4C
        xs = xs.view(dim_B, dim_h, dim_w, -1).permute(0, 3, 1, 2) # BxLx4C -> Bx4Cx16x16
        xs = self.pixelshuffle_x2(xs) # Bx4Cx16x16 -> BxCx32x32
        if return_last_dino_feat:
            x_norm = self.dino_backbone.norm(x)
            x_norm_clstoken = x_norm[:, 0, :] # BxC
            x_norm_patchtokens = x_norm[:, 1:, :] # BxLxC
            return xs, x_norm_clstoken, x_norm_patchtokens
        return xs
    
    def refer_cosegmentation(self, x):
        """
        x: BVxCx32x32
        """
        assert(x.dim() == 4), 'x: {}'.format(x.shape)
        dim_B = self.batch_size
        dim_BVr, _, _, dim_S = x.shape
        dim_Vr = dim_BVr // self.batch_size
        x = self.coseg_aware_projection(x)
        center_coords = torch.zeros((dim_BVr, 2)).to(x.device)      # reference images are object-centric
        x_obj = F.grid_sample(x, center_coords.view(-1, 1, 1, 2), # BVrx1x1x2
                                    mode='bilinear', padding_mode='zeros', align_corners=True,
                                    ).view(dim_B, dim_Vr, -1) # BVrxCxSxS -> BxVrxC
                
        if self.coseg_position_encoding is not None:
            x = self.coseg_position_encoding(x)
        elif self.rope is not None:
            xpos = self.position_getter(x.size(0), x.size(2), x.size(3), x.device) # BVrxLx2
        else:
            xpos = None
        
        x = x.flatten(2).transpose(-2, -1).contiguous()  # BVrxCx32x32 -> BVrxLxC
        xpos = xpos.reshape(dim_BVr, -1, xpos.shape[-1]) # BxVrLx2 -> BVrxLx2

        for lix in range(self.coseg_num_layers):
            x = x.reshape(dim_BVr, -1, x.shape[-1])                      # BxVrLxC -> BVrxLxC
            xpos = xpos.reshape(dim_BVr, -1, xpos.shape[-1])             # BxVrLx2 -> BVrxLx2
            x = self.coseg_refer_selfAttn_framewise[lix](x=x, xpos=xpos) # BVrxLxC -> BVrxLxC

            x = x.reshape(dim_B, -1, x.shape[-1])                        # BVrxLxC -> BxVrLxC
            xpos = xpos.reshape(dim_B, -1, xpos.shape[-1])               # BVrxLx2 -> BxVrLx2
            x = self.coseg_refer_crossAttn[lix](x=x, xpos=xpos, y=x_obj) # BxVrLxC, BxVrxC -> BxVrLxC
            x = self.coseg_refer_selfAttn_groupwise[lix](x=x, xpos=xpos) # BxVrLxC -> BxVrLxC
        
        x = self.coseg_refer_LayerNorm(x)                            # BxVrLxC        
        x = x.reshape(dim_BVr, dim_S, dim_S, -1).permute(0, 3, 1, 2) # BVrxLxC -> BVrxCxSxS
        x = self.coseg_prediction_head(x) # BVrxCx32x32 -> BVrx2x32x32
        x = self.pixelshuffle_x7(x)       # BVrx49x32x32 -> BVrx1x224x224
        return x

    def query_cosegmentation(self, x_que, x_ref, ref_mask, enable_mask_filter=False):
        """
        input params:
            x_que: query features, BVqxCxhxw
            x_ref: refer features, BVrxCxsxs
            ref_mask: refer masks, BVrx1xSxS
        return:
            x_que: query masks,     BVqx1xHxW
        """
        assert(x_ref.dim() == 4), 'x_ref: {}'.format(x_ref.shape)

        x_ref = self.coseg_aware_projection(x_ref)
        x_que = self.coseg_aware_projection(x_que)
        dim_C = x_ref.shape[1]
        dim_B = self.batch_size

        if self.coseg_position_encoding is not None:
            x_ref = self.coseg_position_encoding(x_ref)
            x_que = self.coseg_position_encoding(x_que)
        elif self.rope is not None:
            que_pos = self.position_getter(x_que.size(0), x_que.size(2), x_que.size(3), x_que.device) # BVqxMx2
            ref_pos = self.position_getter(x_ref.size(0), x_ref.size(2), x_ref.size(3), x_ref.device) # BVrxLx2
        else:
            que_pos = None
            ref_pos = None
        
        if ref_mask.shape[-1] != x_ref.shape[-1]:
            ref_mask = F.interpolate(ref_mask, size=x_ref.shape[2:], mode='bilinear', align_corners=True)        
        
        x_ref = x_ref * ref_mask.round()     # BVrxCxSxS, BVrx1xSxS -> BVrxCxSxS 
        x_ref = x_ref.flatten(2).transpose(2, 1).contiguous() # BVrxCxSxS -> BVrxCxL -> BVrxLxC
        x_ref = x_ref.view(dim_B, -1, dim_C) # BVrxLxC -> BxVrLxC
        ref_pos = ref_pos.view(dim_B, -1, ref_pos.shape[-1]) # BVrxLx2 -> BxVrLx2

        if enable_mask_filter:
            mask_inds = ref_mask.round().view(dim_B, -1, 1) > 0.5 # BVrx1xSxS -> BVrxSxS -> BxVrLx1
            x_ref = x_ref[mask_inds.repeat(1, 1, x_ref.shape[-1])].reshape(dim_B, -1, x_ref.shape[-1]) # BxVrLxC -> BxMxC
            ref_pos = ref_pos[mask_inds.repeat(1, 1, ref_pos.shape[-1])].reshape(dim_B, -1, ref_pos.shape[-1]) # BxVrLx2 -> BxMx2

        assert(x_que.dim() == 4), 'x_que: {}'.format(x_que.shape)        
        dim_BVq, dim_C, dim_H, dim_W = x_que.shape
        x_que = x_que.flatten(2).transpose(2, 1).contiguous()  # BVrxCxSxS -> BVrxLxC
        que_pos = que_pos.view(dim_BVq, -1, que_pos.shape[-1]) # BVqxVqHWx2 -> BVqxHWx2
        for lix in range(self.coseg_num_layers):
            
            x_que = x_que.view(dim_BVq, -1, dim_C)                        # BVqxVqHWxC -> BVqxHWxC
            que_pos = que_pos.view(dim_BVq, -1, que_pos.shape[-1])        # BVqxVqHWx2 -> BVqxHWx2
            x_que = self.coseg_query_selfAttn[lix](x=x_que, xpos=que_pos) # BVqxHWxC -> BVqxHWxC

            x_que = x_que.view(dim_B, -1, dim_C)                           # BVqxHWxC -> BxVqHWxC
            que_pos = que_pos.view(dim_B, -1, que_pos.shape[-1])           # BVqxHWx2 -> BxVqHWx2
            x_que = self.coseg_query_crossAttn[lix](x=x_que, xpos=que_pos, y=x_ref, ypos=ref_pos) # BxVqHWxC -> BxVqHWxC
        
        x_que = self.coseg_query_LayerNorm(x_que) # BVqxHWxC

        x_que = x_que.view(dim_BVq, dim_H, dim_W, -1).permute(0, 3, 1, 2)  # BVrxLxC -> BVrxCxSxS
        x_que = self.coseg_prediction_head(x_que) # BVrxCx32x32 -> BVrx49x32x32
        x_que = self.pixelshuffle_x7(x_que)       # BVrx49x32x32 -> BVrx1x224x224
        return x_que
    
    def generate_rotation_aware_embedding(self, x, mask):
        """
        x: BxCxSxS
        """
        assert(x.dim() == 4), 'x: {}'.format(x.shape)
        assert(mask.dim() == 4), 'mask: {}'.format(mask.shape)
        if x.shape[-1] != mask.shape[-1]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=True)

        assert(x.shape[2:] == mask.shape[2:]), 'x: {}, mask: {}'.format(x.shape, mask.shape)

        x = self.pose_aware_projection(x)   # BxCxSxS -> BxCxSxS
        x = torch.cat([x, mask], dim=1)     # Bx(C+1)xSxS
        x = self.rotation_embedding_head(x) # BxCxSxS -> BxC
        x = F.normalize(x.float(), dim=-1, eps=1e-8) # large eps to avoid nan when using mixed precision
        return x

    def CoSegmentation(self, data):
        ref_GT_mask = data['ref_GT_mask']
        que_GT_mask = data['que_GT_mask']
        qnnb_GT_mask = data['qnnb_GT_mask']
        rand_GT_mask = data['rand_GT_mask']
        que_full_GT_mask = data['que_full_GT_mask']

        ref_feat = data['ref_feat']
        que_feat = data['que_feat'] # BVqxCxSxS
        qnnb_feat = data['qnnb_feat']
        rand_feat = data['rand_feat']
        que_full_feat = data.pop('que_full_feat')
        
        ref_pd_mask_logit = self.refer_cosegmentation(ref_feat) # BVrxCx32x32 -> BVrx1xSxS
        ref_mask_loss = self.BCE_criterion(ref_pd_mask_logit.float(), ref_GT_mask.float())

        ref_pd_mask = ref_pd_mask_logit.float().sigmoid()

        que_full_pd_mask = self.query_cosegmentation(
            x_que=que_full_feat, x_ref=ref_feat, ref_mask=ref_pd_mask) # BVqxCxSxS -> BVqx1xSxS
        que_full_mask_loss = self.BCE_criterion(que_full_pd_mask.float(), que_full_GT_mask.float())

        zoom_mask_loss = list()
        que_pd_mask_logit = self.query_cosegmentation(
            x_que=que_feat, x_ref=ref_feat, ref_mask=ref_pd_mask) # BVqxCxSxS -> BVqx1xSxS
        que_mask_loss = self.BCE_criterion(que_pd_mask_logit.float(), que_GT_mask.float())
        zoom_mask_loss.append(que_mask_loss)

        qnnb_pd_mask_logit = self.query_cosegmentation(
            x_que=qnnb_feat, x_ref=ref_feat, ref_mask=ref_pd_mask) # BVdxCxSxS -> BVdx1xSxS
        qnnb_mask_loss = self.BCE_criterion(qnnb_pd_mask_logit.float(), qnnb_GT_mask.float())
        zoom_mask_loss.append(qnnb_mask_loss)

        rand_pd_mask_logit = self.query_cosegmentation(
            x_que=rand_feat,  x_ref=ref_feat, ref_mask=ref_pd_mask) # BVdxCxSxS -> BVdx1xSxS
        rand_mask_loss = self.BCE_criterion(rand_pd_mask_logit.float(), rand_GT_mask.float())
        zoom_mask_loss.append(rand_mask_loss)

        zoom_mask_loss = torch.stack(zoom_mask_loss, dim=0).mean()

        data["ref_pd_mask"] = ref_pd_mask_logit.sigmoid()
        data["que_pd_mask"] = que_pd_mask_logit.sigmoid()

        data["qnnb_pd_mask"] = qnnb_pd_mask_logit.sigmoid()
        data["rand_pd_mask"] = rand_pd_mask_logit.sigmoid()
        
        data["que_full_pd_mask"] = que_full_pd_mask.sigmoid()

        zoom_hei, zoom_wid = que_GT_mask.shape[-2:]
        full_hei, full_wid = que_full_GT_mask.shape[-2:]
        scale_ratio = full_hei * full_wid / zoom_hei / zoom_wid

        data["rm_loss"] = ref_mask_loss
        data["cm_loss"] = zoom_mask_loss
        data["qm_loss"] = que_full_mask_loss * scale_ratio # balance the loss due to varied resolution
 
    def forward(self, data):
        dim_B = self.batch_size

        que_full_rgb = data['query_dict']['rescaled_image']    # BVqx3xHxW
        que_full_GT_mask = data['query_dict']['rescaled_mask'] # BVqx1xHxW

        x_que = data['query_dict']['dzi_image']        # BVqx3xSxS
        que_GT_mask = data['query_dict']['dzi_mask']   # BVqx1xSxS
        que_GT_alloR = data['query_dict']['allo_RT'][..., :3, :3] # BVqx3x3
        dim_BVq = x_que.shape[0]
        dim_Vq = dim_BVq // dim_B

        x_qnnb = data['qnnb_dict']['zoom_image']       # BVqVnx3xSxS
        qnnb_GT_mask = data['qnnb_dict']['zoom_mask']  # BVqVnx1xSxS
        qnnb_GT_alloR = data['qnnb_dict']['allo_RT'][..., :3, :3] # BVqVnx3x3
        dim_VqVn = x_qnnb.shape[0] // dim_B
        dim_Vn = dim_VqVn // dim_Vq

        x_ref = data['refer_dict']['zoom_image']       # BVrx3xSxS
        ref_GT_mask = data['refer_dict']['zoom_mask']  # BVrx1xSxS
        ref_GT_alloR = data['refer_dict']['allo_RT'][..., :3, :3] # BVrx3x3
        dim_Vr = x_ref.shape[0] // dim_B

        dim_Vd = 0
        x_rand = data['rand_dict']['zoom_image']       # BVdx3xSxS
        rand_GT_mask = data['rand_dict']['zoom_mask']  # BVdx1xSxS
        rand_GT_alloR = data['rand_dict']['allo_RT'][..., :3, :3] # BVrx3x3
        dim_Vd = x_rand.shape[0] // dim_B
                
        dim_S = x_ref.shape[-1]
        x = torch.cat([x_que.view(dim_B, -1, 3, dim_S, dim_S),  # BxVqx3xSxS
                       x_qnnb.view(dim_B, -1, 3, dim_S, dim_S), # BxVqVnx3xSxS
                       x_rand.view(dim_B, -1, 3, dim_S, dim_S), # BxVdx3xSxS
                       x_ref.view(dim_B, -1, 3, dim_S, dim_S),  # BxVrx3xSxS
        ], dim=1).flatten(0, 1) # -> B(Vq+VqVn+Vd+Vr)x3xSxS
        backbone_feat = self.extract_DINOv2_feature(x) # B(Vq+VqVn+Vd+Vr)x3xSxS -> B(Vq+VqVn+Vd+Vr)xCx32x32

        backbone_feat = backbone_feat.view(dim_B, -1, *backbone_feat.shape[1:]) # ==> Bx(Vq+VqVn+Vd+Vr)xCx32x32
        assert(backbone_feat.shape[1] == dim_Vq + dim_VqVn + dim_Vd + dim_Vr), backbone_feat.shape
        que_feat = backbone_feat[:, :dim_Vq].flatten(0, 1)                  # ==> BVqxCx32x32
        ref_feat = backbone_feat[:, -dim_Vr:].flatten(0, 1)                 # ==> BVrxCx32x32
        qnnb_feat = backbone_feat[:, dim_Vq:dim_Vq+dim_VqVn].flatten(0, 1)  # ==> BVqVnxCx32x32
        rand_feat = backbone_feat[:, dim_Vq+dim_VqVn:-dim_Vr].flatten(0, 1) # ==> BVdxCx32x32
        
        data_dict = {}
        data_dict["ref_feat"] = ref_feat   # BVrxCx32x32
        data_dict["que_feat"] = que_feat   # BVqxCx32x32
        data_dict["qnnb_feat"] = qnnb_feat # BVqVnxCx32x32
        data_dict["rand_feat"] = rand_feat # BVdxCx32x32
        data_dict["que_full_feat"] = self.extract_DINOv2_feature(que_full_rgb)

        data_dict["ref_GT_mask"] = ref_GT_mask
        data_dict["que_GT_mask"] = que_GT_mask
        data_dict["qnnb_GT_mask"] = qnnb_GT_mask
        data_dict["rand_GT_mask"] = rand_GT_mask
        data_dict["que_full_GT_mask"] = que_full_GT_mask

        self.CoSegmentation(data_dict)

        data_dict['ref_GT_alloR'] = ref_GT_alloR   # BVrx3x3
        data_dict['que_GT_alloR'] = que_GT_alloR   # BVqx3x3
        data_dict['qnnb_GT_alloR'] = qnnb_GT_alloR # BVqVnx3x3
        data_dict['rand_GT_alloR'] = rand_GT_alloR # BVqx3x3

        self.RotationEmbedding(data_dict)

        return data_dict

    def RotationEmbedding(self, data):
        dim_B = self.batch_size
        
        que_feat = data['que_feat']         # BVqxCx32x32
        qnnb_feat = data['qnnb_feat']       # BVqVnxCx32x32
        que_pd_mask = data['que_pd_mask']   # BVqx1xSxS
        qnnb_pd_mask = data['qnnb_pd_mask'] # BVqxVnx1xSxS

        ref_GT_alloR = data['ref_GT_alloR']   # BVrx3x3
        que_GT_alloR = data['que_GT_alloR']   # BVqx3x3
        qnnb_GT_alloR = data['qnnb_GT_alloR'] # BVqVnx3x3
        rand_GT_alloR = data['rand_GT_alloR'] # BVdx3x3

        dim_BVq = que_feat.shape[0]
        dim_Vq = que_feat.shape[0] // dim_B
        dim_Vn = qnnb_feat.shape[0] // que_feat.shape[0]
        que_feat = que_feat.view(dim_B, -1, *que_feat.shape[1:])    # ==> BxVqxCx32x32
        qnnb_feat = qnnb_feat.view(dim_B, -1, *qnnb_feat.shape[1:]) # ==> BxVqVnxCx32x32
        que_pd_mask = que_pd_mask.view(dim_B, -1, *que_pd_mask.shape[1:])
        qnnb_pd_mask = qnnb_pd_mask.view(dim_B, -1, *qnnb_pd_mask.shape[1:])

        ref_feat = data['ref_feat']
        rand_feat = data['rand_feat']
        ref_pd_mask = data['ref_pd_mask']      
        rand_pd_mask = data['rand_pd_mask']  
        ref_feat = ref_feat.view(dim_B, -1, *ref_feat.shape[1:])    # ==> BxVrxCx32x32
        rand_feat = rand_feat.view(dim_B, -1, *rand_feat.shape[1:]) # ==> BxVdxCx32x32
        ref_pd_mask = ref_pd_mask.view(dim_B, -1, *ref_pd_mask.shape[1:])
        rand_pd_mask = rand_pd_mask.view(dim_B, -1, *rand_pd_mask.shape[1:])
        dim_Vd = rand_feat.shape[1] // dim_B
        dim_Vr = ref_feat.shape[1] // dim_B

        que_GT_alloR = que_GT_alloR.view(dim_B, -1, *que_GT_alloR.shape[1:]) # ==> BxVqx3x3
        ref_GT_alloR = ref_GT_alloR.view(dim_B, -1, *ref_GT_alloR.shape[1:])
        qnnb_GT_alloR = qnnb_GT_alloR.view(dim_B, -1, *qnnb_GT_alloR.shape[1:])
        rand_GT_alloR = rand_GT_alloR.view(dim_B, -1, *rand_GT_alloR.shape[1:])
        noise_alloR = torch.cat([qnnb_GT_alloR, rand_GT_alloR, ref_GT_alloR], dim=1) # Bx(VqVn+Vd+Vr)x3x3
        noise_alloR_dist = torch.einsum('bqij,bmjk->bqmik', que_GT_alloR, noise_alloR.transpose(-2, -1))
        noise_alloR_dist = torch.einsum('bqmii->bqm', noise_alloR_dist) / 2.0 - 0.5 # BxVqx(VqVn+Vd+Vr)

        x_feat = torch.cat([
            que_feat, qnnb_feat, rand_feat, ref_feat
        ], dim=1).flatten(0, 1) # Bx(Vq+VqVn+Vd+Vr)xCx32x32        
        x_mask = torch.cat([
            que_pd_mask, qnnb_pd_mask, rand_pd_mask, ref_pd_mask,
        ], dim=1).flatten(0, 1)
        
        x_feat = self.generate_rotation_aware_embedding(x_feat, x_mask) # B(q+VqVn+Vd+Vr)xCxSxS -> B(q+VqVn+Vd+Vr)xC
        x_feat = x_feat.view(dim_B, -1, x_feat.shape[-1])       # ==> Bx(Vq+VqVn+Vd+Vr)xC
        x_que = x_feat[:, :dim_Vq, :].contiguous()   # ==> BxVqxC
        x_noise = x_feat[:, dim_Vq:, :].contiguous() # ==> Bx(VqVn+Vd+Vr)xC

        pos_alloR_inds = torch.topk(noise_alloR_dist, dim=-1, k=1, largest=True).indices # BxVqx1
        x_pos = torch.gather(x_noise, dim=1, index=pos_alloR_inds.repeat(1, 1, x_noise.shape[-1])) # BxVqxC

        pos_logit = torch.einsum('bqc,bqc->bq', x_que, x_pos) / self.temperature     # ==> BxVq
        Remb_logit = torch.einsum('bqc,bmc->bqm', x_que, x_noise) / self.temperature # ==> BxVqx(VqVn+Vd+Vr)
        Remb_loss = torch.logsumexp(Remb_logit.float(), dim=-1) - pos_logit.float()  # BxVqx(VqVn+Vd+Vr) -> BxVq
        data["Remb_loss"] = Remb_loss.mean()
        data["Remb_logit"] = Remb_logit.flatten(0, 1)       # BVqx(VqVn+Vd+Vr)
        data['delta_Rdeg'] = noise_alloR_dist.flatten(0, 1) # BVqx(VqVn+Vd+Vr)


