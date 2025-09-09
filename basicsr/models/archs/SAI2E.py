import torch
import torch.nn as nn
import torch.nn.functional as F

class SAI2E(nn.Module):
    def __init__(self,
                 in_channels=3,
                 train_patch=128,
                 eps=0,
        ):
        super(SAI2E, self).__init__()

        self.in_channels = in_channels
        self.eps = eps
        if not isinstance(train_patch, list):
            self.train_patch = [train_patch, train_patch]
        else:
            self.train_patch = train_patch
        self.offset_predict = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
                                nn.GELU(),
                                nn.Conv2d(in_channels, 4, 1, padding=0, bias=True),
                                )
        self.modulation_predict = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
                                nn.GELU(),
                                nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=True),
                                )

    def get_center_grid(self, x):
        B,_,H,W = x.shape
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='xy'), dim=-1).type(x.dtype).to(x.device)
        norm_coords = coords / torch.tensor([W, H], dtype=x.dtype).to(x.device) * 2 - 1
        return norm_coords

    def forward(self, x, train_mode=False):
        Batch,_,H,W = x.shape

        # 生成积分图（各通道独立）
        integrated_x = torch.cumsum(x, dim=-1)
        integrated_x = torch.cumsum(integrated_x, dim=-2)

        # 获得中心坐标
        center_grid = self.get_center_grid(x).unsqueeze(0)
        normalizer = torch.tensor(
            [self.train_patch[0]/W, self.train_patch[1]/H], dtype=x.dtype, device=x.device).view(1, 1, 1, 2)

        subnet_output = self.offset_predict(x).permute(0, 2, 3, 1)

        off_w, off_h = torch.split(subnet_output, 2, dim=3)
        off_w = off_w - off_w.mean(dim=-1, keepdim=True)
        off_h = off_h - off_h.mean(dim=-1, keepdim=True)
        minimum_patch = 2
        off_w_min = torch.minimum(off_w.min(dim=-1, keepdim=True)[0],
                                    torch.zeros_like(off_w.min(dim=-1, keepdim=True)[0]) - minimum_patch/self.train_patch[0])
        off_w_max = torch.maximum(off_w.max(dim=-1, keepdim=True)[0],
                                    torch.zeros_like(off_w.max(dim=-1, keepdim=True)[0]) + minimum_patch/self.train_patch[0])
        off_h_min = torch.minimum(off_h.min(dim=-1, keepdim=True)[0],
                                    torch.zeros_like(off_h.min(dim=-1, keepdim=True)[0]) - minimum_patch/self.train_patch[1])
        off_h_max = torch.maximum(off_h.max(dim=-1, keepdim=True)[0],
                                    torch.zeros_like(off_h.max(dim=-1, keepdim=True)[0]) + minimum_patch/self.train_patch[1])

        area = (off_h_max - off_h_min) * (off_w_max - off_w_min) * self.train_patch[0] * self.train_patch[1] / 4
        area = area.view(Batch, 1, H, W).clip(1, H * W)
        scale = self.modulation_predict(x)
        if self.eps != 0:
            mask = (scale.abs() < self.eps)
            safe_sign = torch.where(scale >= 0, 1.0, -1.0)
            scale = torch.where(mask, safe_sign * self.eps, scale)
        area = area*scale

        off_tl = (torch.cat([off_w_min, off_h_min], dim=-1)*normalizer + center_grid).clip(-1, 1)
        off_tr = (torch.cat([off_w_max, off_h_min], dim=-1)*normalizer + center_grid).clip(-1, 1)
        off_bl = (torch.cat([off_w_min, off_h_max], dim=-1)*normalizer + center_grid).clip(-1, 1)
        off_br = (torch.cat([off_w_max, off_h_max], dim=-1)*normalizer + center_grid).clip(-1, 1)

        # 采样
        A = F.grid_sample(integrated_x, off_tl, align_corners=True, padding_mode='border', mode='bilinear')
        B = F.grid_sample(integrated_x, off_tr, align_corners=True, padding_mode='border', mode='bilinear')
        C = F.grid_sample(integrated_x, off_bl, align_corners=True, padding_mode='border', mode='bilinear')
        D = F.grid_sample(integrated_x, off_br, align_corners=True, padding_mode='border', mode='bilinear')

        res = (A+D-B-C)/area
        return res