import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.PReLU(ch_out)
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
            
        return fx + x

    
class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.PReLU(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.PReLU(ch_out),
        )

    def forward(self, x):
        return self.f(x)

class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.PReLU(ch_out)
        )

    def forward(self, x):
        return self.f(x)

class GridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs = [32, 64, 96], nrow=3, ncol=6):
        super(GridNet, self).__init__()

        self.n_row = nrow
        self.n_col = ncol
        self.n_chs = grid_chs
        assert len(grid_chs) == self.n_row, 'should give num channels for each row (scale stream)'

        self.lateral_init = LateralBlock(in_chs, self.n_chs[0])
        
        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col-1):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col/2)):
                setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col/2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))
                
        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)
   
    def forward(self, x):
        forward_func = getattr(self, f'forward_{self.n_row}{self.n_col}')
        return forward_func(x) 
                                    
    def forward_36(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)

        return self.lateral_final(state_05)

    def forward_34(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)
            
        state_22 = self.lateral_2_1(state_21)
        state_12 = self.up_1_0(state_22) + self.lateral_1_1(state_11)
        state_02 = self.up_0_0(state_12) + self.lateral_0_1(state_01)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_1(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_1(state_13) + self.lateral_0_2(state_02)

        return self.lateral_final(state_03)

    def forward_32(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)
            
        state_21 = self.lateral_2_0(state_20)
        state_11 = self.up_1_0(state_21) + self.lateral_1_0(state_10)
        state_01 = self.up_0_0(state_11) + self.lateral_0_0(state_00)

        return self.lateral_final(state_01)

    def forward_30(self, x):
        state_00 = self.lateral_init(x)
    
        return self.lateral_final(state_00)

    def forward_24(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
            
        state_12 = self.lateral_1_1(state_11)
        state_02 = self.up_0_0(state_12) + self.lateral_0_1(state_01)

        state_13 = self.lateral_1_2(state_12)
        state_03 = self.up_0_1(state_13) + self.lateral_0_2(state_02)

        return self.lateral_final(state_03)

    def forward_22(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
            
        state_11 = self.lateral_1_0(state_10)
        state_01 = self.up_0_0(state_11) + self.lateral_0_0(state_00)

        return self.lateral_final(state_01)

class down(nn.Module):
    def __init__(self, num_in_ch, num_out_ch):
        super(down, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, stride=2, padding=1),
            nn.PReLU(num_out_ch),
            nn.Conv2d(num_out_ch, num_out_ch, 3, stride=1, padding=1),
            nn.PReLU(num_out_ch),
            nn.Conv2d(num_out_ch, num_out_ch, 3, stride=1, padding=1),
            nn.PReLU(num_out_ch)
        )
        
    def forward(self, x):
        x = self.body(x)
        
        return x
    
class up(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, split=0.5):
        super(up, self).__init__()
        self.split = split
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(num_in_ch,  num_out_ch, 3, stride=1, padding=1), 
            nn.PReLU(num_out_ch)
        )
        self.decouple = nn.Sequential(
            nn.Conv2d(2*num_out_ch, num_out_ch, 3, stride=1, padding=1), 
            nn.PReLU(num_out_ch)
        )
        self.merge = nn.Sequential(
            nn.Conv2d(num_out_ch, num_out_ch, 3, stride=1, padding=1), 
            nn.PReLU(num_out_ch)
            )
           
    def forward(self, x, skip_ch, t):
        x = self.head(x)
        x = self.decouple(torch.cat((x, skip_ch), 1))
        b, c, h, w = x.shape
        p = int(c*self.split)
        x = torch.cat((x[:,:p]*t, x[:,p:]), 1)
        x = self.merge(x)

        return x

@ARCH_REGISTRY.register()
class SDLNet(nn.Module):
    """
    SDLNet architecture
    """

    def __init__(self, num_in_ch, num_out_ch, split=0.5, num_feat=32, nrow=3, ncol=6):
        super(SDLNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(num_in_ch, 32, 3, stride=1, padding=1),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), 
            nn.PReLU(32)
        )
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        
        self.up1   = up(512, 512, split)
        self.up2   = up(512, 256, split)
        self.up3   = up(256, 128, split)
        self.up4   = up(128, 64, split)
        self.up5   = up(64, 32, split)
        self.tail = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1), # 32
            nn.PReLU(32)
        )

        self.gridnet = GridNet(32, num_out_ch, nrow=nrow, ncol=ncol)

    def preforward(self, x, t):
        t = t.view(-1, 1, 1, 1)

        s1 = self.head(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x  = self.down5(s5)
        
        x  = self.up1(x, s5, t)
        x  = self.up2(x, s4, t)
        x  = self.up3(x, s3, t)
        x  = self.up4(x, s2, t)
        x  = self.up5(x, s1, t)
        x  = self.tail(x)

        return x

    def forward(self, x, t):
        x_01 = self.preforward(x, t)
        '''
        idx_rvs = torch.LongTensor(range(-3, 3))
        x_rvs = x[:,idx_rvs]
        x_10 = self.preforward(x_rvs, 1-t)
         
        x = self.gridnet(torch.cat((x_01, x_10), 1))
        '''
        x = self.gridnet(x_01)
        
        return x
