import torch
from torch import nn
from torch.nn import functional as F 
from torch.autograd import Variable
import math

from PSMNet.Extractor import PSM_Extractor
from PSMNet.SPP import SPP
from PSMNet.EncoderDecoder import StackHourglass


class PSMNet(nn.Module):
    def __init__(self, image_channel=3):
        super().__init__()
        self.fea1 = PSM_Extractor(image_channel, 128)
        self.spp = SPP()
        self.hourglass = StackHourglass()

    def forward(self, imgL, imgR, min_disp, max_disp):
        #extract feature map
        featureL1, featureL2 = self.fea1(imgL) 
        featureR1, featureR2 = self.fea1(imgR) # shape -> 32 * H/4 * W/4

        featureL = self.spp(featureL1, featureL2)
        featureR = self.spp(featureR1, featureR2)
        # construct cost volume
        cost_vol = self.cost_volume(featureL, featureR, min_disp, max_disp) # shape -> B * 64 * (maxdisp-mindisp)/4 * H/4 * W/4

        # cost filtering
        cost_vol1, cost_vol2, cost_vol3 = self.hourglass(cost_vol) # shape -> B * 1 * (maxdisp-mindisp)/4 * H/4 * W/4
        if self.training:
            # shape -> B * 1 * (maxdisp-mindisp) * H * W
            cost_vol1 = F.interpolate(cost_vol1, [max_disp-min_disp, imgL.size()[2], imgL.size()[3]], mode='trilinear')
            cost_vol2 = F.interpolate(cost_vol2, [max_disp-min_disp, imgL.size()[2], imgL.size()[3]], mode='trilinear')

            disp1 = self.softargmax(cost_vol1, min_disp, max_disp) # shape -> B * H * W
            disp2 = self.softargmax(cost_vol2, min_disp, max_disp) # shape -> B * H * W

        cost_vol3 = F.interpolate(cost_vol3, [max_disp-min_disp, imgL.size()[2], imgL.size()[3]], mode='trilinear')   
        # # disparity regression
        disp3 = self.softargmax(cost_vol3, min_disp, max_disp) # shape -> B * H * W
        if self.training:
            return disp1, disp2, disp3
        else: 
            return disp3

    def cost_volume(self, feaL:torch.tensor, feaR:torch.tensor, min_disp, max_disp) -> torch.tensor:
        B, C, H, W = feaL.shape
        device = feaL.device

        # feature map has been downsample, so disparity range should be devided by 2
        max_disp = max_disp // 4
        min_disp = min_disp // 4
        cost = torch.zeros(B, C*2, max_disp-min_disp, H, W).to(device)
        # cost[:, 0:C, :, :, :] = feaL.unsqueeze(2).repeat(1,1,max_disp-min_disp,1,1)

        for i in range(min_disp, max_disp):
            if i < 0:
                cost[:, 0:C, i, :, 0:W+i] = feaL[:, :, :, 0:W+i]
                cost[:, C:, i, :, 0:W+i] = feaR[:, :, :, -i:]
            if i >= 0:
                cost[:, 0:C, i, :, i:] = feaR[:, :, :, i:]
                cost[:, C:, i, :, i:] = feaR[:, :, :, :W-i]
        cost = cost.contiguous()
        return cost
    
    def softargmax(self, cost, min_disp, max_disp):
        cost_softmax = F.softmax(cost, dim = 2)
        vec = torch.arange(min_disp, max_disp).to(cost.device)
        vec = vec.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        vec = vec.expand_as(cost_softmax).type_as(cost_softmax)
        disp = torch.sum(vec*cost_softmax, dim=2)
        disp = disp.squeeze(1)
        return disp
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

