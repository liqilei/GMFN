import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift, ResidualDenseBlock_8C


class GFMRDB(nn.Module):
    def __init__(self, num_features, num_blocks, num_refine_feats, num_reroute_feats, act_type, norm_type=None):
        super(GFMRDB, self).__init__()

        self.num_refine_feats = num_refine_feats
        self.num_reroute_feats = num_reroute_feats

        self.RDBs_list = nn.ModuleList([ResidualDenseBlock_8C(
            num_features, kernel_size=3, gc=num_features, act_type=act_type
            ) for _ in range(num_blocks)])

        self.GFMs_list = nn.ModuleList([
                ConvBlock(
                    in_channels=num_reroute_feats*num_features, out_channels=num_features, kernel_size=1,
                    norm_type=norm_type, act_type=act_type
                ),
                ConvBlock(
                    in_channels=2*num_features, out_channels=num_features, kernel_size=1,
                    norm_type=norm_type, act_type=act_type)
            ])


    def forward(self, input_feat, last_feats_list):

        cur_feats_list = []

        if len(last_feats_list) == 0:
            for b in self.RDBs_list:
                input_feat = b(input_feat)
                cur_feats_list.append(input_feat)
        else:
            for idx, b in enumerate(self.RDBs_list):

                # refining the lowest-level features
                if idx < self.num_refine_feats:
                    select_feat = self.GFMs_list[0](torch.cat(last_feats_list, 1))
                    input_feat = self.GFMs_list[1](torch.cat((select_feat, input_feat), 1))

                input_feat = b(input_feat)
                cur_feats_list.append(input_feat)

        # rerouting the highest-level features
        return cur_feats_list[-self.num_reroute_feats:]


class GMFN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_blocks,
                 num_reroute_feats, num_refine_feats, upscale_factor, act_type = 'prelu', norm_type = None):
        super(GMFN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        else:
            raise ValueError("upscale_factor must be 2,3,4.")

        self.num_features = num_features
        self.num_steps = num_steps
        self.upscale_factor = upscale_factor

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # initial low-level feature extraction block
        self.conv_in = ConvBlock(in_channels, 4*num_features, kernel_size=3, act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4*num_features, num_features, kernel_size=1, act_type=act_type, norm_type=norm_type)

        # multiple residual dense blocks (RDBs) and multiple gated feedback modules (GFMs)
        self.block = GFMRDB(num_features, num_blocks, num_refine_feats, num_reroute_feats,
                            act_type=act_type, norm_type=norm_type)

        # reconstruction block
        self.upsample = nn.functional.interpolate
        self.out = DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                               act_type='prelu', norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type)

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)


    def forward(self, lr_img):
        lr_img = self.sub_mean(lr_img)
        up_lr_img = self.upsample(lr_img, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        init_feat = self.feat_in(self.conv_in(lr_img))

        sr_imgs = []
        last_feats_list = []

        for _ in range(self.num_steps):
            last_feats_list = self.block(init_feat, last_feats_list)
            out = torch.add(up_lr_img, self.conv_out(self.out(last_feats_list[-1])))
            out = self.add_mean(out)
            sr_imgs.append(out)

        return sr_imgs


    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name.replace('module.','')
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('out') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('out') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))