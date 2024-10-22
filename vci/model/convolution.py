import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.data_utils import concat_tensors
from ..utils.model_utils import conv_1x1, conv_3x3


class ConvBlock(nn.Module):
    def __init__(self, in_width, emb_width, out_width, feature_width=None,
                 dim=2, heads=None, residual=True, lite_layers=True,
                 use_3x3=True, spectral_norm=False, final_act=None,
                 up_rate=None, down_rate=None, rescale_first=False):
        super().__init__()
        assert up_rate is None or down_rate is None

        self.up_rate = up_rate
        self.down_rate = down_rate
        self.rescale_first = rescale_first
        self.dim = dim
        self.heads = heads
        self.residual = residual
        if heads is not None:
            out_width = out_width * heads
        self.res_layer = None
        if residual and in_width != out_width:
            self.res_layer = conv_1x1(in_width, out_width, dim)

        emb_model = conv_3x3 if use_3x3 else conv_1x1

        layers = [
            conv_1x1(in_width+feature_width, emb_width, dim)
                if feature_width is not None else
            conv_1x1(in_width, emb_width, dim)
                if not lite_layers else None,

            emb_model(in_width, emb_width, dim)
                if feature_width is None and lite_layers else
            emb_model(emb_width, emb_width, dim),

            emb_model(emb_width, emb_width, dim)
                if not lite_layers else
            emb_model(emb_width, out_width, dim),

            conv_1x1(emb_width, out_width, dim)
                if not lite_layers else None
        ]
        layers = [l for l in layers if l is not None]

        if spectral_norm:
            layers = [nn.utils.parametrizations.spectral_norm(l) for l in layers]

        layers, final_layer = layers[:-1], layers[-1]
        self.layers = nn.ModuleList(layers)
        self.final_layer = final_layer

        self.act = nn.GELU()
        if final_act is None:
            self.final_act = nn.Identity()
        elif final_act == "gelu":
            self.final_act = nn.GELU()
        elif final_act == "relu":
            self.final_act = nn.ReLU()
        elif final_act == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif final_act == "softmax":
            self.final_act = nn.Softmax(dim=-3)
        else:
            raise ValueError("final_act not recognized")

    def rescale(self, x):
        if self.up_rate is not None:
            return F.interpolate(x, scale_factor=self.up_rate)
        elif self.down_rate is not None:
            if self.dim == 1:
                return F.avg_pool1d(x, kernel_size=self.down_rate, stride=self.down_rate)
            elif self.dim == 2:
                return F.avg_pool2d(x, kernel_size=self.down_rate, stride=self.down_rate)
            elif self.dim == 3:
                return F.avg_pool3d(x, kernel_size=self.down_rate, stride=self.down_rate)
        else:
            return x

    def forward(self, x, f=None, return_f=False):
        if self.rescale_first:
            x = self.rescale(x)

        out = x
        if f is not None:
            while f.ndim < x.ndim:
                f = f[..., None].expand(*f.size(), x.shape[f.ndim])
            if self.rescale_first:
                f = self.rescale(f)
            out = torch.cat([out, f], dim=1)

        for layer in self.layers:
            out = self.act(layer(out))
        out = self.final_act(self.final_layer(out))

        if self.residual:
            if self.res_layer:
                x = self.res_layer(x)
            out = out + x

        if not self.rescale_first:
            out = self.rescale(out)
            if f is not None:
                f = self.rescale(f)

        if self.heads is not None:
            out = out.view(out.shape[0], -1, *out.shape[2:], self.heads)

        if return_f:
            return out, f
        return out


class ConvChunk(nn.Module):
    def __init__(self, in_width, emb_width, out_width, feature_width=None,
                 dim=2, heads=None, lite_blocks=False, lite_layers=True,
                 use_3x3=True, spectral_norm=False, final_act=None,
                 up_rate=None, down_rate=None, rescale_first=False):
        super().__init__()
        self.in_width = in_width
        self.out_width = out_width

        if lite_blocks:
            self.merge_block = ConvBlock(in_width, emb_width, out_width,
                feature_width=feature_width, dim=dim, heads=heads, lite_layers=lite_layers,
                use_3x3=use_3x3, spectral_norm=spectral_norm, final_act=final_act,
                up_rate=up_rate, down_rate=down_rate, rescale_first=rescale_first
            )

            self.embed_block = None
        else:
            self.merge_block = ConvBlock(in_width, emb_width, in_width,
                feature_width=feature_width, dim=dim, lite_layers=lite_layers,
                use_3x3=use_3x3, spectral_norm=spectral_norm, final_act="gelu",
                up_rate=up_rate, down_rate=down_rate, rescale_first=rescale_first
            )

            self.embed_block = ConvBlock(in_width, emb_width, out_width,
                dim=dim, heads=heads, lite_layers=lite_layers,
                use_3x3=use_3x3, spectral_norm=spectral_norm, final_act=final_act
            )

    def forward(self, x, f=None, return_f=None):
        if return_f is None:
            return_f = f is not None

        x, f = self.merge_block(x, f, return_f=True)
        if self.embed_block is not None:
            x = self.embed_block(x)
        if return_f:
            return x, f
        return x


class ConvModel(nn.Module):
    def __init__(self, resolutions, sizes, num_features=None,
                 dim=2, heads=None, lite_blocks=False, lite_layers=True,
                 spectral_norm=False, rescale_first=False):
        super().__init__()
        assert len(resolutions) == len(sizes)
        dim = len(resolutions[0])
        resolutions = [resolutions[0]] + resolutions

        chunks = []
        for i in range(len(sizes)):
            up_rate = []
            down_rate = []
            for res_in, res_out in zip(resolutions[i], resolutions[i+1]):
                assert (res_in % res_out == 0 or res_out % res_in == 0)
                up_rate.append(res_out // res_in)
                down_rate.append(res_in // res_out)

            chunks.append(
                ConvChunk(*sizes[i], feature_width=num_features, dim=dim,
                    heads=(heads if i == len(sizes) - 1 else None),
                    lite_blocks=lite_blocks, lite_layers=lite_layers,
                    use_3x3=(min(resolutions[i+1]) > 1),
                    spectral_norm=(spectral_norm and i < len(sizes) - 1),
                    final_act=("gelu" if i < len(sizes) - 1 else None),
                    up_rate=(up_rate if math.prod(up_rate) > 1 else None),
                    down_rate=(down_rate if math.prod(down_rate) > 1 else None),
                    rescale_first=rescale_first
                )
            )

        self.chunks = nn.ModuleList(chunks)

    def forward(self, x, *f):
        f = concat_tensors(f) if len(f) > 0 else None

        for chunk in self.chunks:
            x, f = chunk(x, f, return_f=True)

        return x
