import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convolution import ConvChunk

from ..utils.data_utils import concat_tensors


class HConvEncoder(nn.Module):
    def __init__(self, resolutions, sizes, num_features=None,
                 heads=None, lite_blocks=True, lite_layers=True,
                 spectral_norm=False, rescale_first=False,
                 defuse_steps=1, bottleneck_ratio=0.25):
        super().__init__()
        assert len(resolutions) == len(sizes)
        dim = len(resolutions[0])
        resolutions = [resolutions[0]] + resolutions

        # Defusing Chunks
        defuse_chunks = []
        for i in range(defuse_steps):
            down_rate = []
            for res_in, res_out in zip(resolutions[i], resolutions[i+1]):
                assert res_in % res_out == 0
                down_rate.append(res_in // res_out)

            defuse_chunks.append(
                ConvChunk(*sizes[i], feature_width=num_features, dim=dim, heads=None,
                    lite_blocks=lite_blocks, lite_layers=lite_layers,
                    use_3x3=(min(resolutions[i+1]) > 1),
                    spectral_norm=spectral_norm,
                    final_act=("gelu" if i < len(sizes) - 1 else None),
                    down_rate=(down_rate if math.prod(down_rate) > 1 else None),
                    rescale_first=rescale_first
                )
            )

        self.defuse_chunks = nn.ModuleList(defuse_chunks)

        # Encoding Chunks
        groups = []
        chunks = []
        final_chunks = []
        out_size = sizes[defuse_steps][0]
        for i in range(defuse_steps, len(sizes)):
            down_rate = []
            for res_in, res_out in zip(resolutions[i], resolutions[i+1]):
                assert res_in % res_out == 0
                down_rate.append(res_in // res_out)

            if math.prod(down_rate) > 1 and len(chunks) > 0:
                group = nn.Sequential(*chunks)
                groups.append(group)
                chunks = []
                final_chunks.append(
                    ConvChunk(out_size, math.ceil(out_size*bottleneck_ratio), out_size,
                        feature_width=None, dim=dim, heads=heads,
                        lite_blocks=True, lite_layers=True,
                        use_3x3=False, spectral_norm=spectral_norm
                    )
                )

            chunks.append(
                ConvChunk(*sizes[i], feature_width=None, dim=dim, heads=None,
                    lite_blocks=lite_blocks, lite_layers=lite_layers,
                    use_3x3=(min(resolutions[i+1]) > 1),
                    spectral_norm=spectral_norm,
                    final_act=("gelu" if i < len(sizes) - 1 else None),
                    down_rate=(down_rate if math.prod(down_rate) > 1 else None),
                    rescale_first=rescale_first
                )
            )
            out_size = sizes[i][-1]
        if len(chunks) > 0:
            group = nn.Sequential(*chunks)
            groups.append(group)
            final_chunks.append(
                ConvChunk(out_size, math.ceil(out_size*bottleneck_ratio), out_size,
                    feature_width=None, dim=dim, heads=heads,
                    lite_blocks=True, lite_layers=True,
                    use_3x3=(min(resolutions[i+1]) > 1),
                    spectral_norm=spectral_norm
                )
            )

        self.groups = nn.ModuleList(groups)
        self.final_chunks = nn.ModuleList(final_chunks)

    def forward(self, x, *f):
        f = concat_tensors(f)

        y = []
        for i, defuse_chunk in enumerate(self.defuse_chunks):
            x, f = defuse_chunk(x, f)
            y.append(x)

        z = []
        for group, final_chunk in zip(self.groups, self.final_chunks):
            x = group(x)
            h = final_chunk(x)
            z.append(h)
        return z, y[:-1]


class HConvDecoder(nn.Module):
    def __init__(self, resolutions, sizes, num_features,
                 heads=None, lite_blocks=False, lite_layers=True,
                 spectral_norm=False, rescale_first=False,
                 infuse_steps=1):
        super().__init__()
        assert len(resolutions) == len(sizes)
        dim = len(resolutions[0])
        resolutions = [resolutions[0]] + resolutions

        # Decoding Chunks
        groups = []
        chunks = []
        latent_biases = []
        in_size = sizes[0][0]
        feature_width = num_features + in_size
        for i in range(len(sizes)-infuse_steps):
            up_rate = []
            for res_in, res_out in zip(resolutions[i], resolutions[i+1]):
                assert res_out % res_in == 0
                up_rate.append(res_out // res_in)

            chunks.append(
                ConvChunk(*sizes[i], feature_width=feature_width, dim=dim,
                    heads=(heads if i == len(sizes) - 1 else None),
                    lite_blocks=lite_blocks, lite_layers=lite_layers,
                    use_3x3=(min(resolutions[i+1]) > 1),
                    spectral_norm=(spectral_norm and i < len(sizes) - 1),
                    final_act=("gelu" if i < len(sizes) - 1 else None),
                    up_rate=(up_rate if math.prod(up_rate) > 1 else None),
                    rescale_first=rescale_first
                )
            )

            if math.prod(up_rate) > 1:
                chunks = nn.ModuleList(chunks)
                groups.append(chunks)
                chunks = []
                latent_biases.append(
                    nn.Parameter(torch.zeros(in_size, *resolutions[i]))
                )
                in_size = sizes[i][-1]
                feature_width = num_features + in_size
        if len(chunks) > 0:
            chunks = nn.ModuleList(chunks)
            groups.append(chunks)
            latent_biases.append(
                nn.Parameter(torch.zeros(in_size, *resolutions[i]))
            )

        self.groups = nn.ModuleList(groups)
        self.latent_biases = nn.ParameterList(latent_biases)

        # Constructing Chunks
        final_chunks = []
        for i in range(len(sizes)-infuse_steps, len(sizes)):
            up_rate = []
            for res_in, res_out in zip(resolutions[i], resolutions[i+1]):
                assert res_out % res_in == 0
                up_rate.append(res_out // res_in)

            final_chunks.append(
                ConvChunk(*sizes[i], feature_width=num_features, dim=dim,
                    heads=(heads if i == len(sizes) - 1 else None),
                    lite_blocks=lite_blocks, lite_layers=lite_layers,
                    use_3x3=(min(resolutions[i+1]) > 1),
                    spectral_norm=(spectral_norm and i < len(sizes) - 1),
                    final_act=("gelu" if i < len(sizes) - 1 else None),
                    up_rate=(up_rate if math.prod(up_rate) > 1 else None),
                    rescale_first=rescale_first
                )
            )

        self.final_chunks = nn.ModuleList(final_chunks)

    def forward(self, z, *f):
        f = concat_tensors(f)
        while f.ndim < z[0].ndim:
            f = f[..., None]

        x = 0
        for h, chunks, latent_bias in zip(reversed(z), self.groups, self.latent_biases):
            x = x + latent_bias[None, ...].expand(h.shape[0], *latent_bias.size())
            features = torch.cat([f.expand(*f.shape[:2], *h.shape[2:]), h], dim=1)
            for chunk in chunks:
                x, features = chunk(x, features)

        y = []
        for i, final_chunk in enumerate(self.final_chunks):
            features = f.expand(*f.shape[:2], *x.shape[2:])
            x, features = final_chunk(x, features)
            y.append(x)
        return x, y[:-1]
