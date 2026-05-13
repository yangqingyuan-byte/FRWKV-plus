import torch
import torch.nn as nn

from model.FRWKV_CrossBranchPhaseGateAdaptive import (
    LinearFreTransformerCrossBranchPhaseGateAdaptive,
)


class LinearFreTransformerCrossBranchPhaseGateAdaptiveChannelEmb(
    LinearFreTransformerCrossBranchPhaseGateAdaptive
):
    def __init__(self, configs):
        super().__init__(configs)
        self.channel_embeddings = nn.Parameter(
            torch.randn(self.enc_in, self.embed_size) * 0.1
        )

    def tokenEmb(self, x, embeddings=None):
        x = x.transpose(-1, -2).unsqueeze(-1)  # (B, N, T, 1)
        if self.embed_size <= 1:
            return x
        channel_embeddings = self.channel_embeddings.unsqueeze(0).unsqueeze(2)  # (1, N, 1, D)
        return x * channel_embeddings


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LinearFreTransformerCrossBranchPhaseGateAdaptiveChannelEmb(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
