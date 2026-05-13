import torch
import torch.nn as nn

from model.FRWKV_CrossBranchPhaseGateAdaptive import (
    LinearFreTransformerCrossBranchPhaseGateAdaptive,
)


class LinearFreTransformerCrossBranchPhaseGateAdaptiveLinearProj(
    LinearFreTransformerCrossBranchPhaseGateAdaptive
):
    def __init__(self, configs):
        super().__init__(configs)
        self.value_proj = nn.Linear(1, self.embed_size)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.value_proj.bias)

    def tokenEmb(self, x, embeddings=None):
        x = x.transpose(-1, -2).unsqueeze(-1)  # (B, N, T, 1)
        if self.embed_size <= 1:
            return x
        return self.value_proj(x)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LinearFreTransformerCrossBranchPhaseGateAdaptiveLinearProj(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
