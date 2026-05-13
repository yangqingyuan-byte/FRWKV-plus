import torch
import torch.nn as nn

from model.FRWKV_CrossBranchPhaseGateAdaptive import (
    LinearFreTransformerCrossBranchPhaseGateAdaptive,
)


class LinearFreTransformerCrossBranchPhaseGateAdaptivePatchFoldEmb(
    LinearFreTransformerCrossBranchPhaseGateAdaptive
):
    def __init__(self, configs):
        super().__init__(configs)
        self.patch_embed_proj = nn.Linear(self.patch_len, self.embed_size)
        nn.init.normal_(self.patch_embed_proj.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.patch_embed_proj.bias)

    def tokenEmb(self, x, embeddings=None):
        # x: (B, T, N)
        x = x.transpose(-1, -2)  # (B, N, T)
        b, n, t = x.shape
        if self.embed_size <= 1:
            return x.unsqueeze(-1)

        pad_len = max(self.patch_len - self.stride, 0)
        if pad_len > 0:
            tail = x[:, :, -1:].expand(-1, -1, pad_len)
            x_pad = torch.cat([x, tail], dim=-1)
        else:
            x_pad = x

        patches = x_pad.unfold(-1, self.patch_len, self.stride)  # (B, N, L, P)
        patch_emb = self.patch_embed_proj(patches)  # (B, N, L, D)

        token_count = x_pad.size(-1)
        folded = x.new_zeros(b, n, token_count, self.embed_size)
        counts = x.new_zeros(b, n, token_count, 1)
        num_patches = patch_emb.size(2)

        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            folded[:, :, start:end, :] += patch_emb[:, :, i : i + 1, :]
            counts[:, :, start:end, :] += 1.0

        folded = folded / counts.clamp_min(1.0)
        return folded[:, :, :t, :]


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LinearFreTransformerCrossBranchPhaseGateAdaptivePatchFoldEmb(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
