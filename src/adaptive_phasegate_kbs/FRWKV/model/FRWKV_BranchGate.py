import torch
import torch.nn as nn

from model.FRWKV import LinearFreTransformerOptimized


class LinearFreTransformerBranchGate(LinearFreTransformerOptimized):
    def __init__(self, configs):
        super().__init__(configs)
        gate_hidden = max(32, self.embed_size * 8)
        self.branch_gate = nn.Sequential(
            nn.Linear(self.embed_size * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.embed_size),
        )
        self.branch_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        x = self.revin_layer(x, mode="norm")
        x_emb = self.tokenEmb(x, self.embeddings)
        x_fre = self.Fre_Trans(x_emb)

        gate_in = torch.cat([x_emb.mean(dim=2), x_fre.mean(dim=2)], dim=-1)
        gate = torch.sigmoid(self.branch_gate(gate_in)).unsqueeze(2)
        x = x_emb + self.branch_scale * gate * x_fre

        out = self.fc(x.flatten(-2)).transpose(-1, -2)
        out = self.dropout(out)
        out = self.revin_layer(out, mode="denorm")
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LinearFreTransformerBranchGate(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
