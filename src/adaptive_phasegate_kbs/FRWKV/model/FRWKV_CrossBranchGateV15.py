import torch
import torch.nn as nn

from model.FRWKV import LinearFreTransformerOptimized


class LinearFreTransformerCrossBranchGateV15(LinearFreTransformerOptimized):
    def __init__(self, configs):
        super().__init__(configs)
        gate_hidden = max(32, self.embed_size * 8)

        # Baseline v1 gates
        self.real_from_imag_gate = nn.Sequential(
            nn.Linear(self.embed_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.embed_size),
        )
        self.imag_from_real_gate = nn.Sequential(
            nn.Linear(self.embed_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.embed_size),
        )

        # Tiny v2-style correction branch, initialized to do almost nothing.
        self.real_ctx_norm = nn.LayerNorm(self.embed_size)
        self.imag_ctx_norm = nn.LayerNorm(self.embed_size)
        self.real_from_imag_corr = nn.Sequential(
            nn.Linear(self.embed_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.embed_size),
        )
        self.imag_from_real_corr = nn.Sequential(
            nn.Linear(self.embed_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.embed_size),
        )
        nn.init.zeros_(self.real_from_imag_corr[-1].weight)
        nn.init.zeros_(self.real_from_imag_corr[-1].bias)
        nn.init.zeros_(self.imag_from_real_corr[-1].weight)
        nn.init.zeros_(self.imag_from_real_corr[-1].bias)

        self.correction_alpha = nn.Parameter(torch.tensor(0.05))

    def Fre_Trans(self, x):
        B, N, T, D = x.shape
        x = x.transpose(-1, -2)

        x_fre = torch.fft.rfft(x, dim=-1, norm="ortho")
        y_real, y_imag = x_fre.real, x_fre.imag

        y_real_input = y_real.flatten(-2)
        y_imag_input = y_imag.flatten(-2)
        y_real = (self.real_freq_branch(y_real_input) + y_real_input).reshape(
            B, N, D, self.valid_fre_points
        )
        y_imag = (self.imag_freq_branch(y_imag_input) + y_imag_input).reshape(
            B, N, D, self.valid_fre_points
        )

        real_ctx = y_real.mean(dim=-1)
        imag_ctx = y_imag.mean(dim=-1)

        gate_real = torch.sigmoid(self.real_from_imag_gate(imag_ctx)).unsqueeze(-1)
        gate_imag = torch.sigmoid(self.imag_from_real_gate(real_ctx)).unsqueeze(-1)

        corr_real = torch.tanh(
            self.real_from_imag_corr(self.imag_ctx_norm(imag_ctx))
        ).unsqueeze(-1)
        corr_imag = torch.tanh(
            self.imag_from_real_corr(self.real_ctx_norm(real_ctx))
        ).unsqueeze(-1)

        alpha = torch.clamp(self.correction_alpha, min=0.0, max=0.15)
        y_real = y_real * (1.0 + gate_real) * (1.0 + alpha * corr_real)
        y_imag = y_imag * (1.0 + gate_imag) * (1.0 + alpha * corr_imag)

        y = torch.complex(y_real, y_imag)
        x = torch.fft.irfft(y, n=T, dim=-1, norm="ortho")
        x = x.transpose(-1, -2)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LinearFreTransformerCrossBranchGateV15(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
