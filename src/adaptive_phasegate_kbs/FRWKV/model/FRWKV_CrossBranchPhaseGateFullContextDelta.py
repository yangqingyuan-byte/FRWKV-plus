import torch
import torch.nn as nn

from model.FRWKV_CrossBranchPhaseGate import LinearFreTransformerCrossBranchPhaseGate


class LinearFreTransformerCrossBranchPhaseGateFullContextDelta(
    LinearFreTransformerCrossBranchPhaseGate
):
    """CrossBranchPhaseGate variant whose periodic correction sees both branches."""

    def __init__(self, configs):
        super().__init__(configs)
        delta_hidden = max(32, self.embed_size * 8)

        self.delta_impos2re = nn.Sequential(
            nn.Linear(self.embed_size * 3, delta_hidden),
            nn.GELU(),
            nn.Linear(delta_hidden, self.embed_size),
        )
        self.delta_repos2im = nn.Sequential(
            nn.Linear(self.embed_size * 3, delta_hidden),
            nn.GELU(),
            nn.Linear(delta_hidden, self.embed_size),
        )

        nn.init.zeros_(self.delta_impos2re[-1].weight)
        nn.init.zeros_(self.delta_impos2re[-1].bias)
        nn.init.zeros_(self.delta_repos2im[-1].weight)
        nn.init.zeros_(self.delta_repos2im[-1].bias)

    def Fre_Trans(self, x):
        b, n, t, d = x.shape
        period_pos_ctx = self._build_periodic_position_context(x)

        x = x.transpose(-1, -2)
        x_fre = torch.fft.rfft(x, dim=-1, norm="ortho")
        y_real, y_imag = x_fre.real, x_fre.imag

        y_real_input = y_real.flatten(-2)
        y_imag_input = y_imag.flatten(-2)
        y_real = (self.real_freq_branch(y_real_input) + y_real_input).reshape(
            b, n, d, self.valid_fre_points
        )
        y_imag = (self.imag_freq_branch(y_imag_input) + y_imag_input).reshape(
            b, n, d, self.valid_fre_points
        )

        real_ctx = y_real.mean(dim=-1)
        imag_ctx = y_imag.mean(dim=-1)
        full_context = torch.cat([real_ctx, imag_ctx, period_pos_ctx], dim=-1)

        base_re = torch.sigmoid(self.base_im2re(imag_ctx))
        base_im = torch.sigmoid(self.base_re2im(real_ctx))

        delta_re = torch.tanh(self.delta_impos2re(full_context))
        delta_im = torch.tanh(self.delta_repos2im(full_context))

        alpha = torch.clamp(self.pos_alpha, min=0.0, max=0.20)
        gate_real = (1.0 + base_re + alpha * delta_re).unsqueeze(-1)
        gate_imag = (1.0 + base_im + alpha * delta_im).unsqueeze(-1)

        y_real = y_real * gate_real
        y_imag = y_imag * gate_imag

        y = torch.complex(y_real, y_imag)
        x = torch.fft.irfft(y, n=t, dim=-1, norm="ortho")
        x = x.transpose(-1, -2)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LinearFreTransformerCrossBranchPhaseGateFullContextDelta(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)

