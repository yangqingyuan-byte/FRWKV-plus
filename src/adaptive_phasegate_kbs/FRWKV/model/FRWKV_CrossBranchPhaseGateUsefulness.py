import math

import torch
import torch.nn as nn

from model.FRWKV import LinearFreTransformerOptimized


class PhaseRouterContext(nn.Module):
    def __init__(self, embed_size: int, num_routers: int):
        super().__init__()
        self.embed_size = embed_size
        self.num_routers = num_routers

        self.input_norm = nn.LayerNorm(embed_size)
        self.q_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.k_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.v_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.out_proj = nn.Linear(embed_size, embed_size, bias=False)

        self.routers = nn.Parameter(torch.randn(num_routers, embed_size) * 0.02)

    def forward(self, phase_tokens):
        # phase_tokens: (B, N, P, D)
        b, n, p, d = phase_tokens.shape
        x = self.input_norm(phase_tokens.reshape(b * n, p, d))

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        routers = self.routers.unsqueeze(0).expand(b * n, -1, -1)
        scores_router = torch.matmul(routers, k.transpose(1, 2)) / math.sqrt(d)
        attn_router = torch.softmax(scores_router, dim=-1)
        router_buffer = torch.matmul(attn_router, v)

        scores_token = torch.matmul(q, router_buffer.transpose(1, 2)) / math.sqrt(d)
        attn_token = torch.softmax(scores_token, dim=-1)
        token_ctx = torch.matmul(attn_token, router_buffer)

        ctx = token_ctx.mean(dim=1).reshape(b, n, d)
        return self.out_proj(ctx)


class LinearFreTransformerCrossBranchPhaseGateUsefulness(LinearFreTransformerOptimized):
    def __init__(self, configs):
        super().__init__(configs)
        self.phase_period_len = getattr(configs, "phase_period_len", 24)
        self.phase_num_routers = getattr(configs, "phase_num_routers", 4)

        gate_hidden = max(32, self.embed_size * 8)
        phase_hidden = max(32, self.embed_size * 8)
        trust_hidden = max(32, self.embed_size * 12)
        usefulness_hidden = max(32, self.embed_size * 12)

        # Baseline cross-branch interaction
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

        # Phase context extractor
        self.phase_router = PhaseRouterContext(self.embed_size, self.phase_num_routers)
        self.phase_real_gate = nn.Sequential(
            nn.Linear(self.embed_size * 2, phase_hidden),
            nn.GELU(),
            nn.Linear(phase_hidden, self.embed_size),
        )
        self.phase_imag_gate = nn.Sequential(
            nn.Linear(self.embed_size * 2, phase_hidden),
            nn.GELU(),
            nn.Linear(phase_hidden, self.embed_size),
        )

        # Explicit usefulness predictor: learn whether phase info should be trusted.
        self.phase_usefulness = nn.Sequential(
            nn.Linear(self.embed_size * 5, usefulness_hidden),
            nn.GELU(),
            nn.Linear(usefulness_hidden, 1),
        )

        # Branch-specific trust refinements
        self.phase_real_trust = nn.Sequential(
            nn.Linear(self.embed_size * 3, trust_hidden),
            nn.GELU(),
            nn.Linear(trust_hidden, self.embed_size),
        )
        self.phase_imag_trust = nn.Sequential(
            nn.Linear(self.embed_size * 3, trust_hidden),
            nn.GELU(),
            nn.Linear(trust_hidden, self.embed_size),
        )

        # Start close to baseline v1.
        nn.init.zeros_(self.phase_real_gate[-1].weight)
        nn.init.zeros_(self.phase_real_gate[-1].bias)
        nn.init.zeros_(self.phase_imag_gate[-1].weight)
        nn.init.zeros_(self.phase_imag_gate[-1].bias)

        nn.init.zeros_(self.phase_real_trust[-1].weight)
        nn.init.constant_(self.phase_real_trust[-1].bias, -2.0)
        nn.init.zeros_(self.phase_imag_trust[-1].weight)
        nn.init.constant_(self.phase_imag_trust[-1].bias, -2.0)

        nn.init.zeros_(self.phase_usefulness[-1].weight)
        nn.init.constant_(self.phase_usefulness[-1].bias, -2.0)

        self.phase_alpha = nn.Parameter(torch.tensor(0.10))

    def _build_phase_context(self, x):
        # x: (B, N, T, D)
        b, n, t, d = x.shape
        p = self.phase_period_len
        if p <= 0:
            raise ValueError("phase_period_len must be positive")

        pad_len = (p - (t % p)) % p
        if pad_len > 0:
            x = torch.cat([x, x[:, :, :pad_len, :]], dim=2)
        num_periods = x.size(2) // p

        phase_tokens = x.reshape(b, n, num_periods, p, d).mean(dim=2)
        return self.phase_router(phase_tokens)

    def Fre_Trans(self, x):
        b, n, t, d = x.shape
        phase_ctx = self._build_phase_context(x)

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

        gate_real_base = torch.sigmoid(self.real_from_imag_gate(imag_ctx))
        gate_imag_base = torch.sigmoid(self.imag_from_real_gate(real_ctx))

        phase_real_delta = torch.tanh(
            self.phase_real_gate(torch.cat([imag_ctx, phase_ctx], dim=-1))
        )
        phase_imag_delta = torch.tanh(
            self.phase_imag_gate(torch.cat([real_ctx, phase_ctx], dim=-1))
        )

        branch_gap = torch.abs(real_ctx - imag_ctx)
        phase_center = 0.5 * (real_ctx + imag_ctx)
        phase_gap = torch.abs(phase_ctx - phase_center)
        usefulness_in = torch.cat([real_ctx, imag_ctx, phase_ctx, branch_gap, phase_gap], dim=-1)
        usefulness = torch.sigmoid(self.phase_usefulness(usefulness_in))  # (B, N, 1)

        trust_real = usefulness * torch.sigmoid(
            self.phase_real_trust(torch.cat([real_ctx, imag_ctx, phase_ctx], dim=-1))
        )
        trust_imag = usefulness * torch.sigmoid(
            self.phase_imag_trust(torch.cat([real_ctx, imag_ctx, phase_ctx], dim=-1))
        )

        alpha = torch.clamp(self.phase_alpha, min=0.0, max=0.20)
        gate_real = (1.0 + gate_real_base + alpha * trust_real * phase_real_delta).unsqueeze(-1)
        gate_imag = (1.0 + gate_imag_base + alpha * trust_imag * phase_imag_delta).unsqueeze(-1)

        y_real = y_real * gate_real
        y_imag = y_imag * gate_imag

        y = torch.complex(y_real, y_imag)
        x = torch.fft.irfft(y, n=t, dim=-1, norm="ortho")
        x = x.transpose(-1, -2)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LinearFreTransformerCrossBranchPhaseGateUsefulness(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
