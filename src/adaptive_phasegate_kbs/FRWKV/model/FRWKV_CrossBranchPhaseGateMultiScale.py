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


class LinearFreTransformerCrossBranchPhaseGateMultiScale(LinearFreTransformerOptimized):
    def __init__(self, configs):
        super().__init__(configs)
        self.phase_num_routers = getattr(configs, "phase_num_routers", 4)
        self.phase_period_list = self._parse_phase_periods(
            getattr(configs, "phase_period_list", "12,24,48")
        )

        gate_hidden = max(32, self.embed_size * 8)
        phase_hidden = max(32, self.embed_size * 8)
        trust_hidden = max(32, self.embed_size * 12)
        selector_hidden = max(32, self.embed_size * 8)

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

        # Multi-scale phase bank
        self.phase_routers = nn.ModuleList(
            [PhaseRouterContext(self.embed_size, self.phase_num_routers) for _ in self.phase_period_list]
        )

        # Scale selector: one logit per scale.
        self.scale_selector = nn.Sequential(
            nn.Linear(self.embed_size * 3, selector_hidden),
            nn.GELU(),
            nn.Linear(selector_hidden, len(self.phase_period_list)),
        )

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

        # Start close to the baseline v1.
        nn.init.zeros_(self.phase_real_gate[-1].weight)
        nn.init.zeros_(self.phase_real_gate[-1].bias)
        nn.init.zeros_(self.phase_imag_gate[-1].weight)
        nn.init.zeros_(self.phase_imag_gate[-1].bias)
        nn.init.zeros_(self.phase_real_trust[-1].weight)
        nn.init.constant_(self.phase_real_trust[-1].bias, -2.0)
        nn.init.zeros_(self.phase_imag_trust[-1].weight)
        nn.init.constant_(self.phase_imag_trust[-1].bias, -2.0)
        nn.init.zeros_(self.scale_selector[-1].weight)
        nn.init.zeros_(self.scale_selector[-1].bias)

        self.phase_alpha = nn.Parameter(torch.tensor(0.10))

    @staticmethod
    def _parse_phase_periods(value):
        if isinstance(value, (list, tuple)):
            periods = [int(v) for v in value]
        else:
            periods = [int(v.strip()) for v in str(value).split(",") if v.strip()]
        if not periods:
            raise ValueError("phase_period_list must contain at least one period")
        return periods

    def _build_phase_context_for_period(self, x, period_len, router_module):
        # x: (B, N, T, D)
        b, n, t, d = x.shape
        pad_len = (period_len - (t % period_len)) % period_len
        if pad_len > 0:
            x = torch.cat([x, x[:, :, :pad_len, :]], dim=2)
        num_periods = x.size(2) // period_len
        phase_tokens = x.reshape(b, n, num_periods, period_len, d).mean(dim=2)
        return router_module(phase_tokens)

    def _build_multiscale_phase_context(self, x, real_ctx, imag_ctx):
        phase_contexts = []
        for period_len, router_module in zip(self.phase_period_list, self.phase_routers):
            phase_contexts.append(self._build_phase_context_for_period(x, period_len, router_module))

        phase_bank = torch.stack(phase_contexts, dim=2)  # (B, N, S, D)
        selector_in = torch.cat([real_ctx, imag_ctx, phase_bank.mean(dim=2)], dim=-1)
        scale_logits = self.scale_selector(selector_in)  # (B, N, S)
        scale_weights = torch.softmax(scale_logits, dim=-1).unsqueeze(-1)
        phase_ctx = torch.sum(scale_weights * phase_bank, dim=2)
        return phase_ctx, scale_weights.squeeze(-1)

    def Fre_Trans(self, x):
        b, n, t, d = x.shape
        x_time = x

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
        phase_ctx, _ = self._build_multiscale_phase_context(x_time, real_ctx, imag_ctx)

        gate_real_base = torch.sigmoid(self.real_from_imag_gate(imag_ctx))
        gate_imag_base = torch.sigmoid(self.imag_from_real_gate(real_ctx))

        phase_real_delta = torch.tanh(
            self.phase_real_gate(torch.cat([imag_ctx, phase_ctx], dim=-1))
        )
        phase_imag_delta = torch.tanh(
            self.phase_imag_gate(torch.cat([real_ctx, phase_ctx], dim=-1))
        )

        trust_real = torch.sigmoid(
            self.phase_real_trust(torch.cat([real_ctx, imag_ctx, phase_ctx], dim=-1))
        )
        trust_imag = torch.sigmoid(
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
        self.model = LinearFreTransformerCrossBranchPhaseGateMultiScale(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
