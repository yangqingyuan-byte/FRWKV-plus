import math

import torch
import torch.nn as nn

from model.FRWKV import LinearFreTransformerOptimized


class PeriodicPositionRouter(nn.Module):
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

    def forward(self, period_pos_tokens):
        # period_pos_tokens: (B, N, P, D)
        b, n, p, d = period_pos_tokens.shape
        x = self.input_norm(period_pos_tokens.reshape(b * n, p, d))

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        routers = self.routers.unsqueeze(0).expand(b * n, -1, -1)

        # Routers aggregate token information.
        scores_router = torch.matmul(routers, k.transpose(1, 2)) / math.sqrt(d)
        attn_router = torch.softmax(scores_router, dim=-1)
        router_buffer = torch.matmul(attn_router, v)

        # Tokens receive routed information back.
        scores_token = torch.matmul(q, router_buffer.transpose(1, 2)) / math.sqrt(d)
        attn_token = torch.softmax(scores_token, dim=-1)
        token_ctx = torch.matmul(attn_token, router_buffer)

        ctx = token_ctx.mean(dim=1).reshape(b, n, d)
        return self.out_proj(ctx)


PhaseRouterContext = PeriodicPositionRouter


class LinearFreTransformerCrossBranchPhaseGate(LinearFreTransformerOptimized):
    _STATE_DICT_RENAME_MAP = {
        "real_from_imag_gate": "base_im2re",
        "imag_from_real_gate": "base_re2im",
        "phase_router": "period_position_router",
        "phase_real_gate": "delta_impos2re",
        "phase_imag_gate": "delta_repos2im",
        "phase_alpha": "pos_alpha",
    }

    def __init__(self, configs):
        super().__init__(configs)
        self.period_position_len = getattr(
            configs, "period_position_len", getattr(configs, "phase_period_len", 24)
        )
        self.period_position_num_routers = getattr(
            configs, "period_position_num_routers", getattr(configs, "phase_num_routers", 4)
        )
        period_context_alpha_init = float(
            getattr(
                configs,
                "period_context_alpha_init",
                getattr(configs, "phase_alpha_init", 0.10),
            )
        )

        gate_hidden = max(32, self.embed_size * 8)
        delta_hidden = max(32, self.embed_size * 8)

        self.base_im2re = nn.Sequential(
            nn.Linear(self.embed_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.embed_size),
        )
        self.base_re2im = nn.Sequential(
            nn.Linear(self.embed_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.embed_size),
        )

        self.period_position_router = PeriodicPositionRouter(
            self.embed_size, self.period_position_num_routers
        )
        self.delta_impos2re = nn.Sequential(
            nn.Linear(self.embed_size * 2, delta_hidden),
            nn.GELU(),
            nn.Linear(delta_hidden, self.embed_size),
        )
        self.delta_repos2im = nn.Sequential(
            nn.Linear(self.embed_size * 2, delta_hidden),
            nn.GELU(),
            nn.Linear(delta_hidden, self.embed_size),
        )

        # Make the periodic-position correction nearly identity at initialization.
        nn.init.zeros_(self.delta_impos2re[-1].weight)
        nn.init.zeros_(self.delta_impos2re[-1].bias)
        nn.init.zeros_(self.delta_repos2im[-1].weight)
        nn.init.zeros_(self.delta_repos2im[-1].bias)

        self.pos_alpha = nn.Parameter(torch.tensor(period_context_alpha_init))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for old_name, new_name in self._STATE_DICT_RENAME_MAP.items():
            old_prefix = prefix + old_name
            new_prefix = prefix + new_name
            for key in list(state_dict.keys()):
                if key == old_prefix or key.startswith(old_prefix + "."):
                    mapped_key = new_prefix + key[len(old_prefix):]
                    if mapped_key not in state_dict:
                        state_dict[mapped_key] = state_dict[key]
                    state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _build_periodic_position_context(self, x):
        # x: (B, N, T, D)
        b, n, t, d = x.shape
        p = self.period_position_len
        if p <= 0:
            raise ValueError("period_position_len must be positive")

        pad_len = (p - (t % p)) % p
        if pad_len > 0:
            x = torch.cat([x, x[:, :, :pad_len, :]], dim=2)
        num_periods = x.size(2) // p

        period_pos_tokens = x.reshape(b, n, num_periods, p, d).mean(dim=2)
        return self.period_position_router(period_pos_tokens)

    def _build_phase_context(self, x):
        return self._build_periodic_position_context(x)

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

        base_re = torch.sigmoid(self.base_im2re(imag_ctx))
        base_im = torch.sigmoid(self.base_re2im(real_ctx))

        delta_re = torch.tanh(
            self.delta_impos2re(torch.cat([imag_ctx, period_pos_ctx], dim=-1))
        )
        delta_im = torch.tanh(
            self.delta_repos2im(torch.cat([real_ctx, period_pos_ctx], dim=-1))
        )

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
        self.model = LinearFreTransformerCrossBranchPhaseGate(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
