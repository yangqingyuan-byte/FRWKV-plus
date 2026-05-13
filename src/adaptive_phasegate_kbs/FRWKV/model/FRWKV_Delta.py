import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN
from layers.Transformer_EncDec import EncoderStack, EncoderLayer
from model.FRWKV import remap_frequency_branch_state_dict_keys


def lerp(a, b, x):
    return a + (b - a) * x


class LoRAMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=False):
        super().__init__()
        self.A = nn.Linear(input_dim, hidden_dim, bias=False)
        self.B = nn.Linear(hidden_dim, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim)) if bias else None

    def forward(self, x):
        out = self.B(self.A(x))
        if self.bias is not None:
            out = out + self.bias
        return out


class DeltaAttentionLayer(nn.Module):
    """
    A first Delta-style recurrent attention block for FRWKV.

    The overall FRWKV frequency backbone remains unchanged; only the
    recurrent state update is replaced with a delta-rule style update:

        S_t = S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
        y_t = S_t q_t

    This version is intentionally conservative so we can study the effect
    of the state-update rule without changing the rest of the pipeline.
    """

    def __init__(self, d_model, n_heads, token_num, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.token_num = token_num

        self.mu_q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_k = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_v = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_g = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_b = nn.Parameter(torch.randn(d_model) * 0.02)

        lora_dim = max(64, d_model // 8)
        self.beta_lora = LoRAMLP(d_model, lora_dim, bias=True)
        self.gate_lora = LoRAMLP(d_model, lora_dim, bias=False)

        expanded_dim = int(d_model * 1.2)
        self.W_query = nn.Sequential(
            nn.Linear(d_model, expanded_dim, bias=True),
            nn.GELU(),
            nn.Linear(expanded_dim, d_model, bias=True),
        )
        self.W_key = nn.Sequential(
            nn.Linear(d_model, expanded_dim, bias=True),
            nn.GELU(),
            nn.Linear(expanded_dim, d_model, bias=True),
        )
        self.W_value = nn.Sequential(
            nn.Linear(d_model, expanded_dim, bias=True),
            nn.GELU(),
            nn.Linear(expanded_dim, d_model, bias=True),
        )
        self.W_output = nn.Sequential(
            nn.Linear(d_model, expanded_dim, bias=True),
            nn.GELU(),
            nn.Linear(expanded_dim, d_model, bias=True),
        )

        # A lightweight local mixer helps the delta rule keep short-range dynamics.
        self.short_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False
        )
        self.bonus_multiplier = nn.Parameter(torch.ones(d_model) * 0.5)
        self.ln_x = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout * 0.5)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        del keys, values, attn_mask, tau, delta, token_weight
        bsz, steps, channels = queries.shape
        heads, dim = self.n_heads, self.head_size

        # Local temporal bias before recurrent updates.
        mixed = self.short_conv(queries.transpose(1, 2)).transpose(1, 2)
        x_q = mixed + self.mu_q
        x_k = mixed + self.mu_k
        x_v = mixed + self.mu_v
        x_g = mixed + self.mu_g
        x_b = mixed + self.mu_b

        q = self.W_query(x_q)
        k = self.W_key(x_k)
        v = self.W_value(x_v)
        gate = torch.sigmoid(self.gate_lora(x_g))
        beta = torch.sigmoid(self.beta_lora(x_b))

        q = q.view(bsz, steps, heads, dim)
        k = k.view(bsz, steps, heads, dim)
        v = v.view(bsz, steps, heads, dim)
        beta = beta.view(bsz, steps, heads, dim)

        # DeltaNet-style practical normalization.
        q = F.normalize(F.silu(q), dim=-1)
        k = F.normalize(F.silu(k), dim=-1)

        state = torch.zeros(bsz, heads, dim, dim, device=queries.device, dtype=queries.dtype)
        output = torch.zeros(bsz, steps, heads, dim, device=queries.device, dtype=queries.dtype)

        for t in range(steps):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]

            retrieved = torch.einsum("bhij,bhj->bhi", state, k_t)
            delta_v = beta_t * (v_t - retrieved)
            state = state + torch.einsum("bhi,bhj->bhij", delta_v, k_t)
            output[:, t] = torch.einsum("bhij,bhj->bhi", state, q_t)

        bonus_scalar = torch.sum(q * k, dim=-1, keepdim=True) * 0.1
        bonus_multiplier = self.bonus_multiplier.view(1, 1, heads, dim).expand(bsz, steps, heads, dim)
        output = output + bonus_scalar * v * bonus_multiplier

        output = output.reshape(bsz, steps, channels)
        output = self.ln_x(output)
        output = output * gate
        output = self.dropout(self.W_output(output))
        return output, None


class LinearFreTransformerDelta(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads

        self.patch_len = configs.temp_patch_len
        self.stride = configs.temp_stride
        self.embed_size = configs.embed_size

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size) * 0.1)
        self.embeddings2 = nn.Parameter(torch.randn(1, self.embed_size) * 0.1)
        self.embeddings_time = nn.Parameter(torch.randn(1, self.embed_size) * 0.1)

        self.valid_fre_points = int((self.seq_len + 1) / 2 + 0.5)

        self.encoder_fre_real = EncoderStack(
            [
                EncoderLayer(
                    DeltaAttentionLayer(
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        token_num=configs.enc_in,
                        dropout=configs.dropout * 0.5,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout * 0.5,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag,
        )

        self.encoder_fre_imag = EncoderStack(
            [
                EncoderLayer(
                    DeltaAttentionLayer(
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        token_num=configs.enc_in,
                        dropout=configs.dropout * 0.5,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout * 0.5,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag,
        )

        self.real_freq_branch = nn.Sequential(
            nn.Linear(self.valid_fre_points * self.embed_size, self.d_model),
            self.encoder_fre_real,
            nn.Linear(self.d_model, self.valid_fre_points * self.embed_size),
        )

        self.imag_freq_branch = nn.Sequential(
            nn.Linear(self.valid_fre_points * self.embed_size, self.d_model),
            self.encoder_fre_imag,
            nn.Linear(self.d_model, self.valid_fre_points * self.embed_size),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout * 0.3),
            nn.Linear(self.d_ff, self.d_ff // 2),
            nn.GELU(),
            nn.Dropout(configs.dropout * 0.2),
            nn.Linear(self.d_ff // 2, self.pred_len),
        )

        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout * 0.5)

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
        remap_frequency_branch_state_dict_keys(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        return x * embeddings

    def Fre_Trans(self, x):
        bsz, nvars, steps, dim = x.shape
        assert steps == self.seq_len
        x = x.transpose(-1, -2)

        x_fre = torch.fft.rfft(x, dim=-1, norm="ortho")
        assert x_fre.shape[-1] == self.valid_fre_points

        y_real, y_imag = x_fre.real, x_fre.imag

        y_real_input = y_real.flatten(-2)
        y_real_output = self.real_freq_branch(y_real_input)
        y_real = (y_real_output + y_real_input).reshape(bsz, nvars, dim, self.valid_fre_points)

        y_imag_input = y_imag.flatten(-2)
        y_imag_output = self.imag_freq_branch(y_imag_input)
        y_imag = (y_imag_output + y_imag_input).reshape(bsz, nvars, dim, self.valid_fre_points)

        y = torch.complex(y_real, y_imag)
        x = torch.fft.irfft(y, n=steps, dim=-1, norm="ortho")
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        x = self.revin_layer(x, mode="norm")
        x_emb = self.tokenEmb(x, self.embeddings)
        x_fre = self.Fre_Trans(x_emb)
        x = x_fre + x_emb
        out = self.fc(x.flatten(-2)).transpose(-1, -2)
        out = self.dropout(out)
        out = self.revin_layer(out, mode="denorm")
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LinearFreTransformerDelta(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
