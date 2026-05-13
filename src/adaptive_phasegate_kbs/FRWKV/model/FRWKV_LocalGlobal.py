import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN
from layers.Transformer_EncDec import EncoderStack, EncoderLayer
from model.FRWKV import remap_frequency_branch_state_dict_keys


def lerp(a, b, x):
    return a + (b - a) * x


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.scale


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


class LocalGlobalLinearAttention(nn.Module):
    """
    Keep the original FRWKV recurrence, but add:
    - depthwise short convolution
    - global summary modulation
    - RMS-style output norm
    - residual value path
    """

    def __init__(self, d_model, n_heads, token_num, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.token_num = token_num

        self.pre_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False
        )
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.mu_r = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_k = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_v = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_g = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_a = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_d = nn.Parameter(torch.randn(d_model) * 0.02)

        lora_dim = max(64, d_model // 8)
        self.decay_lora = LoRAMLP(d_model, lora_dim, bias=True)
        self.iclr_lora = LoRAMLP(d_model, lora_dim, bias=True)
        self.gate_lora = LoRAMLP(d_model, lora_dim, bias=False)

        expanded_dim = int(d_model * 1.2)
        self.W_receptance = nn.Sequential(
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

        self.removal_key_multiplier = nn.Parameter(torch.randn(d_model) * 0.05)
        self.iclr_mix_amt = nn.Parameter(torch.full((d_model,), 0.7))
        self.bonus_multiplier = nn.Parameter(torch.ones(d_model) * 1.2)
        self.D = nn.Parameter(torch.ones(n_heads))
        self.norm = SimpleRMSNorm(d_model)
        self.dropout = nn.Dropout(dropout * 0.5)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        del keys, values, attn_mask, tau, delta, token_weight
        B, T, C = queries.shape
        H, N = self.n_heads, self.head_size

        x = self.pre_conv(queries.transpose(1, 2)).transpose(1, 2)
        global_ctx = x.mean(dim=1, keepdim=True)
        x = x + self.global_proj(global_ctx)

        x_receptance = x + self.mu_r
        x_key = x + self.mu_k
        x_value = x + self.mu_v
        x_gate = x + self.mu_g
        x_iclr = x + self.mu_a
        x_decay = x + self.mu_d

        r = self.W_receptance(x_receptance)
        k = self.W_key(x_key)
        v = self.W_value(x_value)

        gate = torch.sigmoid(self.gate_lora(x_gate))
        iclr = torch.sigmoid(self.iclr_lora(x_iclr))
        decay_precursor = torch.tanh(self.decay_lora(x_decay))
        decay = torch.exp(-math.exp(-0.3) * torch.sigmoid(decay_precursor))

        removal_k = k * self.removal_key_multiplier
        replacement_k = k * lerp(torch.ones_like(iclr), iclr, self.iclr_mix_amt)

        r = r.view(B, T, H, N)
        removal_k = removal_k.view(B, T, H, N)
        replacement_k = replacement_k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        decay = decay.view(B, T, H, N)
        iclr = iclr.view(B, T, H, N)

        removal_k_norm = F.normalize(removal_k, dim=-1)
        replacement_k_norm = F.normalize(replacement_k, dim=-1)

        wkv_state = torch.zeros(B, H, N, N, device=queries.device, dtype=queries.dtype)
        output = torch.zeros(B, T, H, N, device=queries.device, dtype=queries.dtype)

        for t in range(T):
            decay_t = decay[:, t]
            iclr_t = iclr[:, t]
            removal_k_norm_t = removal_k_norm[:, t]
            replacement_k_t = replacement_k[:, t]
            replacement_k_norm_t = replacement_k_norm[:, t]
            v_t = v[:, t]
            r_t = r[:, t]

            diag_decay = torch.diag_embed(decay_t)
            weighted_removal = iclr_t * removal_k_norm_t
            removal_outer = torch.einsum("bhi,bhj->bhij", removal_k_norm_t, weighted_removal)
            G_t = diag_decay - removal_outer

            wkv_state = torch.bmm(G_t.view(B * H, N, N), wkv_state.view(B * H, N, N)).view(B, H, N, N)
            update = torch.einsum("bhi,bhj->bhij", v_t, replacement_k_norm_t)
            wkv_state = wkv_state + update
            output[:, t] = torch.einsum("bhij,bhj->bhi", wkv_state, r_t)

        bonus_scalar = torch.sum(r * replacement_k, dim=-1, keepdim=True) * 0.2
        bonus_multiplier = self.bonus_multiplier.view(1, 1, H, N).expand(B, T, H, N)
        output = output + bonus_scalar * v * bonus_multiplier
        output = output + self.D.view(1, 1, H, 1) * v

        output = output.view(B, T, C)
        output = self.norm(output)
        output = output * gate
        output = self.dropout(self.W_output(output))
        return output, None


class LinearFreTransformerLocalGlobal(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.embed_size = configs.embed_size

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size) * 0.1)
        self.valid_fre_points = int((self.seq_len + 1) / 2 + 0.5)

        self.encoder_fre_real = EncoderStack(
            [
                EncoderLayer(
                    LocalGlobalLinearAttention(
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
            CKA_flag=configs.CKA_FLAG if hasattr(configs, "CKA_FLAG") else configs.CKA_flag,
        )

        self.encoder_fre_imag = EncoderStack(
            [
                EncoderLayer(
                    LocalGlobalLinearAttention(
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
        B, N, T, D = x.shape
        x = x.transpose(-1, -2)
        x_fre = torch.fft.rfft(x, dim=-1, norm="ortho")
        y_real, y_imag = x_fre.real, x_fre.imag
        y_real_input = y_real.flatten(-2)
        y_imag_input = y_imag.flatten(-2)
        y_real = (self.real_freq_branch(y_real_input) + y_real_input).reshape(B, N, D, self.valid_fre_points)
        y_imag = (self.imag_freq_branch(y_imag_input) + y_imag_input).reshape(B, N, D, self.valid_fre_points)
        y = torch.complex(y_real, y_imag)
        x = torch.fft.irfft(y, n=T, dim=-1, norm="ortho")
        return x.transpose(-1, -2)

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
        self.model = LinearFreTransformerLocalGlobal(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
