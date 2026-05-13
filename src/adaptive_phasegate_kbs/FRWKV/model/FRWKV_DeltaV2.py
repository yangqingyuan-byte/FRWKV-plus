import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN
from layers.Transformer_EncDec import EncoderStack, EncoderLayer
from model.FRWKV import remap_frequency_branch_state_dict_keys


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


class GatedDecayedDeltaAttention(nn.Module):
    """
    Delta v2:
    - decayed delta update inspired by GatedDeltaNet
    - separate remove/add keys inspired by RWKV7
    - lightweight global summary modulation inspired by TimeXer
    """

    def __init__(self, d_model, n_heads, token_num, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.token_num = token_num

        hidden = int(d_model * 1.25)
        low_rank = max(64, d_model // 8)

        self.pre_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False
        )
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=True),
        )

        self.q_proj = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=True),
        )
        self.k_remove_proj = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=True),
        )
        self.k_add_proj = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=True),
        )
        self.v_proj = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=True),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=True),
        )

        self.beta_proj = LoRAMLP(d_model, low_rank, bias=True)
        self.decay_proj = nn.Linear(d_model, n_heads, bias=True)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        self.norm = SimpleRMSNorm(d_model)
        self.dropout = nn.Dropout(dropout * 0.5)
        self.D = nn.Parameter(torch.ones(n_heads))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        del keys, values, attn_mask, tau, delta, token_weight
        bsz, tokens, channels = queries.shape
        heads, dim = self.n_heads, self.head_dim

        x = self.pre_conv(queries.transpose(1, 2)).transpose(1, 2)
        global_ctx = x.mean(dim=1, keepdim=True)
        x = x + self.global_proj(global_ctx)

        q = self.q_proj(x)
        k_remove = self.k_remove_proj(x)
        k_add = self.k_add_proj(x)
        v = self.v_proj(x)
        beta = torch.sigmoid(self.beta_proj(x))
        gate = F.silu(self.gate_proj(x))
        decay = torch.sigmoid(self.decay_proj(x)).unsqueeze(-1)

        q = q.view(bsz, tokens, heads, dim)
        k_remove = k_remove.view(bsz, tokens, heads, dim)
        k_add = k_add.view(bsz, tokens, heads, dim)
        v = v.view(bsz, tokens, heads, dim)
        beta = beta.view(bsz, tokens, heads, dim)
        decay = decay.view(bsz, tokens, heads, 1)

        q = F.normalize(F.silu(q), dim=-1)
        k_remove = F.normalize(F.silu(k_remove), dim=-1)
        k_add = F.normalize(F.silu(k_add), dim=-1)

        state = torch.zeros(bsz, heads, dim, dim, device=queries.device, dtype=queries.dtype)
        output = torch.zeros(bsz, tokens, heads, dim, device=queries.device, dtype=queries.dtype)

        for t in range(tokens):
            q_t = q[:, t]
            kr_t = k_remove[:, t]
            ka_t = k_add[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]
            decay_t = decay[:, t].unsqueeze(-1)

            state = state * decay_t
            retrieved = torch.einsum("bhij,bhj->bhi", state, kr_t)
            delta_v = beta_t * (v_t - retrieved)
            state = state + torch.einsum("bhi,bhj->bhij", delta_v, ka_t)
            output[:, t] = torch.einsum("bhij,bhj->bhi", state, q_t)

        output = output + self.D.view(1, 1, heads, 1) * v
        output = output.reshape(bsz, tokens, channels)
        output = self.norm(output)
        output = self.out_proj(output * gate)
        output = self.dropout(output)
        return output, None


class LinearFreTransformerDeltaV2(nn.Module):
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
                    GatedDecayedDeltaAttention(
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
                    GatedDecayedDeltaAttention(
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
        x = x.transpose(-1, -2)
        x_fre = torch.fft.rfft(x, dim=-1, norm="ortho")

        y_real, y_imag = x_fre.real, x_fre.imag
        y_real_input = y_real.flatten(-2)
        y_imag_input = y_imag.flatten(-2)

        y_real = (self.real_freq_branch(y_real_input) + y_real_input).reshape(
            bsz, nvars, dim, self.valid_fre_points
        )
        y_imag = (self.imag_freq_branch(y_imag_input) + y_imag_input).reshape(
            bsz, nvars, dim, self.valid_fre_points
        )

        y = torch.complex(y_real, y_imag)
        x = torch.fft.irfft(y, n=steps, dim=-1, norm="ortho")
        return x.transpose(-1, -2)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask
        x = self.revin_layer(x, mode="norm")
        x_emb = self.tokenEmb(x, self.embeddings)
        x_fre = self.Fre_Trans(x_emb)
        x = x_emb + x_fre
        out = self.fc(x.flatten(-2)).transpose(-1, -2)
        out = self.dropout(out)
        out = self.revin_layer(out, mode="denorm")
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LinearFreTransformerDeltaV2(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
