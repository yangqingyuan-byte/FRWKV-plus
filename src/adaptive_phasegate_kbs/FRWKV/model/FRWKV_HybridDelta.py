import torch
import torch.nn as nn

from layers.RevIN import RevIN
from layers.Transformer_EncDec import EncoderStack, EncoderLayer

from model.FRWKV import FrequencyRWKVBlock, remap_frequency_branch_state_dict_keys
from model.FRWKV_DeltaV2 import GatedDecayedDeltaAttention


def build_hybrid_attention_layers(configs):
    layers = []
    for layer_idx in range(configs.e_layers):
        # Alternate original FRWKV and Delta-v2 blocks.
        if layer_idx % 2 == 0:
            attn = FrequencyRWKVBlock(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                token_num=configs.enc_in,
                dropout=configs.dropout * 0.5,
            )
        else:
            attn = GatedDecayedDeltaAttention(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                token_num=configs.enc_in,
                dropout=configs.dropout * 0.5,
            )
        layers.append(
            EncoderLayer(
                attn,
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout * 0.5,
                activation=configs.activation,
            )
        )
    return layers


class LinearFreTransformerHybridDelta(nn.Module):
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
            build_hybrid_attention_layers(configs),
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag,
        )

        self.encoder_fre_imag = EncoderStack(
            build_hybrid_attention_layers(configs),
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
        self.model = LinearFreTransformerHybridDelta(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
