import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.Transformer_EncDec import EncoderStack, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import torch.nn.functional as F
import math

def lerp(a, b, x):
    """Linear interpolation: a + (b - a) * x"""
    return a + (b - a) * x

class LoRAMLP(nn.Module):
    """Low-rank MLP for dynamic parameter generation"""
    def __init__(self, input_dim, hidden_dim, bias=False):
        super().__init__()
        self.A = nn.Linear(input_dim, hidden_dim, bias=False)
        self.B = nn.Linear(hidden_dim, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim)) if bias else None
        
    def forward(self, x):
        result = self.B(self.A(x))
        if self.bias is not None:
            result = result + self.bias
        return result

class FrequencyRWKVBlock(nn.Module):
    
    def __init__(self, d_model, n_heads, token_num, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.token_num = token_num
        
        # 优化的Token Shift机制 - 使用可学习的参数
        self.mu_r = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_k = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_v = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_g = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_a = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mu_d = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # 优化的LoRA MLPs - 增加维度提升表达能力
        lora_dim = max(64, d_model // 8)  # 增加LoRA维度
        self.decay_lora = LoRAMLP(d_model, lora_dim, bias=True)
        self.iclr_lora = LoRAMLP(d_model, lora_dim, bias=True)
        self.gate_lora = LoRAMLP(d_model, lora_dim, bias=False)
        
        # 优化的线性层 - 使用更大的维度提升表达能力
        expanded_dim = int(d_model * 1.2)  # 增加20%的维度
        self.W_receptance = nn.Sequential(
            nn.Linear(d_model, expanded_dim, bias=True),
            nn.GELU(),  # 使用GELU激活函数
            nn.Linear(expanded_dim, d_model, bias=True)
        )
        self.W_key = nn.Sequential(
            nn.Linear(d_model, expanded_dim, bias=True),
            nn.GELU(),
            nn.Linear(expanded_dim, d_model, bias=True)
        )
        self.W_value = nn.Sequential(
            nn.Linear(d_model, expanded_dim, bias=True),
            nn.GELU(),
            nn.Linear(expanded_dim, d_model, bias=True)
        )
        self.W_output = nn.Sequential(
            nn.Linear(d_model, expanded_dim, bias=True),
            nn.GELU(),
            nn.Linear(expanded_dim, d_model, bias=True)
        )
        
        # 优化的RWKV7关键参数
        self.removal_key_multiplier = nn.Parameter(torch.randn(d_model) * 0.05)
        self.iclr_mix_amt = nn.Parameter(torch.full((d_model,), 0.7))  # 增加混合比例
        self.bonus_multiplier = nn.Parameter(torch.ones(d_model) * 1.2)  # 增加bonus效果
        
        # 优化的归一化层
        self.ln_x = nn.LayerNorm(d_model, eps=1e-6)  # 使用LayerNorm替代GroupNorm
        self.dropout = nn.Dropout(dropout * 0.5)  # 减少dropout率
        
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        B, T, C = queries.shape
        H, N = self.n_heads, self.head_size
        
        # 优化的Token Shift机制
        x_receptance = queries + self.mu_r
        x_key = queries + self.mu_k
        x_value = queries + self.mu_v
        x_gate = queries + self.mu_g
        x_iclr = queries + self.mu_a
        x_decay = queries + self.mu_d
        
        # 基础变换
        r = self.W_receptance(x_receptance)
        k = self.W_key(x_key)
        v = self.W_value(x_value)
        
        # 动态参数生成 - 优化的LoRA机制
        gate = torch.sigmoid(self.gate_lora(x_gate))
        iclr = torch.sigmoid(self.iclr_lora(x_iclr))
        decay_precursor = torch.tanh(self.decay_lora(x_decay))
        
        # Decay计算 - 优化参数
        decay = torch.exp(-math.exp(-0.3) * torch.sigmoid(decay_precursor))  # 调整衰减参数
        
        # Key processing - 优化处理
        removal_k = k * self.removal_key_multiplier
        replacement_k = k * lerp(torch.ones_like(iclr), iclr, self.iclr_mix_amt)
        
        # 重塑为多头格式
        r = r.view(B, T, H, N)
        removal_k = removal_k.view(B, T, H, N)
        replacement_k = replacement_k.view(B, T, H, N)
        v = v.view(B, T, H, N)
        decay = decay.view(B, T, H, N)
        iclr = iclr.view(B, T, H, N)
        
        # 优化的归一化
        removal_k_norm = F.normalize(removal_k, dim=-1)
        
        # 状态矩阵机制 - 优化的WKV计算
        wkv_state = torch.zeros(B, H, N, N, device=queries.device, dtype=queries.dtype)
        output = torch.zeros(B, T, H, N, device=queries.device, dtype=queries.dtype)
        
        for t in range(T):
            decay_t = decay[:, t]
            iclr_t = iclr[:, t]
            removal_k_norm_t = removal_k_norm[:, t]
            replacement_k_t = replacement_k[:, t]
            v_t = v[:, t]
            r_t = r[:, t]
            
            # Transition matrix构建
            diag_decay = torch.diag_embed(decay_t)
            weighted_removal = iclr_t * removal_k_norm_t
            removal_outer = torch.einsum('bhi,bhj->bhij', removal_k_norm_t, weighted_removal)
            G_t = diag_decay - removal_outer
            
            # 状态更新 - 线性复杂度
            wkv_state = torch.bmm(G_t.view(B*H, N, N), wkv_state.view(B*H, N, N)).view(B, H, N, N)
            update = torch.einsum('bhi,bhj->bhij', v_t, replacement_k_t)
            wkv_state = wkv_state + update
            output[:, t] = torch.einsum('bhij,bhj->bhi', wkv_state, r_t)
        
        # 优化的Bonus机制 - 增强性能
        bonus_scalar = torch.sum(r * replacement_k, dim=-1, keepdim=True) * 0.2
        # 将bonus_multiplier重塑为多头格式
        bonus_multiplier_reshaped = self.bonus_multiplier.view(1, 1, H, N).expand(B, T, H, N)
        bonus = bonus_scalar * v * bonus_multiplier_reshaped
        output = output + bonus
        
        # 重塑回原始格式
        output = output.view(B, T, C)
        
        # 优化的归一化和门控
        output = self.ln_x(output)
        output = output * gate
        output = self.dropout(self.W_output(output))
        
        return output, None

FREQUENCY_BRANCH_STATE_DICT_RENAME_MAP = {
    "fre_trans_real": "real_freq_branch",
    "fre_trans_imag": "imag_freq_branch",
}


def remap_frequency_branch_state_dict_keys(state_dict, prefix):
    for old_name, new_name in FREQUENCY_BRANCH_STATE_DICT_RENAME_MAP.items():
        old_prefix = prefix + old_name
        new_prefix = prefix + new_name
        for key in list(state_dict.keys()):
            if key == old_prefix or key.startswith(old_prefix + "."):
                mapped_key = new_prefix + key[len(old_prefix):]
                if mapped_key not in state_dict:
                    state_dict[mapped_key] = state_dict[key]
                state_dict.pop(key)


class LinearFreTransformerOptimized(nn.Module):
    
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
        self.fre_flatten_mode = getattr(configs, 'fre_flatten_mode', 'joint')
        self.encoder_attention_type = getattr(configs, 'encoder_attention_type', 'linear')

        valid_modes = {'joint', 'embed_tokens', 'freq_mlp'}
        if self.fre_flatten_mode not in valid_modes:
            raise ValueError(
                f"Unsupported fre_flatten_mode={self.fre_flatten_mode}. "
                f"Expected one of {sorted(valid_modes)}."
            )
        valid_attention_types = {'linear', 'standard'}
        if self.encoder_attention_type not in valid_attention_types:
            raise ValueError(
                f"Unsupported encoder_attention_type={self.encoder_attention_type}. "
                f"Expected one of {sorted(valid_attention_types)}."
            )
        
        # 优化的嵌入层 - 使用更好的初始化
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size) * 0.1)
        self.embeddings2 = nn.Parameter(torch.randn(1, self.embed_size) * 0.1)
        self.embeddings_time = nn.Parameter(torch.randn(1, self.embed_size) * 0.1)

        self.valid_fre_points = int((self.seq_len + 1) / 2 + 0.5)

        self.real_freq_branch = self._build_frequency_branch(configs)
        self.imag_freq_branch = self._build_frequency_branch(configs)

        # 优化的输出层 - 增加层数和更好的激活函数
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout * 0.3),  # 减少dropout
            nn.Linear(self.d_ff, self.d_ff // 2),
            nn.GELU(),
            nn.Dropout(configs.dropout * 0.2),  # 进一步减少dropout
            nn.Linear(self.d_ff // 2, self.pred_len)
        )
        
        # RevIN层
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout * 0.5)  # 减少dropout

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

    def _build_frequency_attention(self, configs, token_num, dropout):
        if self.encoder_attention_type == 'linear':
            return FrequencyRWKVBlock(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                token_num=token_num,
                dropout=dropout,
            )

        return AttentionLayer(
            FullAttention(
                mask_flag=False,
                attention_dropout=dropout,
                output_attention=False,
                token_num=token_num,
                imp_mode=False,
                ij_mat_flag=False,
                num_heads=configs.n_heads,
                plot_mat_flag=False,
            ),
            d_model=configs.d_model,
            n_heads=configs.n_heads,
        )

    def _build_frequency_encoder(self, configs, token_num):
        attn_dropout = configs.dropout * 0.5
        return EncoderStack(
            [
                EncoderLayer(
                    self._build_frequency_attention(
                        configs,
                        token_num=token_num,
                        dropout=attn_dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=attn_dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=configs.CKA_flag
        )

    def _build_frequency_branch(self, configs):
        if self.fre_flatten_mode == 'joint':
            input_dim = self.valid_fre_points * self.embed_size
            encoder = self._build_frequency_encoder(configs, token_num=configs.enc_in)
            return nn.Sequential(
                nn.Linear(input_dim, self.d_model),
                encoder,
                nn.Linear(self.d_model, input_dim)
            )

        if self.fre_flatten_mode == 'embed_tokens':
            encoder = self._build_frequency_encoder(configs, token_num=self.embed_size)
            return nn.Sequential(
                nn.Linear(self.valid_fre_points, self.d_model),
                encoder,
                nn.Linear(self.d_model, self.valid_fre_points)
            )

        if self.fre_flatten_mode == 'freq_mlp':
            return nn.Sequential(
                nn.Linear(self.valid_fre_points, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.valid_fre_points)
            )

        raise ValueError(f"Unsupported fre_flatten_mode: {self.fre_flatten_mode}")

    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        return x * embeddings

    def _process_frequency_branch(self, branch, y):
        B, N, D, F = y.shape

        if self.fre_flatten_mode == 'joint':
            y_input = y.flatten(-2)
            y_output = branch(y_input)
            return (y_output + y_input).reshape(B, N, D, F)

        if self.fre_flatten_mode == 'embed_tokens':
            y_input = y.reshape(B * N, D, F)
            y_output = branch(y_input)
            return (y_output + y_input).reshape(B, N, D, F)

        if self.fre_flatten_mode == 'freq_mlp':
            y_output = branch(y)
            return y_output + y

        raise ValueError(f"Unsupported fre_flatten_mode: {self.fre_flatten_mode}")

    def Fre_Trans(self, x):
        B, N, T, D = x.shape
        assert T == self.seq_len
        x = x.transpose(-1, -2)

        # FFT变换
        x_fre = torch.fft.rfft(x, dim=-1, norm='ortho')
        assert x_fre.shape[-1] == self.valid_fre_points

        y_real, y_imag = x_fre.real, x_fre.imag

        # 支持当前的联合 flatten，以及不 flatten 的对照分支
        y_real = self._process_frequency_branch(self.real_freq_branch, y_real)
        y_imag = self._process_frequency_branch(self.imag_freq_branch, y_imag)
        
        y = torch.complex(y_real, y_imag)

        # 逆FFT变换
        x = torch.fft.irfft(y, n=T, dim=-1, norm='ortho')
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, T, N = x.shape

        # RevIN归一化
        x = self.revin_layer(x, mode='norm')

        # 优化的频域处理 - 增强残差连接
        x_emb = self.tokenEmb(x, self.embeddings)
        x_fre = self.Fre_Trans(x_emb)
        x = x_fre + x_emb  # 残差连接

        # 优化的输出映射
        out = self.fc(x.flatten(-2)).transpose(-1, -2)
        out = self.dropout(out)

        # RevIN反归一化
        out = self.revin_layer(out, mode='denorm')

        return out

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.model = LinearFreTransformerOptimized(configs)
    
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
