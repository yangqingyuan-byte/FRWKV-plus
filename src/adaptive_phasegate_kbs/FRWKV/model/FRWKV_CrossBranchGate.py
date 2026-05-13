import torch
import torch.nn as nn

from model.FRWKV import LinearFreTransformerOptimized


class LinearFreTransformerCrossBranchGate(LinearFreTransformerOptimized):
    _STATE_DICT_RENAME_MAP = {
        "real_from_imag_gate": "base_im2re",
        "imag_from_real_gate": "base_re2im",
    }

    def __init__(self, configs):
        super().__init__(configs)
        gate_hidden = max(32, self.embed_size * 8)
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

    def Fre_Trans(self, x):
        B, N, T, D = x.shape
        x = x.transpose(-1, -2)

        x_fre = torch.fft.rfft(x, dim=-1, norm="ortho")
        y_real, y_imag = x_fre.real, x_fre.imag

        y_real_input = y_real.flatten(-2)
        y_imag_input = y_imag.flatten(-2)
        y_real = (self.real_freq_branch(y_real_input) + y_real_input).reshape(B, N, D, self.valid_fre_points)
        y_imag = (self.imag_freq_branch(y_imag_input) + y_imag_input).reshape(B, N, D, self.valid_fre_points)

        real_ctx = y_real.mean(dim=-1)
        imag_ctx = y_imag.mean(dim=-1)
        base_re = torch.sigmoid(self.base_im2re(imag_ctx))
        base_im = torch.sigmoid(self.base_re2im(real_ctx))

        gate_real = (1.0 + base_re).unsqueeze(-1)
        gate_imag = (1.0 + base_im).unsqueeze(-1)

        y_real = y_real * gate_real
        y_imag = y_imag * gate_imag

        y = torch.complex(y_real, y_imag)
        x = torch.fft.irfft(y, n=T, dim=-1, norm="ortho")
        x = x.transpose(-1, -2)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = LinearFreTransformerCrossBranchGate(configs)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x, x_mark_enc, x_dec, x_mark_dec, mask)
