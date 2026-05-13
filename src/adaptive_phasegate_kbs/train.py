import torch
import torch.distributed as dist
from torch import optim
import numpy as np
import argparse
import math
import time
import os
import sys
import random
import h5py
import inspect
import copy
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from adaptive_phasegate_kbs.data_provider.data_loader_emb import (
    Dataset_ETT_hour as Dataset_ETT_hour_Emb,
    Dataset_ETT_minute as Dataset_ETT_minute_Emb,
    Dataset_Custom as Dataset_Custom_Emb,
)
from adaptive_phasegate_kbs.data_provider.data_loader_save import (
    Dataset_ETT_hour as Dataset_ETT_hour_Plain,
    Dataset_ETT_minute as Dataset_ETT_minute_Plain,
    Dataset_Custom as Dataset_Custom_Plain,
)
from adaptive_phasegate_kbs.utils.metrics import MSE, MAE, metric
from adaptive_phasegate_kbs.utils.experiment_logger import log_experiment_result
from adaptive_phasegate_kbs.utils.kbs_recipe_config import resolve_recipe_args
import faulthandler
faulthandler.enable()
torch.cuda.empty_cache()
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:150")


MODEL_REGISTRY = {
    "frwkv": {
        "module": "FRWKV_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_delta": {
        "module": "FRWKV_DELTA_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_delta_v2": {
        "module": "FRWKV_DELTAV2_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_hybrid_delta": {
        "module": "FRWKV_HYBRIDDELTA_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_hybrid_split_delta": {
        "module": "FRWKV_HYBRIDSPLITDELTA_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_localglobal": {
        "module": "FRWKV_LOCALGLOBAL_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_branchgate": {
        "module": "FRWKV_BRANCHGATE_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchgate": {
        "module": "FRWKV_CROSSBRANCHGATE_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchphasegate": {
        "module": "FRWKV_CROSSBRANCHPHASEGATE_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchphasegate_fullcontextdelta": {
        "module": "FRWKV_CROSSBRANCHPHASEGATEFULLCONTEXTDELTA_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchperiodicpositiongate_adaptive": {
        "module": "FRWKV_CROSSBRANCHPHASEGATEADAPTIVE_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchphasegate_adaptive": {
        "module": "FRWKV_CROSSBRANCHPHASEGATEADAPTIVE_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchphasegate_adaptive_channelemb": {
        "module": "FRWKV_CROSSBRANCHPHASEGATEADAPTIVECHANNELEMB_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchphasegate_adaptive_linearproj": {
        "module": "FRWKV_CROSSBRANCHPHASEGATEADAPTIVELINEARPROJ_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchphasegate_adaptive_patchfoldemb": {
        "module": "FRWKV_CROSSBRANCHPHASEGATEADAPTIVEPATCHFOLDEMB_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchphasegate_usefulness": {
        "module": "FRWKV_CROSSBRANCHPHASEGATEUSEFULNESS_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchphasegate_multiscale": {
        "module": "FRWKV_CROSSBRANCHPHASEGATEMULTISCALE_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchgate_v2": {
        "module": "FRWKV_CROSSBRANCHGATEV2_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_crossbranchgate_v15": {
        "module": "FRWKV_CROSSBRANCHGATEV15_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
    "frwkv_hybrid_split_delta_gate": {
        "module": "FRWKV_HYBRIDSPLITDELTAGATE_WRAPPER",
        "class": "Model",
        "default_use_embeddings": False,
    },
}


class WeightedSequenceLoss:
    def __init__(
        self,
        alpha: float = 0.0,
        loss_mode: str = "L2",
        huber_delta: float = 1.0,
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
        huber_weight: float = 1.0,
    ):
        self.alpha = alpha
        self.loss_mode = loss_mode
        self.huber_delta = huber_delta
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.huber_weight = huber_weight
        self.smooth_l1_beta = 1.0
        self.corr_weight = 0.0
        self._weight_cache = {}

    def _get_weights(self, horizon, device, dtype):
        key = (horizon, str(device), dtype)
        weights = self._weight_cache.get(key)
        if weights is None:
            weights = torch.tensor(
                [(i + 1) ** (-self.alpha) for i in range(horizon)],
                device=device,
                dtype=dtype,
            ).view(1, horizon, 1)
            self._weight_cache[key] = weights
        return weights

    def __call__(self, pred, true):
        abs_err = torch.abs(pred - true)
        sq_err = (pred - true) ** 2
        if self.huber_delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {self.huber_delta}")
        if self.smooth_l1_beta <= 0:
            raise ValueError(f"smooth_l1_beta must be positive, got {self.smooth_l1_beta}")
        huber_err = torch.where(
            abs_err <= self.huber_delta,
            0.5 * sq_err,
            self.huber_delta * (abs_err - 0.5 * self.huber_delta),
        )
        smooth_l1_err = torch.where(
            abs_err < self.smooth_l1_beta,
            0.5 * sq_err / self.smooth_l1_beta,
            abs_err - 0.5 * self.smooth_l1_beta,
        )

        if self.loss_mode == "L1":
            loss_vec = self.l1_weight * abs_err
        elif self.loss_mode == "L2":
            loss_vec = self.l2_weight * sq_err
        elif self.loss_mode == "L1L2":
            loss_vec = self.l1_weight * abs_err + self.l2_weight * sq_err
        elif self.loss_mode == "SmoothL1":
            loss_vec = smooth_l1_err
        elif self.loss_mode == "Huber":
            loss_vec = self.huber_weight * huber_err
        elif self.loss_mode == "L2Huber":
            loss_vec = self.l2_weight * sq_err + self.huber_weight * huber_err
        elif self.loss_mode == "MSECorr":
            loss_vec = self.l2_weight * sq_err
        else:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}")

        if pred.ndim >= 3:
            horizon = pred.shape[1]
            weights = self._get_weights(horizon, pred.device, pred.dtype)
            base_loss = torch.mean(loss_vec * weights)
        else:
            base_loss = torch.mean(loss_vec)

        if self.loss_mode == "MSECorr":
            pred_center = pred.float() - pred.float().mean(dim=1, keepdim=True)
            true_center = true.float() - true.float().mean(dim=1, keepdim=True)
            cov = (pred_center * true_center).sum(dim=1)
            std_pred = torch.sqrt((pred_center ** 2).sum(dim=1) + 1e-8)
            std_true = torch.sqrt((true_center ** 2).sum(dim=1) + 1e-8)
            corr = cov / (std_pred * std_true)
            corr_loss = 1 - corr.mean()
            return base_loss + self.corr_weight * corr_loss

        return base_loss


class GlobalBatchDistributedSampler(Sampler):
    def __init__(
        self,
        dataset,
        global_batch_size,
        num_replicas,
        rank,
        shuffle=True,
        seed=0,
        drop_last=True,
    ):
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if global_batch_size % num_replicas != 0:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by world size {num_replicas} "
                "when ddp_batch_size_mode='global'."
            )
        self.local_batch_size = global_batch_size // num_replicas

    def __iter__(self):
        dataset_size = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(dataset_size, generator=generator).tolist()
        else:
            indices = list(range(dataset_size))

        if self.drop_last:
            total_size = (len(indices) // self.global_batch_size) * self.global_batch_size
            indices = indices[:total_size]
        else:
            remainder = len(indices) % self.global_batch_size
            if remainder:
                pad_size = self.global_batch_size - remainder
                indices.extend(indices[:pad_size])

        rank_start = self.rank * self.local_batch_size
        rank_end = rank_start + self.local_batch_size
        local_indices = []
        for offset in range(0, len(indices), self.global_batch_size):
            batch_indices = indices[offset: offset + self.global_batch_size]
            local_indices.extend(batch_indices[rank_start:rank_end])

        return iter(local_indices)

    def __len__(self):
        dataset_size = len(self.dataset)
        if self.drop_last:
            num_global_batches = dataset_size // self.global_batch_size
        else:
            num_global_batches = math.ceil(dataset_size / self.global_batch_size)
        return num_global_batches * self.local_batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch


def resolve_bool_mode(value: str, auto_value: bool) -> bool:
    if value == "auto":
        return auto_value
    return value == "true"


def get_dataloader_kwargs(args):
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def resolve_local_train_batch_size(args, world_size):
    if args.ddp_batch_size_mode == "per_rank":
        return args.batch_size, args.batch_size * world_size

    if args.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} is not divisible by world size {world_size}. "
            "Either choose a divisible batch size or use --ddp_batch_size_mode per_rank."
        )
    return args.batch_size // world_size, args.batch_size


def move_to_device(batch_item, device):
    if batch_item is None:
        return None
    if torch.is_tensor(batch_item):
        return batch_item.to(device=device, dtype=torch.float32, non_blocking=True)
    return torch.as_tensor(batch_item, dtype=torch.float32, device=device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="optional JSON recipe config file; defaults to packaged configs/kbs_ours_recipes.json",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="",
        help="optional recipe name inside the JSON config file",
    )
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument(
        "--model_type",
        type=str,
        default="frwkv_crossbranchphasegate_adaptive",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model variant used in the paper-release package",
    )
    parser.add_argument("--data_path", type=str, default="ETTm1", help="data path")
    parser.add_argument("--channel", type=int, default=32, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout for models that use standard naming")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--d_model", type=int, default=512, help="model hidden size for FRWKV-style models")
    parser.add_argument("--d_ff", type=int, default=2048, help="ffn hidden size for FRWKV-style models")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads for FRWKV-style models")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers for FRWKV-style models")
    parser.add_argument(
        "--encoder_attention_type",
        type=str,
        default="linear",
        choices=["linear", "standard"],
        help="attention implementation inside FRWKV EncoderLayer",
    )
    parser.add_argument("--temp_patch_len", type=int, default=16, help="temporal patch length")
    parser.add_argument("--temp_stride", type=int, default=8, help="temporal patch stride")
    parser.add_argument("--embed_size", type=int, default=8, help="frequency embedding size")
    parser.add_argument(
        "--period_position_len",
        type=int,
        default=None,
        help="period length for periodic position context",
    )
    parser.add_argument(
        "--period_position_list",
        type=str,
        default=None,
        help="comma-separated periods for multiscale periodic position context",
    )
    parser.add_argument(
        "--period_position_num_routers",
        type=int,
        default=None,
        help="number of periodic position routers",
    )
    parser.add_argument(
        "--period_context_alpha_init",
        type=float,
        default=None,
        help="initial periodic context injection strength",
    )
    parser.add_argument(
        "--period_context_trust_bias_init",
        type=float,
        default=None,
        help="initial bias for periodic context trust gate",
    )
    parser.add_argument("--phase_period_len", type=int, default=24, help="legacy name for period_position_len")
    parser.add_argument(
        "--phase_period_list",
        type=str,
        default="12,24,48",
        help="legacy name for period_position_list",
    )
    parser.add_argument("--phase_num_routers", type=int, default=4, help="legacy name for period_position_num_routers")
    parser.add_argument("--phase_alpha_init", type=float, default=0.10, help="legacy name for period_context_alpha_init")
    parser.add_argument("--phase_trust_bias_init", type=float, default=-2.0, help="legacy name for period_context_trust_bias_init")
    parser.add_argument("--use_revin", type=int, default=1, help="whether to enable RevIN in FRWKV-family models")
    parser.add_argument("--revin_affine", type=int, default=1, help="whether RevIN uses learnable affine parameters")
    parser.add_argument("--activation", type=str, default="gelu", help="activation type")
    parser.add_argument("--CKA_flag", type=int, default=0, help="FRWKV auxiliary flag")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="L2",
        choices=["L1", "L2", "L1L2", "SmoothL1", "Huber", "L2Huber", "MSECorr"],
        help="training loss type; L2 is MSE-style",
    )
    parser.add_argument(
        "--lossfun_alpha",
        type=float,
        default=0.0,
        help="horizon weighting alpha for sequence loss",
    )
    parser.add_argument("--loss_huber_delta", type=float, default=1.0, help="delta used by Huber-style losses")
    parser.add_argument("--loss_l1_weight", type=float, default=1.0, help="L1 component weight in composite losses")
    parser.add_argument("--loss_l2_weight", type=float, default=1.0, help="L2 component weight in composite losses")
    parser.add_argument("--loss_huber_weight", type=float, default=1.0, help="Huber component weight in composite losses")
    parser.add_argument("--loss_smooth_l1_beta", type=float, default=0.2, help="beta used by SmoothL1-style loss")
    parser.add_argument("--loss_corr_weight", type=float, default=0.0, help="correlation-loss weight used by MSECorr")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch_factor when num_workers > 0")
    parser.add_argument("--train_percent", type=int, default=100, help="percentage of the training split to use")
    parser.add_argument(
        "--gpu_devices",
        type=str,
        default="",
        help="comma-separated visible GPU ids for DataParallel, e.g. '0,1,2,3'",
    )
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--epochs", type=int, default=150, help="")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument("--es_patience", type=int, default=25, help="quit if no improvement after this many iterations")    
    parser.add_argument("--save", type=str, default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-", help="save path")
    parser.add_argument("--model_tag", type=str, default="", help="extra tag for ablation bookkeeping")
    parser.add_argument("--embed_version", type=str, default="original", 
                        help="嵌入版本标识，用于指定使用哪个版本的embeddings（如 'original', 'wavelet', 'gpt2'）")
    parser.add_argument(
        "--use_embeddings",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="是否使用预计算 embeddings；auto 将根据 model_type 自动决定",
    )
    parser.add_argument("--save_model", action="store_true", default=False,
                        help="是否保存模型文件（默认不保存）")
    parser.add_argument(
        "--speed_mode",
        type=str,
        default="strict",
        choices=["strict", "aggressive"],
        help="strict 尽量保持原始数值路径；aggressive 优先训练速度",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="auto",
        choices=["auto", "off", "fp16", "bf16"],
        help="混合精度模式；auto 在 aggressive 下默认 fp16，其余为 off",
    )
    parser.add_argument(
        "--ddp_batch_size_mode",
        type=str,
        default="global",
        choices=["global", "per_rank"],
        help="分布式训练时 batch_size 的语义：global 表示总 batch，per_rank 表示每卡 batch",
    )
    parser.add_argument(
        "--sync_batchnorm",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="分布式训练时是否启用 SyncBatchNorm；auto 会在 FRWKV 主链路上启用",
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="DDP find_unused_parameters",
    )
    parser.add_argument(
        "--ddp_static_graph",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="DDP static_graph",
    )
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="DDP gradient_as_bucket_view",
    )
    parser.add_argument(
        "--ddp_bucket_cap_mb",
        type=int,
        default=100,
        help="DDP bucket_cap_mb",
    )
    parser.add_argument(
        "--allow_tf32",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="是否允许 TF32 matmul/cudnn",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="是否启用 cudnn benchmark",
    )
    parser.add_argument(
        "--deterministic_train",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="是否尽量走 deterministic 训练路径",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        default=False,
        help="在单卡训练路径上尝试启用 torch.compile",
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    parser.add_argument(
        "--torch_compile_backend",
        type=str,
        default="inductor",
        help="torch.compile backend",
    )
    parser.add_argument(
        "--torch_compile_fullgraph",
        action="store_true",
        default=False,
        help="是否对 torch.compile 使用 fullgraph=True",
    )
    initial_args, _ = parser.parse_known_args()
    if initial_args.config_name:
        try:
            recipe_defaults = resolve_recipe_args(
                initial_args.config_name,
                config_path=initial_args.config_file or None,
            )
        except Exception as exc:
            parser.error(str(exc))
        parser.set_defaults(**recipe_defaults)
    return parser.parse_args()


def resolve_use_embeddings(args):
    if args.use_embeddings == "true":
        return True
    if args.use_embeddings == "false":
        return False
    return MODEL_REGISTRY[args.model_type]["default_use_embeddings"]


def build_model(args, device):
    registry = MODEL_REGISTRY[args.model_type]
    if registry["module"] in {
        "FRWKV_WRAPPER",
        "FRWKV_DELTA_WRAPPER",
        "FRWKV_DELTAV2_WRAPPER",
        "FRWKV_HYBRIDDELTA_WRAPPER",
        "FRWKV_HYBRIDSPLITDELTA_WRAPPER",
        "FRWKV_LOCALGLOBAL_WRAPPER",
        "FRWKV_BRANCHGATE_WRAPPER",
        "FRWKV_CROSSBRANCHGATE_WRAPPER",
        "FRWKV_CROSSBRANCHPHASEGATE_WRAPPER",
        "FRWKV_CROSSBRANCHPHASEGATEFULLCONTEXTDELTA_WRAPPER",
        "FRWKV_CROSSBRANCHPHASEGATEADAPTIVE_WRAPPER",
        "FRWKV_CROSSBRANCHPHASEGATEADAPTIVECHANNELEMB_WRAPPER",
        "FRWKV_CROSSBRANCHPHASEGATEADAPTIVELINEARPROJ_WRAPPER",
        "FRWKV_CROSSBRANCHPHASEGATEADAPTIVEPATCHFOLDEMB_WRAPPER",
        "FRWKV_CROSSBRANCHPHASEGATEUSEFULNESS_WRAPPER",
        "FRWKV_CROSSBRANCHPHASEGATEMULTISCALE_WRAPPER",
        "FRWKV_CROSSBRANCHGATEV2_WRAPPER",
        "FRWKV_CROSSBRANCHGATEV15_WRAPPER",
        "FRWKV_HYBRIDSPLITDELTAGATE_WRAPPER",
    }:
        frwkv_root = str((Path(__file__).resolve().parent / "FRWKV").resolve())
        if frwkv_root not in sys.path:
            sys.path.insert(0, frwkv_root)
        for mod_name in [
            "layers",
            "layers.Transformer_EncDec",
            "layers.RevIN",
            "utils",
            "utils.tools",
            "utils.metrics",
            "model",
            "model.FRWKV",
            "data_provider",
            "experiments",
        ]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        if registry["module"] == "FRWKV_WRAPPER":
            target_module = "model.FRWKV"
        elif registry["module"] == "FRWKV_DELTA_WRAPPER":
            target_module = "model.FRWKV_Delta"
        elif registry["module"] == "FRWKV_DELTAV2_WRAPPER":
            target_module = "model.FRWKV_DeltaV2"
        elif registry["module"] == "FRWKV_HYBRIDDELTA_WRAPPER":
            target_module = "model.FRWKV_HybridDelta"
        elif registry["module"] == "FRWKV_HYBRIDSPLITDELTA_WRAPPER":
            target_module = "model.FRWKV_HybridSplitDelta"
        elif registry["module"] == "FRWKV_LOCALGLOBAL_WRAPPER":
            target_module = "model.FRWKV_LocalGlobal"
        elif registry["module"] == "FRWKV_BRANCHGATE_WRAPPER":
            target_module = "model.FRWKV_BranchGate"
        elif registry["module"] == "FRWKV_CROSSBRANCHGATE_WRAPPER":
            target_module = "model.FRWKV_CrossBranchGate"
        elif registry["module"] == "FRWKV_CROSSBRANCHPHASEGATE_WRAPPER":
            target_module = "model.FRWKV_CrossBranchPhaseGate"
        elif registry["module"] == "FRWKV_CROSSBRANCHPHASEGATEFULLCONTEXTDELTA_WRAPPER":
            target_module = "model.FRWKV_CrossBranchPhaseGateFullContextDelta"
        elif registry["module"] == "FRWKV_CROSSBRANCHPHASEGATEADAPTIVE_WRAPPER":
            target_module = "model.FRWKV_CrossBranchPhaseGateAdaptive"
        elif registry["module"] == "FRWKV_CROSSBRANCHPHASEGATEADAPTIVECHANNELEMB_WRAPPER":
            target_module = "model.FRWKV_CrossBranchPhaseGateAdaptive_ChannelEmb"
        elif registry["module"] == "FRWKV_CROSSBRANCHPHASEGATEADAPTIVELINEARPROJ_WRAPPER":
            target_module = "model.FRWKV_CrossBranchPhaseGateAdaptive_LinearProj"
        elif registry["module"] == "FRWKV_CROSSBRANCHPHASEGATEADAPTIVEPATCHFOLDEMB_WRAPPER":
            target_module = "model.FRWKV_CrossBranchPhaseGateAdaptive_PatchFoldEmb"
        elif registry["module"] == "FRWKV_CROSSBRANCHPHASEGATEUSEFULNESS_WRAPPER":
            target_module = "model.FRWKV_CrossBranchPhaseGateUsefulness"
        elif registry["module"] == "FRWKV_CROSSBRANCHPHASEGATEMULTISCALE_WRAPPER":
            target_module = "model.FRWKV_CrossBranchPhaseGateMultiScale"
        elif registry["module"] == "FRWKV_CROSSBRANCHGATEV2_WRAPPER":
            target_module = "model.FRWKV_CrossBranchGateV2"
        elif registry["module"] == "FRWKV_CROSSBRANCHGATEV15_WRAPPER":
            target_module = "model.FRWKV_CrossBranchGateV15"
        else:
            target_module = "model.FRWKV_HybridSplitDeltaGate"
        module = import_module(target_module)
        model_cls = getattr(module, registry["class"])
        period_position_len_arg = getattr(args, "period_position_len", None)
        period_position_list_arg = getattr(args, "period_position_list", None)
        period_position_num_routers_arg = getattr(args, "period_position_num_routers", None)
        period_context_alpha_init_arg = getattr(args, "period_context_alpha_init", None)
        period_context_trust_bias_init_arg = getattr(args, "period_context_trust_bias_init", None)
        period_position_len = (
            period_position_len_arg
            if period_position_len_arg is not None
            else getattr(args, "phase_period_len", 24)
        )
        period_position_list = (
            period_position_list_arg
            if period_position_list_arg is not None
            else getattr(args, "phase_period_list", "12,24,48")
        )
        period_position_num_routers = (
            period_position_num_routers_arg
            if period_position_num_routers_arg is not None
            else getattr(args, "phase_num_routers", 4)
        )
        period_context_alpha_init = (
            period_context_alpha_init_arg
            if period_context_alpha_init_arg is not None
            else getattr(args, "phase_alpha_init", 0.10)
        )
        period_context_trust_bias_init = (
            period_context_trust_bias_init_arg
            if period_context_trust_bias_init_arg is not None
            else getattr(args, "phase_trust_bias_init", -2.0)
        )
        configs = SimpleNamespace(
            pred_len=args.pred_len,
            enc_in=args.num_nodes,
            seq_len=args.seq_len,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            temp_patch_len=args.temp_patch_len,
            temp_stride=args.temp_stride,
            embed_size=args.embed_size,
            encoder_attention_type=args.encoder_attention_type,
            period_position_len=period_position_len,
            period_position_list=period_position_list,
            period_position_num_routers=period_position_num_routers,
            period_context_alpha_init=period_context_alpha_init,
            period_context_trust_bias_init=period_context_trust_bias_init,
            phase_period_len=period_position_len,
            phase_period_list=period_position_list,
            phase_num_routers=period_position_num_routers,
            phase_alpha_init=period_context_alpha_init,
            phase_trust_bias_init=period_context_trust_bias_init,
            use_revin=args.use_revin,
            revin_affine=args.revin_affine,
            dropout=args.dropout,
            e_layers=args.e_layers,
            activation=args.activation,
            CKA_flag=args.CKA_flag,
        )
        model = model_cls(configs)
        if hasattr(model, "to"):
            model = model.to(device)
        return model
    raise NotImplementedError(
        f"Unsupported model_type {args.model_type} in the open-source release package."
    )


def unpack_batch(batch, device, use_embeddings):
    if use_embeddings:
        x, y, x_mark, y_mark, embeddings = batch
        embeddings = move_to_device(embeddings, device)
    else:
        x, y, x_mark, y_mark = batch
        embeddings = None
    x = move_to_device(x, device)
    y = move_to_device(y, device)
    x_mark = move_to_device(x_mark, device)
    return x, y, x_mark, y_mark, embeddings

class trainer:
    def __init__(
        self,
        model,
        lrate,
        wdecay,
        epochs,
        loss_mode="L2",
        lossfun_alpha=0.0,
        loss_huber_delta=1.0,
        loss_l1_weight=1.0,
        loss_l2_weight=1.0,
        loss_huber_weight=1.0,
        loss_smooth_l1_beta=0.2,
        loss_corr_weight=0.0,
        mixed_precision="off",
    ):
        self.model = model
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 50), eta_min=1e-6)
        self.loss = WeightedSequenceLoss(
            alpha=lossfun_alpha,
            loss_mode=loss_mode,
            huber_delta=loss_huber_delta,
            l1_weight=loss_l1_weight,
            l2_weight=loss_l2_weight,
            huber_weight=loss_huber_weight,
        )
        self.loss.smooth_l1_beta = loss_smooth_l1_beta
        self.loss.corr_weight = loss_corr_weight
        self.MAE = MAE
        self.clip = 5
        self.mixed_precision = mixed_precision
        self.amp_dtype = None
        self.use_amp = False
        if torch.cuda.is_available():
            if mixed_precision == "fp16":
                self.use_amp = True
                self.amp_dtype = torch.float16
            elif mixed_precision == "bf16":
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        if hasattr(self.model, "count_trainable_params"):
            print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        if hasattr(self.model, "param_num"):
            print("The number of parameters: {}".format(self.model.param_num()))

    def _autocast_context(self):
        if self.use_amp and self.amp_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def train(self, input, mark, embeddings, real):
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            predict = self.model(input, mark, embeddings)
        predict_fp32 = predict.float()
        real_fp32 = real.float()
        loss = self.loss(predict_fp32, real_fp32)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            if self.clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
        with torch.no_grad():
            mae = self.MAE(predict_fp32, real_fp32)
        return loss.detach(), mae.detach()
    
    def eval(self, input, mark, embeddings, real_val):
        with torch.no_grad():
            predict = self.model(input,mark, embeddings)
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.detach(), mae.detach()

def load_data(args):
    use_embeddings = resolve_use_embeddings(args)
    if use_embeddings:
        data_map = {
            'ETTh1': Dataset_ETT_hour_Emb,
            'ETTh2': Dataset_ETT_hour_Emb,
            'ETTm1': Dataset_ETT_minute_Emb,
            'ETTm2': Dataset_ETT_minute_Emb
        }
        data_class = data_map.get(args.data_path, Dataset_Custom_Emb)
    else:
        data_map = {
            'ETTh1': Dataset_ETT_hour_Plain,
            'ETTh2': Dataset_ETT_hour_Plain,
            'ETTm1': Dataset_ETT_minute_Plain,
            'ETTm2': Dataset_ETT_minute_Plain
        }
        data_class = data_map.get(args.data_path, Dataset_Custom_Plain)

    common_dataset_kwargs = dict(
        flag='train',
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
    )
    if data_class in {Dataset_Custom_Emb, Dataset_Custom_Plain}:
        common_dataset_kwargs["percent"] = args.train_percent
    if use_embeddings:
        common_dataset_kwargs["embed_version"] = args.embed_version
    train_set = data_class(**common_dataset_kwargs)

    common_dataset_kwargs["flag"] = "val"
    val_set = data_class(**common_dataset_kwargs)
    common_dataset_kwargs["flag"] = "test"
    test_set = data_class(**common_dataset_kwargs)

    scaler = train_set.scaler

    loader_kwargs = get_dataloader_kwargs(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **loader_kwargs)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler


def load_data_distributed(args, rank, world_size):
    use_embeddings = resolve_use_embeddings(args)
    if use_embeddings:
        data_map = {
            'ETTh1': Dataset_ETT_hour_Emb,
            'ETTh2': Dataset_ETT_hour_Emb,
            'ETTm1': Dataset_ETT_minute_Emb,
            'ETTm2': Dataset_ETT_minute_Emb
        }
        data_class = data_map.get(args.data_path, Dataset_Custom_Emb)
    else:
        data_map = {
            'ETTh1': Dataset_ETT_hour_Plain,
            'ETTh2': Dataset_ETT_hour_Plain,
            'ETTm1': Dataset_ETT_minute_Plain,
            'ETTm2': Dataset_ETT_minute_Plain
        }
        data_class = data_map.get(args.data_path, Dataset_Custom_Plain)

    common_dataset_kwargs = dict(
        flag='train',
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
    )
    if data_class in {Dataset_Custom_Emb, Dataset_Custom_Plain}:
        common_dataset_kwargs["percent"] = args.train_percent
    if use_embeddings:
        common_dataset_kwargs["embed_version"] = args.embed_version
    train_set = data_class(**common_dataset_kwargs)
    scaler = train_set.scaler
    local_batch_size, global_batch_size = resolve_local_train_batch_size(args, world_size)

    if args.ddp_batch_size_mode == "global":
        train_sampler = GlobalBatchDistributedSampler(
            train_set,
            global_batch_size=global_batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
    else:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    loader_kwargs = get_dataloader_kwargs(args)

    train_loader = DataLoader(
        train_set,
        batch_size=local_batch_size,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )
    if rank == 0:
        common_dataset_kwargs["flag"] = "val"
        val_set = data_class(**common_dataset_kwargs)
        common_dataset_kwargs["flag"] = "test"
        test_set = data_class(**common_dataset_kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **loader_kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **loader_kwargs)
    else:
        val_set = None
        test_set = None
        val_loader = None
        test_loader = None
    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler, train_sampler


def infer_d_llm_from_embeddings(embed_dir: str):
    """
    根据 Embeddings 目录中的 H5 文件自动推断 d_llm 维度。
    约定：H5 中 key 为 'embeddings'，形状为 (d_llm, num_nodes) 或 (batch, d_llm, num_nodes)。
    对于 2D: shape[0] 是 d_llm，shape[1] 是 num_nodes
    对于 3D: shape[1] 是 d_llm，shape[2] 是 num_nodes
    """
    if not os.path.exists(embed_dir):
        return None
    try:
        files = sorted(
            f for f in os.listdir(embed_dir) if f.endswith(".h5")
        )
    except FileNotFoundError:
        return None
    for fname in files:
        fpath = os.path.join(embed_dir, fname)
        try:
            with h5py.File(fpath, "r") as hf:
                data = hf["embeddings"]
                if data.ndim == 2:
                    # 形状为 (d_llm, num_nodes)
                    return int(data.shape[0])
                elif data.ndim == 3:
                    # 形状为 (batch, d_llm, num_nodes)
                    return int(data.shape[1])
                elif data.ndim >= 2:
                    # 其他情况，尝试第一维
                    return int(data.shape[0])
        except Exception:
            continue
    return None

def seed_it(seed, deterministic=True):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def resolve_mixed_precision(args):
    if args.mixed_precision != "auto":
        return args.mixed_precision
    if args.speed_mode == "aggressive" and torch.cuda.is_available():
        return "fp16"
    return "off"


def get_runtime_execution_config(args):
    aggressive = args.speed_mode == "aggressive"
    deterministic = resolve_bool_mode(args.deterministic_train, not aggressive)
    allow_tf32 = resolve_bool_mode(args.allow_tf32, aggressive)
    cudnn_benchmark = resolve_bool_mode(args.cudnn_benchmark, aggressive and not deterministic)
    mixed_precision = resolve_mixed_precision(args)
    return {
        "deterministic": deterministic,
        "allow_tf32": allow_tf32,
        "cudnn_benchmark": cudnn_benchmark,
        "mixed_precision": mixed_precision,
    }


def apply_runtime_execution_config(runtime_cfg, main_process):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = runtime_cfg["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = runtime_cfg["allow_tf32"]
        torch.backends.cudnn.benchmark = runtime_cfg["cudnn_benchmark"]
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if runtime_cfg["allow_tf32"] else "highest")
    if main_process:
        print(
            "[Info] Runtime config: "
            f"deterministic={runtime_cfg['deterministic']}, "
            f"allow_tf32={runtime_cfg['allow_tf32']}, "
            f"cudnn_benchmark={runtime_cfg['cudnn_benchmark']}, "
            f"mixed_precision={runtime_cfg['mixed_precision']}"
        )


def maybe_enable_torch_compile(model, args, main_process=True):
    if not getattr(args, "torch_compile", False):
        return model

    if not hasattr(torch, "compile"):
        if main_process:
            print(
                f"[Info] torch.compile is unavailable in torch {torch.__version__}; "
                "continuing in eager mode."
            )
        return model

    compile_kwargs = {
        "backend": getattr(args, "torch_compile_backend", "inductor"),
        "mode": getattr(args, "torch_compile_mode", "default"),
        "fullgraph": getattr(args, "torch_compile_fullgraph", False),
    }

    try:
        compiled_model = torch.compile(model, **compile_kwargs)
        if main_process:
            print(f"[Info] Enabled torch.compile with {compile_kwargs}")
        return compiled_model
    except Exception as exc:
        if main_process:
            print(f"[Warn] torch.compile failed, falling back to eager mode: {exc}")
        return model


def setup_distributed():
    if not torch.cuda.is_available():
        return False, 0, 1, torch.device("cpu")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, torch.device("cuda")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(backend="nccl", device_id=local_rank)
    except TypeError:
        dist.init_process_group(backend="nccl")
    return True, rank, world_size, torch.device(f"cuda:{local_rank}")


def is_main_process(rank):
    return rank == 0


def get_ddp_runtime_config(args, distributed):
    is_frwkv_mainline = args.model_type.startswith("frwkv")
    sync_batchnorm = resolve_bool_mode(args.sync_batchnorm, distributed and is_frwkv_mainline)
    find_unused_parameters = resolve_bool_mode(args.ddp_find_unused_parameters, False)
    static_graph = resolve_bool_mode(args.ddp_static_graph, distributed and is_frwkv_mainline)
    gradient_as_bucket_view = resolve_bool_mode(args.ddp_gradient_as_bucket_view, distributed)

    if find_unused_parameters and static_graph:
        static_graph = False

    return {
        "sync_batchnorm": sync_batchnorm,
        "find_unused_parameters": find_unused_parameters,
        "static_graph": static_graph,
        "gradient_as_bucket_view": gradient_as_bucket_view,
        "bucket_cap_mb": args.ddp_bucket_cap_mb,
    }

def main():
    args = parse_args()
    args.use_embeddings = resolve_use_embeddings(args)
    distributed, rank, world_size, device = setup_distributed()
    ddp_runtime = get_ddp_runtime_config(args, distributed)
    runtime_cfg = get_runtime_execution_config(args)
    main_process = is_main_process(rank)
    apply_runtime_execution_config(runtime_cfg, main_process)
    train_sampler = None
    if distributed:
        train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler, train_sampler = load_data_distributed(args, rank, world_size)
    else:
        train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)

    # 如果存在预生成的 Embeddings，则尝试自动推断 d_llm 维度（支持 GPT2 / Qwen3 等）
    if args.use_embeddings and hasattr(train_set, "embed_path"):
        inferred_dim = infer_d_llm_from_embeddings(train_set.embed_path)
        if inferred_dim is not None and inferred_dim != args.d_llm:
            if main_process:
                print(
                f"[Info] Detected embedding dimension {inferred_dim} from {train_set.embed_path}. "
                f"Overriding d_llm (was {args.d_llm})."
            )
            args.d_llm = inferred_dim

    if main_process:
        print()
    seed_it(args.seed + rank if distributed else args.seed, deterministic=runtime_cfg["deterministic"])
    dp_device_ids = []
    if (not distributed) and torch.cuda.is_available() and args.gpu_devices:
        dp_device_ids = [int(x.strip()) for x in args.gpu_devices.split(",") if x.strip()]
    
    loss = 9999999
    test_log = 999999
    epochs_since_best_mse = 0
    best_model_state = None  # 用于保存最佳模型状态（不保存文件时使用）

    run_name_parts = [
        args.model_type,
        args.model_tag or "default",
        f"sl{args.seq_len}",
        f"pl{args.pred_len}",
        f"lr{args.learning_rate}",
        f"seed{args.seed}",
        f"loss{args.loss_mode}",
    ]
    path = os.path.join(args.save, args.data_path, "_".join(run_name_parts))
    if main_process and not os.path.exists(path):
        os.makedirs(path)
    if distributed:
        dist.barrier()
     
    his_loss = []
    val_time = []
    train_time = []
    if main_process:
        print(args)
        if distributed:
            local_batch_size, global_batch_size = resolve_local_train_batch_size(args, world_size)
            print(
                f"[Info] DDP batch mode={args.ddp_batch_size_mode}, "
                f"local_batch_size={local_batch_size}, global_batch_size={global_batch_size}"
            )

    model = build_model(args, device)
    compile_allowed = True
    compile_skip_reason = None
    if distributed:
        compile_allowed = False
        compile_skip_reason = "DistributedDataParallel path"
    elif torch.cuda.is_available() and len(dp_device_ids) > 1:
        compile_allowed = False
        compile_skip_reason = "DataParallel path"

    if args.torch_compile and not compile_allowed and main_process:
        print(f"[Info] Skipping torch.compile on {compile_skip_reason}.")
    if compile_allowed:
        model = maybe_enable_torch_compile(model, args, main_process=main_process)

    if distributed:
        if ddp_runtime["sync_batchnorm"]:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if main_process:
                print("[Info] Enabled SyncBatchNorm for distributed training.")
        local_rank = int(os.environ["LOCAL_RANK"])
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_runtime["find_unused_parameters"],
            static_graph=ddp_runtime["static_graph"],
            gradient_as_bucket_view=ddp_runtime["gradient_as_bucket_view"],
            bucket_cap_mb=ddp_runtime["bucket_cap_mb"],
        )
        if main_process:
            print(
                f"[Info] Enabled DDP on rank {rank}/{world_size} "
                f"(find_unused_parameters={ddp_runtime['find_unused_parameters']}, "
                f"static_graph={ddp_runtime['static_graph']}, "
                f"gradient_as_bucket_view={ddp_runtime['gradient_as_bucket_view']}, "
                f"bucket_cap_mb={ddp_runtime['bucket_cap_mb']})"
            )
    elif torch.cuda.is_available() and len(dp_device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=dp_device_ids, output_device=dp_device_ids[0])
        if main_process:
            print(f"[Info] Enabled DataParallel on local device ids: {dp_device_ids}")
    engine = trainer(
        model=model,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        epochs=args.epochs,
        loss_mode=args.loss_mode,
        lossfun_alpha=args.lossfun_alpha,
        loss_huber_delta=args.loss_huber_delta,
        loss_l1_weight=args.loss_l1_weight,
        loss_l2_weight=args.loss_l2_weight,
        loss_huber_weight=args.loss_huber_weight,
        loss_smooth_l1_beta=args.loss_smooth_l1_beta,
        loss_corr_weight=args.loss_corr_weight,
        mixed_precision=runtime_cfg["mixed_precision"],
    )

    if main_process:
        print("Start training...", flush=True)

    for i in range(1, args.epochs + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(i)

        t1 = time.time()
        train_loss_sum = torch.zeros((), device=device)
        train_mae_sum = torch.zeros((), device=device)
        train_steps = 0
        engine.model.train()
        
        for iter, batch in enumerate(train_loader):
            trainx, trainy, trainx_mark, _, train_embedding = unpack_batch(batch, device, args.use_embeddings)
            metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)
            train_loss_sum = train_loss_sum + metrics[0]
            train_mae_sum = train_mae_sum + metrics[1]
            train_steps += 1

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        if main_process:
            print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # validation
        val_loss_sum = None
        val_mae_sum = None
        val_steps = 0
        s1 = time.time()

        if main_process and val_loader is not None:
            engine.model.eval()
            val_loss_sum = torch.zeros((), device=device)
            val_mae_sum = torch.zeros((), device=device)
            for iter, batch in enumerate(val_loader):
                valx, valy, valx_mark, _, val_embedding = unpack_batch(batch, device, args.use_embeddings)
                metrics = engine.eval(valx, valx_mark, val_embedding, valy)
                val_loss_sum = val_loss_sum + metrics[0]
                val_mae_sum = val_mae_sum + metrics[1]
                val_steps += 1

        s2 = time.time()
        log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
        if main_process:
            print(log.format(i, (s2 - s1)))
            val_time.append(s2 - s1)

        if distributed:
            train_count = torch.tensor(float(train_steps), device=device)
            dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_mae_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
            mtrain_loss = (train_loss_sum / train_count).item()
            mtrain_mae = (train_mae_sum / train_count).item()
        else:
            mtrain_loss = (train_loss_sum / max(train_steps, 1)).item()
            mtrain_mae = (train_mae_sum / max(train_steps, 1)).item()

        if main_process and val_steps > 0:
            mvalid_loss = (val_loss_sum / val_steps).item()
            mvalid_mae = (val_mae_sum / val_steps).item()
        else:
            mvalid_loss = None
            mvalid_mae = None

        if main_process:
            his_loss.append(mvalid_loss)
            print("-----------------------")

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} "
        if main_process:
            print(
                log.format(i, mtrain_loss, mtrain_mae),
                flush=True,
            )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}"
        if main_process:
            print(
                log.format(i, mvalid_loss, mvalid_mae),
                flush=True,
            )

        if main_process and mvalid_loss < loss:
            print("###Update tasks appear###")
            if i <= 10:
                
                loss = mvalid_loss
                best_model_state = copy.deepcopy(engine.model.state_dict())  # 保存最佳模型状态到内存
                if args.save_model:
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mse = 0
                print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
                print("epoch: ", i)
            else:
                test_outputs = []
                test_y = []

                for iter, batch in enumerate(test_loader):
                    testx, testy, testx_mark, _, test_embedding = unpack_batch(batch, device, args.use_embeddings)
                    with torch.no_grad():
                        preds = engine.model(testx, testx_mark, test_embedding)
                    test_outputs.append(preds)
                    test_y.append(testy)
                
                test_pre = torch.cat(test_outputs, dim=0)
                test_real = torch.cat(test_y, dim=0)

                amse = []
                amae = []
                
                for j in range(args.pred_len):
                    pred = test_pre[:, j,].to(device)
                    real = test_real[:, j, ].to(device)
                    metrics = metric(pred, real)
                    log = "Evaluate best model on test data for horizon {:d}, Test MSE: {:.4f}, Test MAE: {:.4f}"
                    amse.append(metrics[0])
                    amae.append(metrics[1])

                log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
                print(
                    log.format(
                        np.mean(amse), np.mean(amae)
                    )
                )

                if np.mean(amse) < test_log:
                    test_log = np.mean(amse)
                    loss = mvalid_loss
                    best_model_state = copy.deepcopy(engine.model.state_dict())  # 保存最佳模型状态到内存
                    if args.save_model:
                        torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mse = 0
                    print("Test low! Updating! Test Loss: {:.4f}".format(np.mean(amse)), end=", ")
                    print("Test low! Updating! Valid Loss: {:.4f}".format(mvalid_loss), end=", ")

                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mse += 1
                    print("No update")

        elif main_process:
            epochs_since_best_mse += 1
            print("No update")

        engine.scheduler.step()

        if main_process and epochs_since_best_mse >= args.es_patience and i >= args.epochs//2: # early stop
            break
        if distributed:
            stop_tensor = torch.tensor(
                1 if (main_process and epochs_since_best_mse >= args.es_patience and i >= args.epochs//2) else 0,
                device=device,
            )
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                break

    # Output consumption
    if main_process:
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))

    # Test
    if not main_process:
        if distributed:
            dist.destroy_process_group()
        return

    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))
   
    # 加载最佳模型状态（从内存或文件）
    if args.save_model and os.path.exists(path + "best_model.pth"):
        engine.model.load_state_dict(torch.load(path + "best_model.pth"))
    elif 'best_model_state' in locals():
        engine.model.load_state_dict(best_model_state)
    
    test_outputs = []
    test_y = []

    for iter, batch in enumerate(test_loader):
        testx, testy, testx_mark, _, test_embedding = unpack_batch(batch, device, args.use_embeddings)
        with torch.no_grad():
            preds = engine.model(testx, testx_mark, test_embedding)
        test_outputs.append(preds)
        test_y.append(testy)

    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.cat(test_y, dim=0)

    amse = []
    amae = []
    
    for j in range(args.pred_len):
        pred = test_pre[:, j,].to(device)
        real = test_real[:, j, ].to(device)
        metrics = metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MSE: {:.4f}, Test MAE: {:.4f}"
        amse.append(metrics[0])
        amae.append(metrics[1])

    log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
    final_test_mse = np.mean(amse)
    final_test_mae = np.mean(amae)
    print(log.format(final_test_mse, final_test_mae))
    
    # 记录实验结果到统一日志文件
    log_experiment_result(
        data_path=args.data_path,
        pred_len=args.pred_len,
        model_name=args.model_type,
        seed=args.seed,
        test_mse=final_test_mse,
        test_mae=final_test_mae,
        embed_version=args.embed_version if args.use_embeddings else "none",
        seq_len=args.seq_len,
        channel=args.channel,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_n=args.dropout_n,
        additional_info={
            "use_embeddings": args.use_embeddings,
            "config_name": args.config_name,
            "config_file": args.config_file,
            "loss_mode": args.loss_mode,
            "lossfun_alpha": args.lossfun_alpha,
            "loss_huber_delta": args.loss_huber_delta,
            "loss_l1_weight": args.loss_l1_weight,
            "loss_l2_weight": args.loss_l2_weight,
            "loss_huber_weight": args.loss_huber_weight,
            "loss_smooth_l1_beta": args.loss_smooth_l1_beta,
            "loss_corr_weight": args.loss_corr_weight,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "n_heads": args.n_heads,
            "e_layers": args.e_layers,
            "embed_size": args.embed_size,
            "encoder_attention_type": args.encoder_attention_type,
            "temp_patch_len": args.temp_patch_len,
            "temp_stride": args.temp_stride,
            "period_position_len": args.period_position_len,
            "period_position_list": args.period_position_list,
            "period_position_num_routers": args.period_position_num_routers,
            "period_context_alpha_init": args.period_context_alpha_init,
            "period_context_trust_bias_init": args.period_context_trust_bias_init,
            "use_revin": args.use_revin,
            "revin_affine": args.revin_affine,
            "model_tag": args.model_tag,
        }
    )
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
