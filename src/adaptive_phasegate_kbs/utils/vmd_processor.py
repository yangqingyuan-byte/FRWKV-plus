import os
import numpy as np

try:
    from vmdpy import VMD
except ImportError:
    VMD = None


def compute_vmd(signal: np.ndarray, K: int = 4, alpha: float = 2000, tau: float = 0.0,
                DC: int = 0, init: int = 1, tol: float = 1e-7):
    """
    计算单通道信号的 VMD 分解。

    Args:
        signal: 1D numpy array，形状 [L]
        K: 模态数
        alpha: 带宽约束
        tau: Lagrange multiplier of the data-fidelity constraint
        DC: 1 表示包含 DC 模态
        init: 初始化方式，1 为随机
        tol: 收敛阈值
    Returns:
        imfs: [K, L] 的分解结果
    """
    if VMD is None:
        raise ImportError("vmdpy 未安装，无法执行 VMD 分解。请先安装: pip install vmdpy")
    if signal.ndim != 1:
        raise ValueError("signal 必须是一维向量")

    imfs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    return imfs


def compute_vmd_with_cache(signal: np.ndarray, cache_path: str, **vmd_kwargs):
    """
    带磁盘缓存的 VMD 计算。若缓存存在则直接读取，否则计算后保存。

    Args:
        signal: 1D numpy array，形状 [L]
        cache_path: 缓存文件路径（.npy）
        vmd_kwargs: 传递给 compute_vmd 的参数
    Returns:
        imfs: [K, L] 的 numpy array
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    imfs = compute_vmd(signal, **vmd_kwargs)
    np.save(cache_path, imfs)
    return imfs


__all__ = ["compute_vmd", "compute_vmd_with_cache"]

