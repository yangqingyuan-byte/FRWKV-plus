import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.tools import StandardScaler
from utils.timefeatures import time_features
from utils.vmd_processor import compute_vmd


FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = str(FRAMEWORK_ROOT / "dataset")
DEFAULT_EMBED_ROOT = str(FRAMEWORK_ROOT / "Embeddings")


class Dataset_ETT_hour_VMD(Dataset):
    """
    带 VMD 模态的 ETT hour 数据集（ETTh1/ETTh2）。
    在不修改原 Dataset_ETT_hour 的前提下，新增 x_modes 输出：
        seq_x: [seq_len, N]
        seq_y: [pred_len, N]
        seq_x_mark, seq_y_mark
        embeddings: [E, N, 1]（沿用原 embedding loader 逻辑）
        x_modes: [seq_len, N, K]  VMD 分量
    """

    def __init__(
        self,
        root_path: str = DEFAULT_DATASET_ROOT,
        flag: str = "train",
        size: Optional[Tuple[int, int, int]] = None,
        features: str = "M",
        data_path: str = "ETTh1",
        num_nodes: int = 7,
        target: str = "OT",
        scale: bool = True,
        inverse: bool = False,
        timeenc: int = 0,
        freq: str = "h",
        model_name: str = "gpt2",
        embed_version: str = "original",
        vmd_k: int = 4,
        vmd_alpha: float = 2000.0,
        vmd_root: str = "./vmd_cache",
    ):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.freq = freq
        self.timeenc = timeenc
        self.num_nodes = num_nodes
        self.root_path = root_path
        self.data_path = data_path

        # 文件名（不带 .csv）
        if not data_path.endswith(".csv"):
            data_path_file = data_path
            data_path = data_path + ".csv"
        else:
            data_path_file = os.path.splitext(os.path.basename(data_path))[0]
        self.data_path = os.path.join(root_path, data_path)
        self.data_path_file = data_path_file

        self.model_name = model_name
        self.embed_version = embed_version
        # Embeddings 路径: ./Embeddings/{data_path}/{embed_version}/{flag}/
        self.embed_path = os.path.join(
            DEFAULT_EMBED_ROOT, data_path_file, embed_version, flag
        )

        # VMD 缓存根目录: ./vmd_cache/{data_path_file}/k{vmd_k}_a{vmd_alpha}/{flag}/
        self.vmd_k = vmd_k
        self.vmd_alpha = vmd_alpha
        self.vmd_root = os.path.join(
            vmd_root,
            data_path_file,
            f"k{vmd_k}_a{int(vmd_alpha)}",
            flag,
        )
        os.makedirs(self.vmd_root, exist_ok=True)

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features type: {self.features}")

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["year"] = df_stamp.date.apply(lambda row: row.year)
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        else:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def _load_embeddings(self, index: int) -> torch.Tensor:
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No embedding file found at {file_path}")
        import h5py

        embeddings_stack = []
        with h5py.File(file_path, "r") as hf:
            data = hf["embeddings"][:]
            tensor = torch.from_numpy(data)
            embeddings_stack.append(tensor.squeeze(0))
        embeddings = torch.stack(embeddings_stack, dim=-1)
        return embeddings

    def _load_or_compute_vmd(self, index: int, seq_x: np.ndarray) -> np.ndarray:
        """
        对当前窗口 seq_x 计算或加载 VMD 分量。
        seq_x: [seq_len, N]
        返回: [seq_len, N, K]
        """
        os.makedirs(self.vmd_root, exist_ok=True)
        cache_path = os.path.join(self.vmd_root, f"window_{index:06d}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

        L, N = seq_x.shape
        imfs_list = []
        for j in range(N):
            imfs = compute_vmd(
                seq_x[:, j],
                K=self.vmd_k,
                alpha=self.vmd_alpha,
            )  # [K, L]
            imfs_list.append(imfs)
        imfs_arr = np.stack(imfs_list, axis=0)  # [N, K, L]
        imfs_arr = np.transpose(imfs_arr, (2, 0, 1))  # [L, N, K]
        np.save(cache_path, imfs_arr.astype(np.float32))
        return imfs_arr

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  # [L, N]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        embeddings = self._load_embeddings(index)  # 与原 emb loader 一致
        x_modes = self._load_or_compute_vmd(index, seq_x)  # [L, N, K]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings, x_modes

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


__all__ = ["Dataset_ETT_hour_VMD"]
