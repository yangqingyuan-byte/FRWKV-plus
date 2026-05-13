"""
实验结果日志记录工具
统一记录所有实验的结果到 experiment_results.log
"""
import os
import json
from datetime import datetime


def log_experiment_result(
    data_path: str,
    pred_len: int,
    model_name: str,
    seed: int,
    test_mse: float,
    test_mae: float,
    embed_version: str = None,
    seq_len: int = None,
    channel: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    dropout_n: float = None,
    additional_info: dict = None
):
    """
    将实验结果追加到统一的日志文件
    
    Args:
        data_path: 数据集名称（如 ETTh1）
        pred_len: 预测长度
        model_name: 模型名称（如 T3Time, T3Time_Wavelet_Qwen）
        seed: 随机种子
        test_mse: 测试集 MSE
        test_mae: 测试集 MAE
        embed_version: 嵌入版本（如 original, qwen3_0.6b）
        seq_len: 输入序列长度
        channel: 通道数
        batch_size: 批次大小
        learning_rate: 学习率
        dropout_n: Dropout 率
        additional_info: 其他额外信息（字典格式）
    """
    log_file = "./experiment_results.log"
    
    # 构建结果字典
    result = {
        "data_path": data_path,
        "pred_len": pred_len,
        "test_mse": round(test_mse, 6),
        "test_mae": round(test_mae, 6),
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
    }
    
    # 添加可选字段
    if embed_version is not None:
        result["embed_version"] = embed_version
    if seq_len is not None:
        result["seq_len"] = seq_len
    if channel is not None:
        result["channel"] = channel
    if batch_size is not None:
        result["batch_size"] = batch_size
    if learning_rate is not None:
        result["learning_rate"] = learning_rate
    if dropout_n is not None:
        result["dropout_n"] = dropout_n
    if additional_info:
        result.update(additional_info)
    
    # 追加写入日志文件
    with open(log_file, "a", encoding="utf-8") as f:
        # 使用 JSON 格式，每行一个结果
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # 同时打印到控制台
    print(f"\n{'='*60}")
    print(f"实验结果已记录到: {log_file}")
    print(f"数据集: {data_path}, 预测长度: {pred_len}")
    print(f"Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")
    print(f"模型: {model_name}, 种子: {seed}")
    if channel: print(f"Channel: {channel}", end=", ")
    if embed_version: print(f"Embed: {embed_version}", end=", ")
    if learning_rate: print(f"LR: {learning_rate}")
    else: print()
    print(f"{'='*60}\n")

