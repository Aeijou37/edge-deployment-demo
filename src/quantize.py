"""
量化模块 — INT8 动态量化

职责：
1. 对 ONNX FP32 模型做 INT8 动态量化
2. 量化前后文件大小对比
3. 量化前后精度对比
4. 量化前后推理速度对比

动态量化：只量化权重（int8），激活值保持 FP32。
优势：不需要校准数据集，速度快，适合 CPU 推理。
劣势：精度损失比静态量化略大，但对分类任务影响小。
"""
import onnx
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List
import time


def quantize_onnx_int8(
    fp32_onnx_path: str,
    quantized_onnx_path: str,
) -> str:
    """对 ONNX 模型做 INT8 动态量化

    Args:
        fp32_onnx_path: FP32 ONNX 模型路径
        quantized_onnx_path: 量化后 ONNX 模型路径

    Returns:
        量化后模型路径
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    Path(quantized_onnx_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"INT8 动态量化: {fp32_onnx_path}")
    quantize_dynamic(
        fp32_onnx_path,
        quantized_onnx_path,
        weight_type=QuantType.QUInt8,
    )

    fp32_size = Path(fp32_onnx_path).stat().st_size / 1024 / 1024
    int8_size = Path(quantized_onnx_path).stat().st_size / 1024 / 1024
    compression_ratio = fp32_size / int8_size

    print(f"  FP32 大小: {fp32_size:.2f} MB")
    print(f"  INT8 大小: {int8_size:.2f} MB")
    print(f"  压缩比: {compression_ratio:.2f}x")

    return quantized_onnx_path


def compare_models(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    num_points: int = 1024,
    num_samples: int = 100,
) -> Dict:
    """对比 FP32 和 INT8 模型的精度和速度

    Returns:
        包含精度差异和速度对比的字典
    """
    print(f"\n对比 FP32 vs INT8 ({num_samples} 样本)...")

    fp32_session = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(int8_onnx_path, providers=["CPUExecutionProvider"])

    input_name = fp32_session.get_inputs()[0].name

    max_diff = 0
    total_diff = 0
    pred_match = 0
    cosine_sims = []

    fp32_times = []
    int8_times = []

    for i in range(num_samples):
        dummy_input = np.random.randn(1, num_points, 3).astype(np.float32)

        t0 = time.perf_counter()
        fp32_logits = fp32_session.run(["logits"], {input_name: dummy_input})[0]
        fp32_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        int8_logits = int8_session.run(["logits"], {input_name: dummy_input})[0]
        int8_times.append(time.perf_counter() - t0)

        diff = np.abs(fp32_logits - int8_logits).max()
        max_diff = max(max_diff, diff)
        total_diff += diff

        fp32_pred = np.argmax(fp32_logits[0])
        int8_pred = np.argmax(int8_logits[0])
        if fp32_pred == int8_pred:
            pred_match += 1

        sim = np.dot(fp32_logits[0], int8_logits[0]) / (
            np.linalg.norm(fp32_logits[0]) * np.linalg.norm(int8_logits[0]) + 1e-8
        )
        cosine_sims.append(sim)

    avg_diff = total_diff / num_samples
    avg_sim = np.mean(cosine_sims)
    pred_match_rate = pred_match / num_samples

    fp32_avg_ms = np.mean(fp32_times) * 1000
    int8_avg_ms = np.mean(int8_times) * 1000
    speedup = fp32_avg_ms / int8_avg_ms

    print(f"\n  精度对比:")
    print(f"    最大绝对误差: {max_diff:.6f}")
    print(f"    平均绝对误差: {avg_diff:.6f}")
    print(f"    平均余弦相似度: {avg_sim:.6f}")
    print(f"    Top-1 预测一致率: {pred_match_rate:.2%}")

    print(f"\n  速度对比:")
    print(f"    FP32 平均: {fp32_avg_ms:.2f} ms")
    print(f"    INT8 平均: {int8_avg_ms:.2f} ms")
    print(f"    加速比: {speedup:.2f}x")

    return {
        "max_diff": float(max_diff),
        "avg_diff": float(avg_diff),
        "avg_cosine_sim": float(avg_sim),
        "pred_match_rate": float(pred_match_rate),
        "fp32_avg_ms": float(fp32_avg_ms),
        "int8_avg_ms": float(int8_avg_ms),
        "speedup": float(speedup),
        "fp32_times": [t * 1000 for t in fp32_times],
        "int8_times": [t * 1000 for t in int8_times],
    }
