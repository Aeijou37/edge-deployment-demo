"""
推理对比模块 — PyTorch vs ONNX FP32 vs ONNX INT8

职责：
1. 三种推理方式的精度对比
2. 三种推理方式的速度对比（延迟 + 吞吐量）
3. 生成对比图表
"""
import torch
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def benchmark_all(
    torch_model,
    fp32_onnx_path: str,
    int8_onnx_path: str,
    num_points: int = 1024,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict:
    """对比三种推理方式的速度

    Args:
        torch_model: PyTorch 模型
        fp32_onnx_path: FP32 ONNX 模型路径
        int8_onnx_path: INT8 ONNX 模型路径
        num_warmup: 预热次数（不计时）
        num_runs: 正式测试次数

    Returns:
        包含三种推理方式速度数据的字典
    """
    print(f"\n推理速度基准测试 (warmup={num_warmup}, runs={num_runs})")
    print("=" * 60)

    fp32_session = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(int8_onnx_path, providers=["CPUExecutionProvider"])
    input_name = fp32_session.get_inputs()[0].name

    torch_model.eval()

    results = {"torch": [], "onnx_fp32": [], "onnx_int8": []}

    for i in range(num_warmup + num_runs):
        dummy_input = torch.randn(1, num_points, 3)
        np_input = dummy_input.numpy()

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch_model(dummy_input)
        if i >= num_warmup:
            results["torch"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = fp32_session.run(["logits"], {input_name: np_input})
        if i >= num_warmup:
            results["onnx_fp32"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = int8_session.run(["logits"], {input_name: np_input})
        if i >= num_warmup:
            results["onnx_int8"].append((time.perf_counter() - t0) * 1000)

    summary = {}
    for name, times in results.items():
        avg = np.mean(times)
        std = np.std(times)
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        fps = 1000 / avg
        summary[name] = {
            "avg_ms": float(avg),
            "std_ms": float(std),
            "p50_ms": float(p50),
            "p95_ms": float(p95),
            "p99_ms": float(p99),
            "fps": float(fps),
            "times": times,
        }
        print(f"  {name:12s} | avg={avg:.2f}ms | p50={p50:.2f}ms | p95={p95:.2f}ms | FPS={fps:.1f}")

    torch_vs_int8 = summary["torch"]["avg_ms"] / summary["onnx_int8"]["avg_ms"]
    fp32_vs_int8 = summary["onnx_fp32"]["avg_ms"] / summary["onnx_int8"]["avg_ms"]
    print(f"\n  PyTorch → INT8 加速比: {torch_vs_int8:.2f}x")
    print(f"  FP32 → INT8 加速比: {fp32_vs_int8:.2f}x")

    summary["speedup_torch_vs_int8"] = float(torch_vs_int8)
    summary["speedup_fp32_vs_int8"] = float(fp32_vs_int8)
    return summary


def visualize_benchmark(benchmark_results: Dict, output_path: str = "output_benchmark.png") -> str:
    """可视化推理速度对比"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = ["PyTorch", "ONNX FP32", "ONNX INT8"]
    keys = ["torch", "onnx_fp32", "onnx_int8"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    avgs = [benchmark_results[k]["avg_ms"] for k in keys]
    bars = axes[0].bar(names, avgs, color=colors)
    axes[0].set_ylabel("平均延迟 (ms)")
    axes[0].set_title("平均推理延迟")
    for bar, val in zip(bars, avgs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.2f}ms", ha="center", fontsize=10)

    fpses = [benchmark_results[k]["fps"] for k in keys]
    bars = axes[1].bar(names, fpses, color=colors)
    axes[1].set_ylabel("FPS")
    axes[1].set_title("吞吐量 (FPS)")
    for bar, val in zip(bars, fpses):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.0f}", ha="center", fontsize=10)

    data = [benchmark_results[k]["times"] for k in keys]
    bp = axes[2].boxplot(data, labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[2].set_ylabel("延迟 (ms)")
    axes[2].set_title("延迟分布")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n基准测试图表已保存: {output_path}")
    return output_path


def visualize_accuracy(
    torch_logits: np.ndarray,
    fp32_logits: np.ndarray,
    int8_logits: np.ndarray,
    output_path: str = "output_accuracy.png",
) -> str:
    """可视化三种推理方式的精度对比"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(range(len(torch_logits[0])), torch_logits[0], color="#2196F3", alpha=0.7)
    axes[0].set_title("PyTorch Logits")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Logit")

    axes[1].bar(range(len(fp32_logits[0])), fp32_logits[0], color="#FF9800", alpha=0.7)
    axes[1].set_title("ONNX FP32 Logits")

    axes[2].bar(range(len(int8_logits[0])), int8_logits[0], color="#4CAF50", alpha=0.7)
    axes[2].set_title("ONNX INT8 Logits")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"精度对比图表已保存: {output_path}")
    return output_path
