"""
模型导出模块 — PyTorch → ONNX

职责：
1. 创建 PointNet 分类模型（和 pointcloud-demo 共享架构）
2. 导出为 ONNX 格式
3. 验证 ONNX 模型的数值精度（vs PyTorch）

这是边缘部署的第一步：把训练框架的模型转为跨平台标准格式。
"""
import torch
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, Dict


def create_pointnet_classifier(num_classes: int = 40, num_points: int = 1024):
    """创建 PointNet 分类模型"""
    from src.pointnet_model import PointNetClassifier
    model = PointNetClassifier(num_classes=num_classes, num_points=num_points)
    model.eval()
    return model


def export_to_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    num_points: int = 1024,
    num_classes: int = 40,
    opset_version: int = 17,
) -> str:
    """导出 PyTorch 模型为 ONNX 格式

    Args:
        model: PyTorch 模型
        onnx_path: 输出 ONNX 文件路径
        num_points: 点云点数
        opset_version: ONNX opset 版本

    Returns:
        ONNX 文件路径
    """
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, num_points, 3)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits", "global_feat"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "global_feat": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"ONNX 导出完成: {onnx_path}")
    print(f"  文件大小: {Path(onnx_path).stat().st_size / 1024 / 1024:.2f} MB")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX 模型验证通过")

    return onnx_path


def verify_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    num_points: int = 1024,
    num_samples: int = 10,
) -> Dict:
    """验证 ONNX 模型 vs PyTorch 模型的数值精度

    Returns:
        包含最大误差、平均误差的字典
    """
    print(f"\n验证 ONNX 数值精度 ({num_samples} 样本)...")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_diff = 0
    total_diff = 0
    cosine_sims = []

    for i in range(num_samples):
        dummy_input = torch.randn(1, num_points, 3)

        with torch.no_grad():
            torch_logits, _ = model(dummy_input)
        torch_logits = torch_logits.numpy()

        ort_logits = session.run(
            ["logits"],
            {input_name: dummy_input.numpy()},
        )[0]

        diff = np.abs(torch_logits - ort_logits).max()
        max_diff = max(max_diff, diff)
        total_diff += diff

        sim = np.dot(torch_logits[0], ort_logits[0]) / (
            np.linalg.norm(torch_logits[0]) * np.linalg.norm(ort_logits[0]) + 1e-8
        )
        cosine_sims.append(sim)

    avg_diff = total_diff / num_samples
    avg_sim = np.mean(cosine_sims)

    print(f"  最大绝对误差: {max_diff:.6f}")
    print(f"  平均绝对误差: {avg_diff:.6f}")
    print(f"  平均余弦相似度: {avg_sim:.6f}")

    if max_diff < 1e-3:
        print("  ✅ 数值精度验证通过（误差 < 1e-3）")
    else:
        print("  ⚠️ 数值误差偏大，检查导出配置")

    return {
        "max_diff": float(max_diff),
        "avg_diff": float(avg_diff),
        "avg_cosine_sim": float(avg_sim),
    }


if __name__ == "__main__":
    model = create_pointnet_classifier(num_classes=40, num_points=1024)
    onnx_path = export_to_onnx(model, "models/pointnet_fp32.onnx", num_points=1024)
    verify_onnx(model, onnx_path, num_points=1024)
