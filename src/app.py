"""
Gradio 前端 — 边缘部署对比 Demo

功能：
1. 输入点云（预设形状或随机生成）
2. 三种推理方式同时推理：PyTorch / ONNX FP32 / ONNX INT8
3. 展示对比：
   - 分类结果一致性
   - 推理延迟对比
   - 模型大小对比
4. 一键导出 + 量化（自包含，不依赖 export_onnx.py）

运行:
  python src/app.py
"""
import sys
import time
import torch
import numpy as np
import gradio as gr
import onnx
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pointnet_model import PointNetClassifier
from src.quantize import quantize_onnx_int8, compare_models
from src.benchmark import benchmark_all, visualize_benchmark
from src.utils import generate_shape, SHAPES, MODELNET40_CLASSES


class EdgeDeployApp:
    def __init__(self, num_points: int = 1024):
        print("初始化边缘部署 Demo...")
        self.num_points = num_points
        self.torch_model = PointNetClassifier(num_classes=40, num_points=num_points)
        self.torch_model.eval()

        self.fp32_session = None
        self.int8_session = None
        self.fp32_path = "models/pointnet_fp32.onnx"
        self.int8_path = "models/pointnet_int8.onnx"

        self._try_load_onnx()
        print("初始化完成\n")

    def _try_load_onnx(self):
        """尝试加载已有的 ONNX 模型"""
        if Path(self.fp32_path).exists():
            try:
                self.fp32_session = ort.InferenceSession(self.fp32_path, providers=["CPUExecutionProvider"])
                size_mb = Path(self.fp32_path).stat().st_size / 1024 / 1024
                print(f"  已加载 FP32 ONNX: {self.fp32_path} ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"  ⚠️ FP32 ONNX 加载失败: {e}")
        else:
            print("  ⚠️ 无 FP32 ONNX，请先导出")

        if Path(self.int8_path).exists():
            try:
                self.int8_session = ort.InferenceSession(self.int8_path, providers=["CPUExecutionProvider"])
                size_mb = Path(self.int8_path).stat().st_size / 1024 / 1024
                print(f"  已加载 INT8 ONNX: {self.int8_path} ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"  ⚠️ INT8 ONNX 加载失败: {e}")
        else:
            print("  ⚠️ 无 INT8 ONNX，请先量化")

    def export_and_quantize(self):
        """一键导出 + 量化（自包含，直接写死 dynamo=False + opset=17）"""
        results = []

        try:
            Path("models").mkdir(exist_ok=True)
            dummy = torch.randn(1, self.num_points, 3)
            torch.onnx.export(
                self.torch_model,
                dummy,
                self.fp32_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits", "global_feat"],
                dynamic_axes={
                    "input": {0: "batch"},
                    "logits": {0: "batch"},
                    "global_feat": {0: "batch"},
                },
                dynamo=False,
            )
            fp32_size = Path(self.fp32_path).stat().st_size / 1024 / 1024
            results.append(f"✅ ONNX 导出完成 ({fp32_size:.2f} MB)")
            self.fp32_session = ort.InferenceSession(self.fp32_path, providers=["CPUExecutionProvider"])
        except Exception as e:
            results.append(f"❌ ONNX 导出失败: {e}")
            return "\n".join(results)

        try:
            onnx_model = onnx.load(self.fp32_path)
            onnx.checker.check_model(onnx_model)
            results.append("✅ ONNX 模型验证通过")
        except Exception as e:
            results.append(f"❌ ONNX 验证失败: {e}")

        try:
            quantize_onnx_int8(self.fp32_path, self.int8_path)
            results.append("✅ INT8 量化完成")
            self.int8_session = ort.InferenceSession(self.int8_path, providers=["CPUExecutionProvider"])
        except Exception as e:
            results.append(f"❌ 量化失败: {e}")

        if Path(self.fp32_path).exists() and Path(self.int8_path).exists():
            fp32_size = Path(self.fp32_path).stat().st_size / 1024 / 1024
            int8_size = Path(self.int8_path).stat().st_size / 1024 / 1024
            torch_size = sum(p.numel() * p.element_size() for p in self.torch_model.parameters()) / 1024 / 1024
            results.append(f"\n模型大小:")
            results.append(f"  PyTorch:    {torch_size:.2f} MB")
            results.append(f"  ONNX FP32:  {fp32_size:.2f} MB")
            results.append(f"  ONNX INT8:  {int8_size:.2f} MB")
            if int8_size > 0:
                results.append(f"  压缩比:     {fp32_size / int8_size:.2f}x")

        return "\n".join(results)

    def infer_compare(self, shape_name: str):
        """三种推理方式对比"""
        if self.fp32_session is None or self.int8_session is None:
            return None, "请先点击「导出+量化」按钮", ""

        points = generate_shape(shape_name, self.num_points)
        input_tensor = torch.from_numpy(points).unsqueeze(0)
        np_input = input_tensor.numpy()

        results_text = []
        results_text.append(f"输入形状: {shape_name} ({self.num_points} 点)")
        results_text.append("=" * 50)

        t0 = time.perf_counter()
        with torch.no_grad():
            torch_logits, _ = self.torch_model(input_tensor)
        torch_time = (time.perf_counter() - t0) * 1000
        torch_logits = torch_logits.numpy()[0]
        torch_top5 = np.argsort(torch_logits)[::-1][:5]
        results_text.append(f"\nPyTorch:")
        results_text.append(f"  延迟: {torch_time:.2f} ms")
        results_text.append(f"  Top-5: {[MODELNET40_CLASSES[i] for i in torch_top5]}")

        input_name = self.fp32_session.get_inputs()[0].name

        t0 = time.perf_counter()
        fp32_logits = self.fp32_session.run(["logits"], {input_name: np_input})[0]
        fp32_time = (time.perf_counter() - t0) * 1000
        fp32_logits = fp32_logits[0]
        fp32_top5 = np.argsort(fp32_logits)[::-1][:5]
        results_text.append(f"\nONNX FP32:")
        results_text.append(f"  延迟: {fp32_time:.2f} ms")
        results_text.append(f"  Top-5: {[MODELNET40_CLASSES[i] for i in fp32_top5]}")

        t0 = time.perf_counter()
        int8_logits = self.int8_session.run(["logits"], {input_name: np_input})[0]
        int8_time = (time.perf_counter() - t0) * 1000
        int8_logits = int8_logits[0]
        int8_top5 = np.argsort(int8_logits)[::-1][:5]
        results_text.append(f"\nONNX INT8:")
        results_text.append(f"  延迟: {int8_time:.2f} ms")
        results_text.append(f"  Top-5: {[MODELNET40_CLASSES[i] for i in int8_top5]}")

        results_text.append(f"\n{'=' * 50}")
        results_text.append(f"速度对比:")
        results_text.append(f"  PyTorch → INT8 加速: {torch_time / int8_time:.2f}x")
        results_text.append(f"  FP32 → INT8 加速: {fp32_time / int8_time:.2f}x")

        pred_match = "✅ 一致" if torch_top5[0] == int8_top5[0] else "⚠️ 不一致"
        results_text.append(f"\n预测一致性: {pred_match}")
        results_text.append(f"  PyTorch Top-1: {MODELNET40_CLASSES[torch_top5[0]]}")
        results_text.append(f"  INT8 Top-1:    {MODELNET40_CLASSES[int8_top5[0]]}")

        from src.benchmark import visualize_accuracy
        img_path = visualize_accuracy(
            torch_logits.reshape(1, -1),
            fp32_logits.reshape(1, -1),
            int8_logits.reshape(1, -1),
        )

        return img_path, "\n".join(results_text), ""

    def run_benchmark(self):
        """运行完整基准测试"""
        if self.fp32_session is None or self.int8_session is None:
            return None, "请先点击「导出+量化」按钮"

        results = benchmark_all(
            self.torch_model,
            self.fp32_path,
            self.int8_path,
            num_points=self.num_points,
            num_warmup=10,
            num_runs=50,
        )

        img_path = visualize_benchmark(results)

        summary = f"""基准测试结果 (50次运行)

延迟对比:
  PyTorch:    {results['torch']['avg_ms']:.2f} ms (FPS: {results['torch']['fps']:.1f})
  ONNX FP32:  {results['onnx_fp32']['avg_ms']:.2f} ms (FPS: {results['onnx_fp32']['fps']:.1f})
  ONNX INT8:  {results['onnx_int8']['avg_ms']:.2f} ms (FPS: {results['onnx_int8']['fps']:.1f})

加速比:
  PyTorch → INT8: {results['speedup_torch_vs_int8']:.2f}x
  FP32 → INT8:    {results['speedup_fp32_vs_int8']:.2f}x

P95 延迟:
  PyTorch: {results['torch']['p95_ms']:.2f} ms
  INT8:    {results['onnx_int8']['p95_ms']:.2f} ms"""
        return img_path, summary

    def build(self):
        with gr.Blocks(title="Edge Deployment Demo") as demo:
            gr.Markdown("# Edge Deployment Demo")
            gr.Markdown("PyTorch → ONNX → INT8 quantization. Compare accuracy and speed across 3 inference paths.")

            with gr.Tab("Export & Quantize"):
                gr.Markdown("### Step 1: Export ONNX + INT8 Quantize")
                export_btn = gr.Button("Export + Quantize", variant="primary")
                export_output = gr.Textbox(label="Result", lines=12, interactive=False)
                export_btn.click(fn=self.export_and_quantize, outputs=[export_output])

            with gr.Tab("Inference Compare"):
                gr.Markdown("### Single inference comparison")
                shape = gr.Dropdown(SHAPES, label="Select shape", value="chair")
                infer_btn = gr.Button("Run Inference", variant="primary")
                infer_image = gr.Image(label="Logits Comparison")
                infer_output = gr.Textbox(label="Result", lines=15, interactive=False)
                infer_btn.click(fn=self.infer_compare, inputs=[shape], outputs=[infer_image, infer_output])

            with gr.Tab("Benchmark"):
                gr.Markdown("### Batch benchmark (50 runs)")
                bench_btn = gr.Button("Run Benchmark", variant="primary")
                bench_image = gr.Image(label="Speed Comparison")
                bench_output = gr.Textbox(label="Result", lines=12, interactive=False)
                bench_btn.click(fn=self.run_benchmark, outputs=[bench_image, bench_output])

            with gr.Tab("Help"):
                gr.Markdown("""
                ### Edge Deployment Pipeline

                ```
                PyTorch model (.pth)
                    ↓ torch.onnx.export (dynamo=False, opset=17)
                ONNX FP32 (.onnx)
                    ↓ quantize_dynamic (INT8)
                ONNX INT8 (.onnx)
                    ↓ onnxruntime inference
                Edge device deployment
                ```

                ### Comparison Metrics

                | Metric | Description |
                |---|---|
                | Model size | FP32 vs INT8 file size |
                | Latency | Single inference time (ms) |
                | Throughput | FPS |
                | Accuracy | Logits diff + Top-1 consistency |
                | Latency dist | P50/P95/P99 |

                ### Technical Notes

                - **ONNX**: Open Neural Network Exchange, cross-platform standard
                - **INT8 dynamic quantization**: Weights int8, activations fp32, no calibration needed
                - **ONNX Runtime**: Microsoft inference engine, CPU optimized
                - **Dynamic batch**: dynamic_axes supports variable batch size

                ### Relation to Real Projects

                This demo shows the standard industrial deployment pipeline:
                - Shoe project: model lightweighting for 3s/cycle production line
                - Thesis: edge device deployment
                """)

        return demo


if __name__ == "__main__":
    app = EdgeDeployApp()
    demo = app.build()
    demo.launch(server_name="0.0.0.0", server_port=7860)
