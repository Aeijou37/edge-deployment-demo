---
title: Edge Deployment Demo
emoji: ⚡
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "4.0.0"
app_file: src/app.py
pinned: false
---

# Edge Deployment Demo

> PyTorch → ONNX → INT8 Quantization → Inference Benchmark
> Compare three inference paths: PyTorch vs ONNX FP32 vs ONNX INT8 — accuracy, latency, model size.

---

## 📌 Overview

This project demonstrates the **standard edge deployment pipeline** for deep learning models:

1. **Export**: PyTorch → ONNX (cross-platform standard format)
2. **Quantize**: ONNX FP32 → ONNX INT8 (4x smaller, 2-3x faster on CPU)
3. **Benchmark**: Compare PyTorch / ONNX FP32 / ONNX INT8 on accuracy, latency, throughput
4. **Deploy**: ONNX Runtime inference (CPU-optimized)

This is the same pipeline I use in my industrial projects (shoe-sole robot vision: model lightweighting for 3s/cycle production line; thesis: edge device deployment).

---

## ✨ Features

- **One-click export + quantize**: PyTorch → ONNX FP32 → ONNX INT8
- **Accuracy comparison**: Logits diff, cosine similarity, Top-1 prediction consistency
- **Latency benchmark**: Average / P50 / P95 / P99 latency across 50 runs
- **Throughput**: FPS comparison
- **Model size**: FP32 vs INT8 file size + compression ratio
- **Gradio web interface**: interactive comparison, no code needed
- **Zero hardware dependency**: runs on free Colab CPU

---

## 🧠 Technical Stack

| Component | Selection |
|---|---|
| Model | PointNet (1024 points, 40 classes) |
| Export | torch.onnx.export (opset 14, dynamic batch) |
| Quantization | onnxruntime.quantization (INT8 dynamic) |
| Inference | ONNX Runtime (CPUExecutionProvider) |
| Frontend | Gradio |
| Visualization | Matplotlib |

---

## 🚀 Quick Start

### Run on Google Colab (no GPU needed)

1. Open `notebooks/edge-deployment-demo.ipynb` in Colab
2. Run all cells
3. Get a public Gradio link at the end

Or run manually:

```bash
git clone https://github.com/Aeijou37/edge-deployment-demo.git
cd edge-deployment-demo
pip install -r requirements.txt
python src/app.py
```

Open `http://localhost:7860`.

### Workflow in the demo

1. **导出与量化** tab → Click "导出 + 量化" → generates FP32 + INT8 ONNX models
2. **推理对比** tab → Select a shape → compare inference results across 3 paths
3. **基准测试** tab → Click "运行基准测试" → 50-run benchmark with visualization

---

## 📁 Project Structure

```
edge-deployment-demo/
├── README.md
├── requirements.txt
├── notebooks/
│   └── edge-deployment-demo.ipynb   # Colab one-click notebook
└── src/
    ├── export_onnx.py     # PyTorch → ONNX export + verification
    ├── quantize.py        # INT8 dynamic quantization + comparison
    ├── benchmark.py       # Speed benchmark + visualization
    ├── app.py             # Gradio frontend
    └── utils.py           # Point cloud generation + class names
```

---

## 📊 What You'll See

### Model Size

| Format | Size | Compression |
|---|---|---|
| PyTorch (FP32) | ~7 MB | 1x |
| ONNX FP32 | ~7 MB | 1x |
| ONNX INT8 | ~2 MB | ~4x |

### Latency (CPU, 1024 points)

| Path | Avg Latency | FPS |
|---|---|---|
| PyTorch | ~15 ms | ~65 |
| ONNX FP32 | ~5 ms | ~200 |
| ONNX INT8 | ~3 ms | ~330 |

> Actual numbers vary by hardware. INT8 is typically 2-3x faster than FP32 on CPU.

### Accuracy

| Metric | FP32 vs INT8 |
|---|---|
| Max logit diff | < 0.1 |
| Cosine similarity | > 0.999 |
| Top-1 prediction match | > 95% |

---

## 📊 Key Design Decisions

### 1. Dynamic quantization (not static)

Dynamic quantization only quantizes weights (int8), keeping activations in FP32. Advantages: no calibration dataset needed, fast to apply. Trade-off: slightly less compression than static quantization, but sufficient for classification tasks.

### 2. Dynamic batch axes

ONNX export uses `dynamic_axes={0: "batch_size"}`, allowing variable batch sizes at inference — essential for production (single sample vs batch processing).

### 3. ONNX Runtime over TensorRT

ONNX Runtime works on CPU (free Colab, edge devices) and GPU. TensorRT requires NVIDIA GPU + specific architecture. For a public demo, ORT is more accessible.

---

## 👤 Author

**Guojie Li**

GitHub: [Aeijou37](https://github.com/Aeijou37)
