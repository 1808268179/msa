# msa

## Grad-CAM 可视化（仅 Grad-CAM 版本）

当前仓库已提供 **Grad-CAM 单图可视化脚本**：

- 脚本路径：`MSANET/tools/visualize_heatmap.py`
- 作用：加载你的配置文件和最佳 `pth` 模型，对单张图片做推理并生成 Grad-CAM 热力图。

---

## 1. 你需要准备什么

请先准备以下内容：

1. 配置文件（例如）：`MSANET/configs/custom_resnet50.yaml`
2. 你训练保存的最佳模型（例如）：`/path/to/best.pth`
3. 待可视化图片（例如）：`/path/to/test.jpg`

> 注意：`--checkpoint` 请传你保存的最佳 `pth` 文件路径。

---

## 2. 基础用法（解释模型预测类别）

### 方式 A：在仓库根目录执行（推荐）

```bash
python MSANET/tools/visualize_heatmap.py \
  --config MSANET/configs/custom_resnet50.yaml \
  --checkpoint /path/to/best.pth \
  --image /path/to/test.jpg \
  --output outputs/gradcam
```

### 方式 B：先进入 `MSANET/` 目录再执行

```bash
cd MSANET
python tools/visualize_heatmap.py \
  --config configs/custom_resnet50.yaml \
  --checkpoint /path/to/best.pth \
  --image /path/to/test.jpg \
  --output outputs/gradcam
```

含义说明：

- `--config`：模型结构和数据预处理配置。
- `--checkpoint`：你保存的最佳模型权重（`.pth`）。
- `--image`：输入图片路径。
- `--output`：输出目录，默认是 `outputs/gradcam`。

该命令默认会对 **模型预测出来的类别** 计算 Grad-CAM。

---

## 3. 指定解释某个类别（可选）

如果你想看指定类别（比如类别ID=10）的关注区域：

```bash
# 在仓库根目录
python MSANET/tools/visualize_heatmap.py \
  --config MSANET/configs/custom_resnet50.yaml \
  --checkpoint /path/to/best.pth \
  --image /path/to/test.jpg \
  --target-class 10 \
  --output outputs/gradcam

# 或在 MSANET 目录
python tools/visualize_heatmap.py \
  --config configs/custom_resnet50.yaml \
  --checkpoint /path/to/best.pth \
  --image /path/to/test.jpg \
  --target-class 10 \
  --output outputs/gradcam
```

额外参数：

- `--target-class`：指定要解释的类别 ID。
- `--alpha`：叠加透明度，默认 `0.45`，例如可设为 `0.6`。

---

## 4. 输出文件说明

运行后会在输出目录下生成 3 张图：

1. `*_input.png`：送入模型前处理后的输入图。
2. `*_gradcam.png`：Grad-CAM 伪彩色热力图（JET colormap）。
3. `*_overlay.png`：热力图和输入图叠加后的可视化结果。

同时终端会打印：

- 预测类别与置信度
- 实际用于 Grad-CAM 的类别 ID
- 三张输出图的保存路径

---

## 5. 常见问题

1. **报错找不到 checkpoint**
   - 检查 `--checkpoint` 路径是否正确，是否真的是你的最佳 `pth` 文件。
2. **热力图看起来很粗糙**
   - 这是正常现象，Grad-CAM 来自深层特征图，分辨率通常较低，再上采样到输入尺寸。
3. **CPU 也能跑吗**
   - 可以。脚本会自动检测 CUDA，不可用时使用 CPU。


4. **报错 `ModuleNotFoundError: No module named datasets`**
   - 已在脚本中兼容该问题。请确保使用最新版脚本。
   - 如果你在 `MSANET/` 目录运行，请使用相对路径：`--config configs/xxx.yaml`，不要写成 `MSANET/configs/xxx.yaml`。
