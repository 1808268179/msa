# msa

## Grad-CAM 可视化（仅 Grad-CAM 版本）

当前仓库已提供 **Grad-CAM 可视化脚本**：

- 脚本路径：`MSANET/tools/visualize_heatmap.py`
- 支持：
  - 单张图片可视化（`--image`）
  - 批量处理目录下全部图片（`--image-dir`）
  - 可选递归子目录（`--recursive`）

---

## 1. 你需要准备什么

1. 配置文件（例如）：`MSANET/configs/custom_resnet50.yaml`
2. 你训练保存的最佳模型（例如）：`/path/to/best.pth`
3. 图片（单张）或图片目录（批量）

> 注意：`--checkpoint` 请传你保存的最佳 `pth` 文件路径。

---

## 2. 单张图片可视化（`--image`）

### 方式 A：在仓库根目录执行（推荐）

```bash
python MSANET/tools/visualize_heatmap.py \
  --config MSANET/configs/custom_resnet50.yaml \
  --checkpoint /path/to/best.pth \
  --image "株 1_IMG_20250922_105708_1.jpg" \
  --target-class 10 \
  --output outputs/gradcam
```

### 方式 B：先进入 `MSANET/` 目录执行

```bash
cd MSANET
python tools/visualize_heatmap.py \
  --config configs/custom_resnet50.yaml \
  --checkpoint /path/to/best.pth \
  --image "株 1_IMG_20250922_105708_1.jpg" \
  --target-class 10 \
  --output outputs/gradcam
```

---

## 3. 查看“全部图片”（批量）

你现在的命令只会看一张图，因为你用了 `--image`。  
如果要处理一个目录下**全部图片**，请改用 `--image-dir`。

### 3.1 批量处理当前目录下全部图片（不含子目录）

```bash
python MSANET/tools/visualize_heatmap.py \
  --config MSANET/configs/custom_resnet50.yaml \
  --checkpoint /path/to/best.pth \
  --image-dir /path/to/all_images \
  --target-class 10 \
  --output outputs/gradcam_all
```

### 3.2 批量处理目录 + 所有子目录（递归）

```bash
python MSANET/tools/visualize_heatmap.py \
  --config MSANET/configs/custom_resnet50.yaml \
  --checkpoint /path/to/best.pth \
  --image-dir /path/to/all_images \
  --recursive \
  --target-class 10 \
  --output outputs/gradcam_all
```

脚本会在终端显示：

- 待处理图片数量
- 每张图的处理进度（如 `[3/120]`）
- 预测类别与置信度
- 每张图对应的输出文件路径

---

## 4. 参数详细说明

- `--config`：yaml 配置文件。
- `--checkpoint`：最佳模型 `pth`。
- `--image`：单张图片路径（和 `--image-dir` 二选一）。
- `--image-dir`：批量图片目录（和 `--image` 二选一）。
- `--recursive`：仅在 `--image-dir` 下生效，递归读取子目录。
- `--target-class`：指定 Grad-CAM 解释类别；不传时默认解释模型预测类别。
- `--alpha`：叠加透明度，默认 `0.45`。
- `--output`：输出目录。

支持的图片格式：`.jpg .jpeg .png .bmp .webp`

---

## 5. 输出文件说明

每张输入图会生成 3 张结果图：

1. `*_input.png`：模型输入图（经过预处理后的可视化）。
2. `*_gradcam.png`：Grad-CAM 伪彩热力图。
3. `*_overlay.png`：热力图叠加图。

例如输入 `株 1_IMG_20250922_105708_1.jpg`，会生成：

- `株 1_IMG_20250922_105708_1_input.png`
- `株 1_IMG_20250922_105708_1_gradcam.png`
- `株 1_IMG_20250922_105708_1_overlay.png`

---

## 6. 常见问题

1. **为什么只处理一张图？**
   - 因为你使用了 `--image`。要处理全部图片，请改用 `--image-dir`。

2. **报错 `ModuleNotFoundError: No module named datasets`**
   - 请使用最新版脚本。
   - 若在 `MSANET/` 目录运行，配置应写成 `configs/xxx.yaml`，不要写 `MSANET/configs/xxx.yaml`。

3. **报错 `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'`**
   - 已在脚本内兼容不同 matplotlib 版本，请更新到最新版脚本。

4. **图片名有空格/中文怎么办？**
   - 路径加引号：
   - `--image "株 1_IMG_20250922_105708_1.jpg"`
   - 或 `--image-dir "/data/我的图片目录"`

