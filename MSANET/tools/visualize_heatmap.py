import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
try:
    from matplotlib import colormaps
except Exception:
    colormaps = None
from matplotlib import cm

from datasets.transforms import build_eval_transform
from models.msanet import build_model
from utils.misc import ensure_dir, load_config


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def build_gradcam(feature_map: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    weights = gradients.mean(dim=(1, 2), keepdim=True)
    cam = (weights * feature_map).sum(dim=0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam




def apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    if colormaps is not None:
        return colormaps["jet"](heatmap)[..., :3]
    if hasattr(cm, "get_cmap"):
        return cm.get_cmap("jet")(heatmap)[..., :3]
    return cm.jet(heatmap)[..., :3]

def blend_heatmap(rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = apply_jet_colormap(heatmap)
    blended = (1 - alpha) * rgb_img + alpha * colored
    return np.clip(blended, 0, 1)


def main():
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM for MSANet")
    parser.add_argument("--config", type=str, required=True, help="训练/评估用的yaml配置文件")
    parser.add_argument("--checkpoint", type=str, required=True, help="你保存的最佳pth模型")
    parser.add_argument("--image", type=str, required=True, help="要可视化的单张图片路径")
    parser.add_argument("--output", type=str, default="outputs/gradcam", help="输出目录")
    parser.add_argument("--alpha", type=float, default=0.45, help="热力图叠加透明度")
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="指定要解释的类别ID（默认使用模型预测类别）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aug = cfg["augmentation"]
    transform = build_eval_transform(aug["resize"], aug["crop"])

    image_path = Path(args.image)
    pil_img = Image.open(image_path).convert("RGB")
    image_tensor = transform(pil_img).unsqueeze(0).to(device)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    logits, aux = model(image_tensor, return_aux=True)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax().item())
    pred_score = float(probs[pred_idx].item())

    target_idx = pred_idx if args.target_class is None else int(args.target_class)
    target_score = logits[0, target_idx]

    feat_map = aux["feat"]
    feat_map.retain_grad()

    model.zero_grad(set_to_none=True)
    target_score.backward()

    grad = feat_map.grad[0]
    fmap = feat_map[0]
    cam = build_gradcam(fmap, grad)
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0].detach().cpu().numpy()

    rgb = denormalize_image(image_tensor[0])
    overlay = blend_heatmap(rgb, cam, alpha=args.alpha)

    out_dir = Path(args.output)
    ensure_dir(out_dir)

    stem = image_path.stem
    input_path = out_dir / f"{stem}_input.png"
    cam_path = out_dir / f"{stem}_gradcam.png"
    overlay_path = out_dir / f"{stem}_overlay.png"

    Image.fromarray((rgb * 255).astype(np.uint8)).save(input_path)
    Image.fromarray((apply_jet_colormap(cam) * 255).astype(np.uint8)).save(cam_path)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)

    print(f"预测类别: {pred_idx}, 预测置信度: {pred_score:.4f}")
    print(f"Grad-CAM解释类别: {target_idx}")
    print(f"已保存原图: {input_path}")
    print(f"已保存热力图: {cam_path}")
    print(f"已保存叠加图: {overlay_path}")


if __name__ == "__main__":
    main()
