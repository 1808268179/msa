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
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def collect_images(single_image: str = None, image_dir: str = None, recursive: bool = False):
    if bool(single_image) == bool(image_dir):
        raise ValueError("必须且只能提供 --image 或 --image-dir 其中一个参数")

    if single_image:
        image_path = Path(single_image)
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")
        return [image_path]

    root = Path(image_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {root}")

    iterator = root.rglob("*") if recursive else root.glob("*")
    images = [p for p in iterator if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]
    images.sort()
    if not images:
        raise FileNotFoundError(f"目录下未找到可用图片: {root}")
    return images


def run_one_image(model, transform, image_path: Path, device: str, target_class, alpha: float, output_dir: Path):
    pil_img = Image.open(image_path).convert("RGB")
    image_tensor = transform(pil_img).unsqueeze(0).to(device)

    logits, aux = model(image_tensor, return_aux=True)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax().item())
    pred_score = float(probs[pred_idx].item())

    target_idx = pred_idx if target_class is None else int(target_class)
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
    overlay = blend_heatmap(rgb, cam, alpha=alpha)

    stem = image_path.stem
    input_path = output_dir / f"{stem}_input.png"
    cam_path = output_dir / f"{stem}_gradcam.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    Image.fromarray((rgb * 255).astype(np.uint8)).save(input_path)
    Image.fromarray((apply_jet_colormap(cam) * 255).astype(np.uint8)).save(cam_path)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)

    return {
        "image": str(image_path),
        "pred_idx": pred_idx,
        "pred_score": pred_score,
        "target_idx": target_idx,
        "input_path": str(input_path),
        "cam_path": str(cam_path),
        "overlay_path": str(overlay_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM for MSANet")
    parser.add_argument("--config", type=str, required=True, help="训练/评估用的yaml配置文件")
    parser.add_argument("--checkpoint", type=str, required=True, help="你保存的最佳pth模型")
    parser.add_argument("--image", type=str, default=None, help="要可视化的单张图片路径")
    parser.add_argument("--image-dir", type=str, default=None, help="批量可视化目录下全部图片")
    parser.add_argument("--recursive", action="store_true", help="递归读取 --image-dir 的子目录")
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

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out_dir = Path(args.output)
    ensure_dir(out_dir)

    image_list = collect_images(args.image, args.image_dir, args.recursive)
    print(f"待处理图片数量: {len(image_list)}")

    for idx, image_path in enumerate(image_list, start=1):
        result = run_one_image(
            model=model,
            transform=transform,
            image_path=image_path,
            device=device,
            target_class=args.target_class,
            alpha=args.alpha,
            output_dir=out_dir,
        )
        print(
            f"[{idx}/{len(image_list)}] {result['image']} -> "
            f"pred={result['pred_idx']}({result['pred_score']:.4f}), "
            f"target={result['target_idx']}"
        )
        print(f"  原图: {result['input_path']}")
        print(f"  热力图: {result['cam_path']}")
        print(f"  叠加图: {result['overlay_path']}")


if __name__ == "__main__":
    main()
