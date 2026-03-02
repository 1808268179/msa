import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.transforms import build_eval_transform
from datasets.fgvc_datasets import build_dataset
from models.msanet import build_model
from utils.misc import load_config, accuracy, AverageMeter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aug = cfg["augmentation"]
    eval_tf = build_eval_transform(aug["resize"], aug["crop"])
    dataset = build_dataset(
        cfg["dataset"]["name"], cfg["dataset"]["root"], "test", eval_tf
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Test"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            acc = accuracy(logits, targets)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

    print(f"Test loss: {loss_meter.avg:.4f}")
    print(f"Test acc : {acc_meter.avg:.4f}")


if __name__ == "__main__":
    main()
