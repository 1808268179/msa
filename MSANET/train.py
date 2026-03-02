import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from datasets.transforms import build_train_transform, build_eval_transform
from datasets.fgvc_datasets import build_dataset
from models.msanet import build_model
from utils.misc import (
    set_seed,
    load_config,
    accuracy,
    AverageMeter,
    ensure_dir,
    save_checkpoint,
)


def build_loaders(cfg):
    aug = cfg["augmentation"]
    train_tf = build_train_transform(aug["resize"], aug["crop"])
    eval_tf = build_eval_transform(aug["resize"], aug["crop"])

    ds_train = build_dataset(
        cfg["dataset"]["name"], cfg["dataset"]["root"], "train", train_tf
    )
    ds_val = build_dataset(
        cfg["dataset"]["name"], cfg["dataset"]["root"], "val", eval_tf
    )

    print(f"Train samples: {len(ds_train)}")
    print(f"Val samples  : {len(ds_val)}")

    if hasattr(ds_train, "classes"):
        print(f"Classes      : {len(ds_train.classes)}")
        print(f"class_to_idx : {ds_train.class_to_idx}")

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
        drop_last=True,
    )

    loader_val = DataLoader(
        ds_val,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
        drop_last=False,
    )
    return loader_train, loader_val


def build_optimizer(cfg, model):
    backbone_params = []
    new_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            new_params.append(p)

    opt = SGD(
        [
            {"params": backbone_params, "lr": cfg["optimizer"]["backbone_lr"]},
            {"params": new_params, "lr": cfg["optimizer"]["new_lr"]},
        ],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    return opt


def evaluate(model, loader, criterion, device, amp=False):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Eval", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)

            acc = accuracy(logits, targets)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"].get("seed", 42))
    ensure_dir(cfg["output"]["dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    loader_train, loader_val = build_loaders(cfg)

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=cfg["scheduler"].get("eta_min", 1e-6),
    )
    scaler = GradScaler(enabled=cfg["train"].get("amp", True))

    best_acc = 0.0
    best_path = str(Path(cfg["output"]["dir"]) / "best.pth")
    last_path = str(Path(cfg["output"]["dir"]) / "last.pth")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(loader_train, desc=f'Epoch {epoch + 1}/{cfg["train"]["epochs"]}')
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg["train"].get("amp", True)):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = accuracy(logits, targets)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

        scheduler.step()

        val_loss, val_acc = evaluate(
            model, loader_val, criterion, device, amp=cfg["train"].get("amp", True)
        )
        print(f"[Epoch {epoch + 1}] val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "config": cfg,
        }
        save_checkpoint(state, last_path)

        if val_acc > best_acc:
            best_acc = val_acc
            state["best_acc"] = best_acc
            save_checkpoint(state, best_path)
            print(f"New best: {best_acc:.4f}")

    print(f"Training done. Best acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
