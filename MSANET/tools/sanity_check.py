import torch
from utils.misc import load_config
from models.msanet import build_model


def main():
    cfg = load_config("configs/cub_resnet50.yaml")
    model = build_model(cfg)
    x = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        y, aux = model(x, return_aux=True)
    print("logits:", y.shape)
    for k, v in aux.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, type(v))


if __name__ == "__main__":
    main()
