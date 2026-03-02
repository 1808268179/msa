from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as sio
from torchvision.datasets import ImageFolder


class CUBDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        images = {}
        labels = {}
        splits = {}

        with open(self.root / "images.txt", "r") as f:
            for line in f:
                idx, path = line.strip().split(" ")
                images[int(idx)] = path

        with open(self.root / "image_class_labels.txt", "r") as f:
            for line in f:
                idx, label = line.strip().split(" ")
                labels[int(idx)] = int(label) - 1

        with open(self.root / "train_test_split.txt", "r") as f:
            for line in f:
                idx, is_train = line.strip().split(" ")
                splits[int(idx)] = int(is_train)

        want_train = split == "train"
        for idx in images:
            if bool(splits[idx]) == want_train:
                self.samples.append((self.root / "images" / images[idx], labels[idx]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class AircraftDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if split == "train":
            ann_files = ["images_variant_train.txt", "images_variant_val.txt"]
        else:
            ann_files = ["images_variant_test.txt"]

        classes = {}
        cls_id = 0
        tmp = []
        for ann_file in ann_files:
            with open(self.root / ann_file, "r") as f:
                for line in f:
                    line = line.strip()
                    image_id, variant = line.split(" ", 1)
                    if variant not in classes:
                        classes[variant] = cls_id
                        cls_id += 1
                    tmp.append(
                        (self.root / "images" / f"{image_id}.jpg", classes[variant])
                    )
        self.samples = tmp

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class StanfordCarsDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if split == "train":
            anno_path = self.root / "devkit" / "cars_train_annos.mat"
            image_dir = self.root / "cars_train"
            key = "annotations"
        else:
            anno_path = self.root / "cars_test_annos_withlabels.mat"
            image_dir = self.root / "cars_test"
            key = "annotations"

        mat = sio.loadmat(anno_path)
        annos = mat[key][0]
        for a in annos:
            fname = str(a["fname"][0]) if isinstance(a, dict) else str(a[-1][0])
            cls = (
                int(a["class"][0][0]) - 1
                if isinstance(a, dict)
                else int(a[4][0][0]) - 1
            )
            self.samples.append((image_dir / fname, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_dataset(name, root, split, transform):
    name = name.lower()

    if name == "cub":
        return CUBDataset(root, split=split, transform=transform)

    if name in ["aircraft", "fgvc-aircraft", "fgvc_aircraft"]:
        return AircraftDataset(root, split=split, transform=transform)

    if name in ["cars", "stanford_cars", "stanford-cars"]:
        return StanfordCarsDataset(root, split=split, transform=transform)

    if name in ["imagefolder", "folder", "generic_folder", "custom"]:

        split_root = Path(root) / split
        if not split_root.exists():
            raise FileNotFoundError(f"Dataset split path does not exist: {split_root}")
        return ImageFolder(str(split_root), transform=transform)

    raise ValueError(f"Unsupported dataset: {name}")
