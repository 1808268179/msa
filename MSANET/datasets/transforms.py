from torchvision import transforms


def build_train_transform(resize=512, crop=448):
    return transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.RandomCrop((crop, crop)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(resize=512, crop=448):
    return transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
