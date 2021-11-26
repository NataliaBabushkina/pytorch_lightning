from torchvision import transforms

def compose_transforms():
    return transforms.Compose([transforms.ToTensor(), transforms.Pad(32), transforms.RandomCrop(28)])