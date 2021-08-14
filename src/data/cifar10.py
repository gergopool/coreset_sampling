from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

__all__ = ['get_cifar10_dataloaders']

MEAN = [0.4914, 0.4824, 0.4467]
STD = [0.2470, 0.2436, 0.2616]

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def get_cifar10_dataloaders(root='/data/pytorch', batch_size=128, sampler=None, **kwargs):

    trainset = CIFAR10(root=root, train=True, download=True, transform=train_transforms)
    testset = CIFAR10(root=root, train=False, download=True, transform=test_transforms)

    if sampler:
        sampler = sampler(trainset)
        shuffle=False
    else:
        shuffle=True

    train = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    test  = DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)

    return train, test

if __name__ == '__main__':
    train, test = get_cifar10_dataloaders()
    x, y = next(iter(test))
    print(x.shape, y.shape)