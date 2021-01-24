import torch
import torchvision
from torchvision.transforms import transforms

from model import CNN
from test import test
from train import train


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    cnn = CNN().cuda()
    print('train start:')
    cnn = train(cnn, train_loader, 2)
    print()

    print('test start')
    test(cnn, test_loader)
    print()

    print('DONE')


if __name__ == '__main__':
    main()
