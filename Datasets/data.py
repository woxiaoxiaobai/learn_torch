import torchvision
import torchvision.transforms as transforms

def load_data(download=True):
    transform = transforms.ToTensor()
    train_data = torchvision.datasets.MNIST(root='./Datasets/', train=True, transform=transform, download=download)
    test_data = torchvision.datasets.MNIST(root='./Datasets/', train=False, transform=transform, download=download)
    return train_data, test_data

def get_train_test_loaders(train_data,test_data, batch_size=64):
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader