import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from skorch import NeuralNetClassifier
from torch.utils.data import random_split

if __name__ == '__main__':

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    masks_dataset = datasets.ImageFolder(root='./img/train',
                                         transform=data_transform)


    # n = len(masks_dataset)
    # trainset, testset = torch.utils.data.random_split(masks_dataset, [int(n - int(n * 0.25)), int(n * 0.25)])
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
    #                                            shuffle=True, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
    #                                           shuffle=False, num_workers=2)
    # classes = ('cloth', 'n95', 'nomask', 'surgical')

    num_epochs = 4
    num_classes = 4
    learning_rate = 0.001
    #
    n = len(masks_dataset)
    trainset, testset = torch.utils.data.random_split(masks_dataset, [int(n - int(n * 0.25)), int(n * 0.25)])
    m = len(trainset)
    train_data, val_data = random_split(trainset, [int(m - int(m * 0.2)), int(m * 0.2)])
    DEVICE = torch.device("cpu")
    y_train = np.array([y for x, y in iter(train_data)])
    classes = ('cloth', 'n95', 'nomask', 'surgical')


    class CNN(nn.Module):

        def __init__(self):
            super(CNN, self).__init__()
            self.conv_layer = nn.Sequential(

                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.fc_layer = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(3136, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(512, 4)
            )

        def forward(self, x):
            # conv layers
            x = self.conv_layer(x)

            # flatten
            x = x.view(x.size(0), -1)

            # fc layer
            x = self.fc_layer(x)

            return x

    torch.manual_seed(0)
    net = NeuralNetClassifier(
        CNN,
        max_epochs=num_epochs,
        iterator_train__num_workers=4,
        iterator_valid__num_workers=4,
        lr=1e-3,
        batch_size=64,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device=DEVICE
    )

    net.fit(train_data, y=y_train)
    y_pred = net.predict(testset)
    y_test = np.array([y for x, y in iter(testset)])
    # print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(net, testset, y_test.reshape(-1, 1))
    plt.show()
