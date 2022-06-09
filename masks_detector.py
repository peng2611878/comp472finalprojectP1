import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from skorch import NeuralNetClassifier

if __name__ == '__main__':
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


    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    sample_dataset = datasets.ImageFolder(root='./img/sample',
                                          transform=data_transform)

    # download saved trained model for sample dataset evaluation
    num_epochs = 4
    DEVICE = torch.device("cpu")
    new_net = NeuralNetClassifier(
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
    new_net.initialize()
    new_net.load_params(f_params='model.pkl')
    y_pred = new_net.predict(sample_dataset)
    y_test = np.array([y for x, y in iter(sample_dataset)])
    # print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    labels = ['cloth', 'n95', 'nomask', 'surgical']
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot()
    plt.show()
