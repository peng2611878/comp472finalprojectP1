import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import model_selection
from skorch.helper import SliceDataset
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, \
    make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from skorch import NeuralNetClassifier
from torch.utils.data import random_split

if __name__ == '__main__':
    # set up the dataset for train
    data_transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    masks_dataset = datasets.ImageFolder(root='./img/train',
                                         transform=data_transform)
    num_epochs = 35
    num_classes = 4
    learning_rate = 0.001
    n = len(masks_dataset)
    trainset, testset = torch.utils.data.random_split(masks_dataset, [int(n - int(n * 0.25)), int(n * 0.25)])
    m = len(trainset)
    train_data, val_data = random_split(trainset, [int(m - int(m * 0.2)), int(m * 0.2)])
    DEVICE = torch.device("cuda")
    y_train = np.array([y for x, y in iter(train_data)]).astype(np.int64)

    # CNN architecture for training process
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
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=256, out_channels=425, kernel_size=3, padding=1),
                nn.BatchNorm2d(425),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=425, out_channels=425, kernel_size=3, padding=1),
                nn.BatchNorm2d(425),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),


            )

            self.fc_layer = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(1700, 1000),
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
        device=DEVICE,

    )
    # net.fit(train_data, y=y_train)
    # net.save_params(f_params='model2+.pkl')
# Load the training model for train/test split evaluation
    net.initialize()
    net.load_params(f_params='model2+.pkl')
#     y_pred = net.predict(testset)
#     y_test = np.array([y for x, y in iter(testset)])
#     # print(accuracy_score(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
#     labels = ['cloth', 'n95', 'nomask', 'surgical']
#     cm = confusion_matrix(y_test, y_pred)
#     print(cm)
#     ConfusionMatrixDisplay(cm, display_labels=labels).plot()
#     plt.show()

# K_fold cross-validation
    train_sliceable = SliceDataset(train_data)
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='macro'),
               'recall': make_scorer(recall_score, average='macro'),
               'f1_score': make_scorer(f1_score, average='macro')}

    results = model_selection.cross_validate(net, train_sliceable, y_train, cv=10,
                                             scoring=scoring)
    print('Precision', results['test_precision'])
    print('Recall', results['test_recall'])
    print('F1-score', results['test_f1_score'])
    print('Accuracy', results['test_accuracy'])
    print('Aggregate Precision', results['test_precision'].mean())
    print('Aggregate Recall', results['test_recall'].mean())
    print('Aggregate F1-score', results['test_f1_score'].mean())
    print('Aggregate Accuracy', results['test_accuracy'].mean())
