import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleCNNModel(nn.Module):
    """ SimpleCNNModel is a simple CNN model to use as a baseline
    
        Model Structure:
            2x Convolutional Layers:
                - ReLU Activation
                - Batch Normalisation
                - Uniform Xavier Weigths
                - Max Pooling
      
            1x Fully Connected Layer:
                - ReLU activation
      
            1x Fully Connected Layer:
                - Output Layer
    """

    def __init__(self):
        super(SimpleCNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform_(self.cnn1.weight)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
        )
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform_(self.cnn2.weight)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4096, 4096)
        self.fcrelu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 5)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fcrelu(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim = 1)

class SimpleCNNModelWM(nn.Module):
    """ SimpleCNNModel is a simple CNN model to use as a baseline
    
        Model Structure:
            2x Convolutional Layers:
                - ReLU Activation
                - Batch Normalisation
                - Uniform Xavier Weigths
                - Max Pooling
      
            1x Fully Connected Layer:
                - ReLU activation
      
            1x Fully Connected Layer:
                - Output Layer
    """

    def __init__(self):
        super(SimpleCNNModelWM, self).__init__()

        self.cnn1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform_(self.cnn1.weight)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
        )
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform_(self.cnn2.weight)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(4096, 4096)
        self.fcrelu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 5)
        self.wmlayer1 = nn.Linear(5, 10)
        self.wmlayer2 = nn.Linear(10, 20)
        self.wmlayer3 = nn.Linear(20, 6)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fcrelu(out)
        out = self.fc2(out)

        out2 = self.wmlayer1(F.log_softmax(out, dim = 1))
        out2 = self.wmlayer2(out2)
        out2 = self.wmlayer3(out2)
        return F.log_softmax(out, dim = 1), F.log_softmax(out2, dim = 1)



class SimpleCNNModelATTACKERWM(nn.Module):
    """ SimpleCNNModel is a simple CNN model to use as a baseline
    
        Model Structure:
            2x Convolutional Layers:
                - ReLU Activation
                - Batch Normalisation
                - Uniform Xavier Weigths
                - Max Pooling
      
            1x Fully Connected Layer:
                - ReLU activation
      
            1x Fully Connected Layer:
                - Output Layer
    """

    def __init__(self):
        super(SimpleCNNModelATTACKERWM, self).__init__()

        self.cnn1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform_(self.cnn1.weight)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
        )
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform_(self.cnn2.weight)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(4096, 4096)
        self.fcrelu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 5)

        self.wmlayer1 = nn.Linear(5, 20)
        self.wmlayer2 = nn.Linear(20, 10)
        self.wmlayer3 = nn.Linear(10, 6)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fcrelu(out)
        out = self.fc2(out)

        out2 = self.wmlayer1(F.log_softmax(out, dim = 1))
        out2 = self.wmlayer2(out2)
        out2 = self.wmlayer3(out2)
        return F.log_softmax(out, dim = 1), F.log_softmax(out2, dim = 1)


class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(120)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return F.softmax(logits, dim=1)


class LeNet5WM(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5WM, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(120)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

        self.wmlayer1 = nn.Linear(5, 10)
        self.wmlayer2 = nn.Linear(10, 20)
        self.wmlayer3 = nn.Linear(20, 6)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        out2 = self.wmlayer1(F.softmax(logits, dim=1))
        out2 = self.wmlayer2(out2)
        out2 = self.wmlayer3(out2)
        return F.softmax(logits, dim=1), F.softmax(out2, dim=1)


class LeNet5AttackerWM(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5AttackerWM, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.BatchNorm2d(120)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

        self.wmlayer1 = nn.Linear(5, 20)
        self.wmlayer2 = nn.Linear(20, 10)
        self.wmlayer3 = nn.Linear(10, 6)


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        out2 = self.wmlayer1(F.softmax(logits, dim=1))
        out2 = self.wmlayer2(out2)
        out2 = self.wmlayer3(out2)
        return F.softmax(logits, dim=1), F.softmax(out2, dim=1)


class LeNet(nn.Module):
    def __init__(self, num_of_class=5):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_of_class)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

class LeNetWM(nn.Module):
    def __init__(self, num_of_class=5):
        super(LeNetWM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_of_class)

        self.wmlayer1 = nn.Linear(5, 10)
        self.wmlayer2 = nn.Linear(10, 20)
        self.wmlayer3 = nn.Linear(20, 6)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        out2 = self.wmlayer1(F.log_softmax(out, dim=1))
        out2 = self.wmlayer2(out2)
        out2 = self.wmlayer3(out2)

        return F.log_softmax(out, dim=1), F.log_softmax(out2, dim = 1)


class LeNetAttackerWM(nn.Module):
    def __init__(self, num_of_class=5):
        super(LeNetAttackerWM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_of_class)

        self.wmlayer1 = nn.Linear(5, 20)
        self.wmlayer2 = nn.Linear(20, 10)
        self.wmlayer3 = nn.Linear(10, 6)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        out2 = self.wmlayer1(F.log_softmax(out, dim=1))
        out2 = self.wmlayer2(out2)
        out2 = self.wmlayer3(out2)

        return F.log_softmax(out, dim=1), F.log_softmax(out2, dim = 1)