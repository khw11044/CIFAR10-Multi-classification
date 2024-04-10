import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class BasicNet(nn.Module):
    def __init__(self, num_classes=10):
        super(BasicNet, self).__init__()

        #input = 3, output = 6, kernal = 5
        self.conv1 = nn.Conv2d(3, 6, 5)
        #kernal = 2, stride = 2, padding = 0 (default)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #input feature, output feature
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    # 값 계산
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MyCNNNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNNNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout25 = nn.Dropout(p=0.25)
        self.dropout50 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(1 * 1 * 256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    # 값 계산
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pooling(x)
        x = self.dropout25(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.pooling(x)
        x = self.dropout25(x)

        x = self.conv5(x)
        x = torch.relu(x)
        x = self.pooling(x)
        x = self.dropout25(x)

        x = self.conv6(x)
        x = torch.relu(x)
        x = self.pooling(x)
        x = self.dropout25(x)

        x = self.conv7(x)
        x = torch.relu(x)
        x = self.pooling(x)
        x = self.dropout25(x)

        x = x.view(-1, 1 * 1 * 256)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout50(x)
        x = self.fc2(x)
        
        return x
    
# Transfer Learning 모델
class TransModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        model = timm.create_model('densenet121', pretrained=True, in_chans=3, num_classes=1)
        num_features = model.num_features
        self.extractor_features = nn.Sequential(*(list(model.children())[:-1]))
        self.fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.extractor_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x