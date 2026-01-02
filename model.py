import torch
import torch.nn as nn

n = 2
num_layers = f"{n}"

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=11, stride=stride, padding=5) 
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=11, stride=stride, padding=5)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class SV_CNN(nn.Module):
    def __init__(self, num_cat_channels, num_classes):
        super(SV_CNN, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(num_cat_channels, 128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False)
        )

        self.layers = nn.ModuleList()
        self.layers.append(ResidualBlock(128, 128))
        for _ in range(n):  
            self.layers.append(ResidualBlock(128, 128))

        self.final_conv = nn.Conv1d(128, 32, kernel_size=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(12800, 128)
        self.fc2 = nn.Linear(128, num_classes)
        

    def forward(self, x_cat):
        x_cat = self.initial(x_cat)
        for layer in self.layers:
            x_cat = layer(x_cat)
        x_cat = self.final_conv(x_cat)
        
        x_cat = x_cat.view(x_cat.size(0), -1)  # Flatten the output
        x_cat = self.fc1(x_cat)
        x_cat = self.relu(x_cat)
        x_cat = self.fc2(x_cat)
        
        x = torch.sigmoid(x_cat)
        return x

    def extract_features(self, x_cat):
        x_cat = self.initial(x_cat)
        for layer in self.layers:
            x_cat = layer(x_cat)
        x_cat = self.final_conv(x_cat)
        x_cat = x_cat.view(x_cat.size(0), -1)
        x_cat = self.fc1(x_cat)
        x_cat = self.relu(x_cat)
        return x_cat 