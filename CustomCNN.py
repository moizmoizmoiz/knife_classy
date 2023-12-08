import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 61 * 61, 512)  # Adjust the input size for 244x244 images
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.dropout1(self.relu1(self.conv1(x))))
        x = self.pool(self.dropout2(self.relu2(self.conv2(x))))
        x = self.pool(self.dropout3(self.relu3(self.conv3(x))))
        x = self.pool(self.dropout4(self.relu4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten the feature map

        x = self.dropout5(self.relu5(self.fc1(x)))
        x = self.fc2(x)
        return x


num_classes = 192  # 192 classes for your specific task
model = CustomCNN(num_classes)
print(model)
