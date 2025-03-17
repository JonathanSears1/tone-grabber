import torch
from torchvision.models import resnet18
from .parameter_prediction import CNN


class CNN(torch.nn.Module):
    def __init__(self, batch_size):
        super(CNN,self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.AdaptiveAvgPool2d((1,1))
        )
        self.batch_size = batch_size
    def  forward(self,x):
        while len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        return x.view(self.batch_size,-1)
    

class EffectClassifier(torch.nn.Module):
    def __init__(self, n_classes,batch_size=4, embed_dim=256):
        super(EffectClassifier, self).__init__()
        self.cnn = CNN(batch_size=batch_size)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),  # Adjust input size to match flattened output
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim)
        )
        self.fc = torch.nn.Linear(embed_dim * 2, embed_dim)
        self.cls = torch.nn.Linear(embed_dim, n_classes)
    def forward(self, x_wet, x_dry):
        x_wet = self.cnn(x_wet)  # Adjust unsqueeze dimension
        x_dry = self.cnn(x_dry)  # Adjust unsqueeze dimension
        x_wet = self.mlp(x_wet)
        x_dry = self.mlp(x_dry)
        x = torch.cat([x_wet, x_dry], dim=1)# Unpack attn output
        x = self.cls(self.fc(x))
        return x
    
class ResnetClassifier(torch.nn.Module):
    def __init__(self, n_classes,batch_size=4, embed_dim=512):
        super(ResnetClassifier, self).__init__()
    
    def make_resnet(self,embed_dim):
        model = resnet18(weights=None)
        model.conv1 = torch.nn.Conv2d(
            in_channels=1,  # Change from 3 to 1
            out_channels=model.conv1.out_channels,  # Keep the same number of output channels
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias is not None  # Keep bias settings the same
        )
        model.fc = torch.nn.Linear(model.fc.in_features,embed_dim)
    # Add adaptive pooling before the fully connected layer
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        return model