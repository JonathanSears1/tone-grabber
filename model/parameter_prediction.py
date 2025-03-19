import torch
import tqdm
from pedalboard import Pedalboard
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error,r2_score
from pedalboard.io import ReadableAudioFile
import numpy as np
from dataset.feature_extractor_torch import FeatureExtractorTorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import math
# from model.classifier import EffectClassifier
# from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18

class ParameterPredictionResNet(torch.nn.Module):
    def __init__(self,embed_dim,num_parameters):
        super(ParameterPredictionResNet,self).__init__()
        self.resnet = self.make_resnet(embed_dim)
        self.fc = torch.nn.Linear(embed_dim,num_parameters)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(embed_dim,embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm(embed_dim)
        )
        
    def make_resnet(self,embed_dim):
        model = resnet18(weights=None)
        model.conv1 = torch.nn.Conv2d(
            in_channels=2,  # Change from 3 to 1
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
    
    @torch.autocast("cuda",torch.bfloat16)
    def forward(self,joint_spec):
        features = self.resnet(joint_spec)
        # features, _ = self.attention(features,features,features)
        features = self.MLP(features)
        param = self.fc(features)
        return param
