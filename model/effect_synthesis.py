import torch
import torch.nn as nn
from torchvision.models import resnet18

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(models.ResNet):
    def __init__(self, block, layers):
        super().__init__(block, layers)
        # Remove the final fully connected layer
        del self.fc  # Alternatively, you can set it as an identity layer: self.fc = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # Keeps the (batch, channels, 1, 1) shape

        return x  # Now returning feature maps without flattening

# Example usage with ResNet-18
def resnet18_feature_extractor():
    return ResNetFeatureExtractor(models.resnet.BasicBlock, [2, 2, 2, 2])


class EffectChainSynthesis(nn.Module):
    def __init__(self,embed_dim) -> None:
        super(EffectChainSynthesis,self).__init__()
        self.encoder = resnet18_feature_extractor()
        self.hidden = nn.Sequential(
            nn.Linear(embed_dim,embed_dim),
            nn.Linear()
            )
        