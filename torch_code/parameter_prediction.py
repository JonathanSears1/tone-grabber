import torch
import auraloss

class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim):
        super(MLP,self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embed_dim),  # Adjust input size to match flattened output
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, output_dim),
        )
    def forward(self,x):
        return self.mlp(x)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Flatten()
        )
    def  forward(self,x):
        while len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        return x

class F0_Embed(torch.nn.Module):
    def __init__(self,input_dim, output_dim):
        super(F0_Embed,self).__init__()
        self.flatten = torch.nn.Flatten()
        self.embed = torch.nn.Linear(input_dim, output_dim)
    def forward(self,x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        x = self.flatten(x)
        x = self.embed(x)
        return x

class ParameterPrediction(torch.nn.Module):
    def __init__(self, num_effects : int,num_parameters, parameter_mask : dict, num_heads = 2, dropout = .1, embed_dim=768):
        super(ParameterPrediction, self).__init__()
        self.parameter_mask = parameter_mask
        self.cnn = CNN()
        self.loudness_embedding = MLP(257,embed_dim,embed_dim)
        self.f0_embedding = F0_Embed(21912,embed_dim)
        self.mlp_feat = MLP(embed_dim*4, embed_dim*2,embed_dim*2)
        self.mlp_1 = MLP(128*1764,embed_dim,embed_dim)
        self.self_attn = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(embed_dim * 2, embed_dim)
        self.cls = torch.nn.Linear(embed_dim, num_effects)
        self.parameter_pred = MLP(embed_dim*2,embed_dim,num_parameters)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x_wet, x_dry, loudness_wet, f0_wet, loudness_dry, f0_dry):
        '''
        Args:
            x_wet (torch.Tensor): Input spectrogram of the wet (processed) audio signal [batch_size, 1, height, width]
            x_dry (torch.Tensor): Input spectrogram of the dry (unprocessed) audio signal [batch_size, 1, height, width]
            loudness_wet (torch.Tensor): Loudness features of the wet signal [batch_size, 257]
            f0_wet (torch.Tensor): F0 (fundamental frequency) features of the wet signal [batch_size, 357, 88]
            loudness_dry (torch.Tensor): Loudness features of the dry signal [batch_size, 257]
            f0_dry (torch.Tensor): F0 (fundamental frequency) features of the dry signal [batch_size, 357, 88]

        Returns:
            torch.Tensor: Processed features containing effect classification and parameter predictions
            
        The forward pass processes wet and dry audio spectrograms through a CNN, combines them,
        and applies self-attention. It then processes loudness and F0 features through MLPs,
        applies cross-attention between spectrogram and audio features, and finally predicts
        both the effect type and its parameters using the parameter mask for the predicted effect.
        '''
        # Check and adjust dimensions of input spectrograms
        x_wet = self.cnn(x_wet)  # Adjust unsqueeze dimension
        x_dry = self.cnn(x_dry)  # Adjust unsqueeze dimension
        x_wet = self.mlp_1(x_wet)
        x_dry = self.mlp_1(x_dry)
        x = torch.cat([x_wet, x_dry], dim=1)
        x, _ = self.self_attn(x, x, x)  # Unpack attn output
        # wet features
        loudness_embed_wet = self.loudness_embedding(loudness_wet)
        f0_embed_wet = self.f0_embedding(f0_wet)
        features_wet = torch.cat([f0_embed_wet,loudness_embed_wet],dim=1)
        # dry features
        loudness_embed_dry = self.loudness_embedding(loudness_dry)
        f0_embed_dry = self.f0_embedding(f0_dry)
        features_dry = torch.cat([f0_embed_dry,loudness_embed_dry],dim=1)
        features = torch.cat([features_wet,features_dry],dim=1)
        features = self.mlp_feat(features)
        x, _ = self.cross_attn(x,features,x)
        effect = self.cls(self.fc(x))
        mask = torch.tensor(self.parameter_mask[int(torch.argmax(effect))])
        params = self.parameter_pred(x)
        params = self.softmax(params) * mask
        return x, effect, params
    
