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
    def __init__(self,embed_dim,num_parameters,num_heads=8,dropout=.25):
        super(ParameterPredictionResNet,self).__init__()
        self.resnet = self.make_resnet(embed_dim)
        self.fc = torch.nn.Linear(embed_dim,num_parameters)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(embed_dim,embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm(embed_dim)
        )
        self.attention = torch.nn.MultiheadAttention(embed_dim,num_heads,dropout,batch_first=True)

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



class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(MLP,self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embed_dim),  # Adjust input size to match flattened output
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim)
        )
    def forward(self,x):
        return self.mlp(x)

class CNN(torch.nn.Module):
    def __init__(self, batch_size):
        super(CNN,self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(),
            torch.nn.AdaptiveAvgPool2d((1,1))
        )
        self.batch_size = batch_size
    def  forward(self,x):
        while len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        return x.view(self.batch_size,-1)

class CNN_cls(torch.nn.Module):
    def __init__(self,batch_size):
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(512,1024, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.AdaptiveAvgPool2d((1,1))
            )
        self.batch_size = batch_size
        return
    def  forward(self,x):
        while len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        return x.view(self.batch_size,-1)

# class EffectClassifier(torch.nn.Module):
#     def __init__(self, n_classes, batch_size,embed_dim=768):
#         super(EffectClassifier, self).__init__()
#         self.cnn = CNN_cls(batch_size)
#         self.mlp = MLP()
#         self.attn = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=2, dropout=.1, batch_first=True)
#         self.fc = torch.nn.Linear(embed_dim * 2, embed_dim)
#         self.cls = torch.nn.Linear(embed_dim, n_classes)

#     @torch.autocast(device_type="cuda",dtype=torch.bfloat16)
#     def forward(self, x_wet, x_dry):
#         x_wet = self.cnn(x_wet.unsqueeze(1))  # Adjust unsqueeze dimension
#         x_dry = self.cnn(x_dry.unsqueeze(1))  # Adjust unsqueeze dimension
#         x_wet = self.mlp(x_wet)
#         x_dry = self.mlp(x_dry)
#         x = torch.cat([x_wet, x_dry], dim=1)
#         x, _ = self.attn(x, x, x)  # Unpack attn output
#         x = self.cls(self.fc(x))
#         return x





class Trainer():
    def __init__(self, model, metadata,min_val,max_val, lambda_=.5, sample_rate=16000):
        self.model = model
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractorTorch()
        self.sigmoid = torch.nn.Sigmoid()
        self.lambda_ = lambda_
        return
    
    
    def train_param_pred(self, model, train_loader, test_loader, loss_fn, optimizer, lr_scheduler,batch_size,effect_name,epochs):
        model.train()
        best_rmse = 9999999
        param_labels = []
        param_preds = []
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm.tqdm(train_loader):
                optimizer.zero_grad()
                # wet_tone_feat = batch['wet_tone']
                # dry_tone_feat = batch['dry_tone']
                # ,wet_tone_feat['loudness'].to(device),wet_tone_feat['f0'].to(device),dry_tone_feat['loudness'].to(device),dry_tone_feat['f0'].to(device)
                params = model(batch['joint_spectrogram'].to(device))
                # ]scaled_params = self.scaler.fit_transform(params)
                # ]scaled_labels = self.scaler.fit_transform(batch['param_dict'][0][param_name].to(torch.float32).to(device))
                loss = loss_fn(params,batch['parameters'].to(torch.bfloat16).to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                for pred in params:
                    param_preds.append(pred.to(torch.float32).detach().cpu().numpy())
                for label in batch['parameters']:
                    param_labels.append(label.detach().cpu().numpy())
            r2 = r2_score(np.array(param_labels),np.array(param_preds))
            rmse_final = math.sqrt(total_loss / (len(test_loader) * batch_size))
            print(f"Training {effect_name}: Epoch {epoch+1} | Parameter R^2: {r2} | Average Parameter RMSE Loss: {rmse_final}")
            test_rmse, r2 = self.eval_param_pred(model, test_loader, loss_fn, batch_size,effect_name, epoch)
            lr_scheduler.step(loss)
            if test_rmse < best_rmse:
                print(f"saving model at epoch {epoch+1}")
                best_rmse = test_rmse
                torch.save(model.state_dict(), f"saved_models/{effect_name}_parameter_prediction.pth")
        return
    @torch.no_grad
    def eval_param_pred(self,model, test_loader, loss_fn, batch_size,effect_name, epoch):
        param_labels = []
        param_preds = []
        total_loss = 0
        for batch in tqdm.tqdm(test_loader):
            # wet_tone_feat = batch['wet_tone']
            # dry_tone_feat = batch['dry_tone']
            params = model(batch['joint_spectrogram'].to(device))
            
            # scaled_params = self.scaler.fit_transform(params)
            # scaled_labels = self.scaler.fit_transform(batch['param_dict'][0][param_name].to(torch.float32).to(device))
            loss = loss_fn(params,batch['parameters'].to(torch.bfloat16).to(device))
            total_loss += loss.item()
            for pred in params:
                param_preds.append(pred.to(torch.float32).detach().cpu().numpy())
            for label in batch['parameters'].to(torch.float32):
                param_labels.append(label.detach().cpu().numpy())
        r2 = r2_score(np.array(param_labels),np.array(param_preds))
        rmse_final = math.sqrt(total_loss / (len(test_loader) * batch_size))
        print(f"Test {effect_name}: Epoch {epoch+1} | Parameter R^2: {r2} | Parameter RMSE Loss: {rmse_final}")
        return rmse_final, r2
    

class PostProcessor():
    def __init__(self, metadata,sample_rate=16000):
        self.metadata = metadata
        self.sample_rate = sample_rate
        return
    def process_audio_from_outputs(self, effect, params, dry_tone_path):
        with ReadableAudioFile(dry_tone_path) as f:
            re_sampled = f.resampled_to(self.sample_rate)
            dry_tone = np.squeeze(re_sampled.read(int(self.sample_rate * f.duration)),axis=0)
            re_sampled.close()
            f.close()
        predicted_effect_pb = self.metadata['effects'][int(torch.argmax(effect))]
        predicted_params = [float(param) for param in list(params.detach().squeeze(0)) if param != 0]
        effect_name = self.metadata['index_to_effect'][int(torch.argmax(effect))]
        param_names = self.metadata['effects_to_parameters'][effect_name].keys()
        
        # Scale parameters between min and max values
        scaled_params = []
        for param_name, param_value in zip(param_names, predicted_params):
            min_val, max_val = self.metadata['effects_to_parameters'][effect_name][param_name]
            # Squash between 0 and 1 using sigmoid
            squashed = 1 / (1 + np.exp(-param_value))
            # Scale to range
            scaled_value = min_val + squashed * (max_val - min_val)
            scaled_params.append(scaled_value)
            
        matched_params = {param_name:value for param_name,value in zip(param_names,scaled_params)}
        predicted_effect_with_params = predicted_effect_pb(**matched_params)
        
        predicted_wet = predicted_effect_with_params.process(dry_tone,self.sample_rate)
        return effect_name, predicted_wet, predicted_effect_with_params