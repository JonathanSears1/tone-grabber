import torch
import tqdm
from pedalboard import Pedalboard
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error,r2_score
from pedalboard.io import ReadableAudioFile
import numpy as np
from dataset.feature_extractor_torch import FeatureExtractorTorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import math
from .classifier import EffectClassifier
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18

class ParameterPredictionResNet(torch.nn.Module):
    def __init__(self,embed_dim,param_dict,num_heads=8,dropout=.25):
        super(ParameterPredictionResNet,self).__init__()
        self.resnet = self.make_resnet(embed_dim)
        for param in param_dict.keys():
            setattr(self,f"{param}_fc",torch.nn.Linear(embed_dim*2,1))
        self.param_dict = param_dict
        self.attention = torch.nn.MultiheadAttention(embed_dim*2,num_heads,dropout,batch_first=True)

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
    def forward(self,dry_tone,wet_tone):
        dry_features = self.resnet(dry_tone)
        wet_features = self.resnet(wet_tone)
        features = torch.cat([wet_features,dry_features],dim=1)
        features, _ = self.attention(features,features,features)
        out_dict = {}
        params = []
        for param in self.param_dict.keys():
            pred = getattr(self,f"{param}_fc").forward((features))
            out_dict[param] = pred
            params.append(pred)
        params = torch.stack(params, dim=1).squeeze(-1)  # Stack along dim=1 and squeeze last dim
        return params



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
    def __init__(self, num_effects : int, num_parameters : int, parameter_mask : dict,batch_size=4, num_heads = 2, dropout = .5, embed_dim=768):
        super(ParameterPrediction, self).__init__()
        self.parameter_mask = parameter_mask
        self.cnn_cls = CNN(batch_size)
        self.cnn_reg = CNN(batch_size)
        self.mlp_1_dry_cls = MLP(1024,embed_dim)
        self.mlp_1_wet_cls = MLP(1024,embed_dim)
        self.mlp_1_dry_reg = MLP(1024,embed_dim)
        self.mlp_1_wet_reg = MLP(1024,embed_dim)
        self.self_attn_cls = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_attn_reg = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mlp_2 = MLP(embed_dim*2,embed_dim)
        self.cls = torch.nn.Linear(embed_dim, num_effects)
        self.mlp_3 = MLP(embed_dim*2,embed_dim)
        self.mlp_4 = MLP(embed_dim,embed_dim // 2)
        self.mlp_5 = MLP(embed_dim // 2, embed_dim // 4)
        self.parameter_pred = MLP(embed_dim // 4,num_parameters)
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_size = batch_size
    @torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16)
    def forward(self, x_wet, x_dry):
        '''
        Args:
            x_wet (torch.Tensor): Input spectrogram of the wet (processed) audio signal [batch_size, 1, height, width]
            x_dry (torch.Tensor): Input spectrogram of the dry (unprocessed) audio signal [batch_size, 1, height, width]
        Returns:
            torch.Tensor: Processed features containing effect classification and parameter predictions
            
        The forward pass processes wet and dry audio spectrograms through a CNN, combines them,
        and applies self-attention. It then processes loudness and F0 features through MLPs,
        applies cross-attention between spectrogram and audio features, and finally predicts
        both the effect type and its parameters using the parameter mask for the predicted effect.
        '''
        # Check and adjust dimensions of input spectrograms
        # x_wet_cls = self.cnn_cls(x_wet)
        # x_dry_cls = self.cnn_cls(x_dry)
        x_wet = self.cnn_reg(x_dry)
        x_dry = self.cnn_reg(x_wet)
        
        # x_wet_cls = self.mlp_1_wet_cls(x_wet_cls)
        # x_dry_cls = self.mlp_1_dry_cls(x_dry_cls)
        x_wet = self.mlp_1_wet_reg(x_wet)
        x_dry = self.mlp_1_dry_reg(x_dry)
        # x_cls = torch.cat([x_wet_cls, x_dry_cls], dim=1)
        x = torch.cat([x_wet, x_dry], dim=1)
        # x_cls, _ = self.self_attn_cls(x_cls, x_cls, x_cls)  # Unpack attn output
        x, _ = self.self_attn_reg(x, x, x)
        # effect = self.cls(self.mlp_2(x_cls))
        # effect_indices = torch.argmax(effect, dim=1)  # Get predicted effect index for each item in batch
        # mask = torch.stack([torch.tensor(self.parameter_mask[idx.item()]) for idx in effect_indices]).to(device)
        params = self.parameter_pred(self.mlp_5(self.mlp_4(self.mlp_3(x))))
        # params = params * mask
        return x, params

class EffectParameterPredictor(torch.nn.Module):
    def __init__(self,num_parameters,batch_size,embed_dim=768):
        super(EffectParameterPredictor,self).__init__()
        self.num_parameters=num_parameters
        self.embed_dim = embed_dim
        self.cnn = CNN(batch_size)
        self.embed_1 = MLP(2048,1024)
        self.embed_2 = MLP(1024,embed_dim)
        self.embed_3 = MLP(embed_dim,num_parameters)
        return
    @torch.autocast(device_type="cuda",dtype=torch.bfloat16)
    def forward(self,wet_tone,dry_tone):
        x_wet = self.cnn(wet_tone)
        x_dry = self.cnn(dry_tone)
        x = torch.cat([x_wet,x_dry],dim=1)
        x = self.embed_1(x)
        x = self.embed_2(x)
        x = self.embed_3(x)
        return x
    

class EffectClassifier(torch.nn.Module):
    def __init__(self, n_classes, batch_size,embed_dim=768):
        super(EffectClassifier, self).__init__()
        self.cnn = CNN_cls(batch_size)
        self.mlp = MLP()
        self.attn = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=2, dropout=.1, batch_first=True)
        self.fc = torch.nn.Linear(embed_dim * 2, embed_dim)
        self.cls = torch.nn.Linear(embed_dim, n_classes)

    @torch.autocast(device_type="cuda",dtype=torch.bfloat16)
    def forward(self, x_wet, x_dry):
        x_wet = self.cnn(x_wet.unsqueeze(1))  # Adjust unsqueeze dimension
        x_dry = self.cnn(x_dry.unsqueeze(1))  # Adjust unsqueeze dimension
        x_wet = self.mlp(x_wet)
        x_dry = self.mlp(x_dry)
        x = torch.cat([x_wet, x_dry], dim=1)
        x, _ = self.attn(x, x, x)  # Unpack attn output
        x = self.cls(self.fc(x))
        return x


class ParameterPrediction2(torch.nn.Module):
    """
    This is a wrapper for the effect classifier and individual effect Parameter prediction models
    It is only intended to be used for inference only, each classifier and parameter prediciton model will be trained seperately
    """
    def __init__(self,metadata,batch_size,state_dict_paths={}):
        num_effects = len(metadata['effects_to_parameters'].keys())
        self.classifier = EffectClassifier(num_effects,batch_size)
        if "classifier" in state_dict_paths.keys():
            self.classifier.load_state_dict(state_dict_paths["classifier"])

        for effect,params in metadata['effects_to_parameters'].items():
            num_params = len(params.keys())
            setattr(self, f"{effect.lower()}_predictor", EffectParameterPredictor(num_params, batch_size))
            if effect in state_dict_paths.keys():
                state_dict = torch.load(state_dict_paths[effect])
                getattr(self, f"{effect.lower()}_predictor").load_state_dict(state_dict)
            break
        self.post_processor = PostProcessor(metadata)
        return
    @torch.no_grad
    def forward(self,wet_tone,dry_tone,dry_tone_path):
        effect = self.classifier(wet_tone,dry_tone)
        effect_name = self.metadata['index_to_effect'][int(torch.argmax(effect))]
        param_model = getattr(self,f"{effect_name.lower()}_predictor")
        params = param_model(wet_tone)
        out = self.post_processor.process_audio_from_outputs(effect,params,dry_tone_path)
        return out

class MinMaxScaler:
    def __init__(self, feature_min=None, feature_max=None, range_min=0.0, range_max=1.0):
        """
        Initializes the Min-Max Scaler.

        Args:
            feature_min (torch.Tensor, optional): Minimum values per feature. If None, computed from data.
            feature_max (torch.Tensor, optional): Maximum values per feature. If None, computed from data.
            range_min (float): Desired minimum of scaled data (default=0.0).
            range_max (float): Desired maximum of scaled data (default=1.0).
        """
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.range_min = range_min
        self.range_max = range_max

    def fit(self, X):
        """Computes and stores min-max values from the input data."""
        self.feature_min = X.min(dim=0, keepdim=True).values
        self.feature_max = X.max(dim=0, keepdim=True).values

    def transform(self, X):
        """Applies Min-Max scaling to the input tensor."""
        if self.feature_min is None or self.feature_max is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")

        # Avoid division by zero
        scale = self.feature_max - self.feature_min
        scale[scale == 0] = 1  # Prevents division by zero for constant features

        return self.range_min + (X - self.feature_min) / scale * (self.range_max - self.range_min)

    def fit_transform(self, X):
        """Fits the scaler and transforms the input data."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """Reverts the transformation back to the original scale."""
        if self.feature_min is None or self.feature_max is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")

        return self.feature_min + (X_scaled - self.range_min) / (self.range_max - self.range_min) * (self.feature_max - self.feature_min)



class Trainer():
    def __init__(self, model, metadata, lambda_=.5, sample_rate=16000):
        self.model = model
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractorTorch()
        self.sigmoid = torch.nn.Sigmoid()
        self.lambda_ = lambda_
        self.scaler = MinMaxScaler()
        return
    
    @torch.no_grad
    def eval_cls(self,model, loss_fn, dl, batch_size = 4):
        model.eval()
        total_loss = 0
        labels = []
        labels_ = []
        preds = []
        logits = []
        for batch in tqdm.tqdm(dl):
            wet_features = batch['wet_tone_features'].to(device)
            dry_features = batch['dry_tone_features'].to(device)
            label = batch['effects'].to(device)
            with torch.no_grad():
                logits_ = model(wet_features, dry_features)
            loss = loss_fn(logits_, label)
            total_loss += loss.item()
            for i in range(logits_.shape[0]):
                preds.append(torch.argmax(logits_[i], dim=0).cpu().numpy())
                labels.append(torch.argmax(label[i], dim=0).cpu().numpy())
                labels_.append(torch.nn.functional.one_hot(torch.argmax(label[i], dim=0), num_classes=5).cpu().numpy())
                logits.append(logits_[i].cpu().numpy())
        loss = total_loss / len(dl) * batch_size
        accuracy = accuracy_score(labels, preds)
        auroc = roc_auc_score(labels_, logits)
        print(f"Test: Accuracy:{accuracy} | AUROC: {auroc} | Avg Loss:{loss}")
        return loss, accuracy, auroc

    def train_cls(self,model, optimizer, loss_fn, train_loader,test_loader,lr_scheduler, epochs=10):
        model.train()
        best_accuracy = 0
        labels = []
        labels_ = []
        preds = []
        logits = []
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm.tqdm(train_loader):
                optimizer.zero_grad()
                wet_features = batch['wet_tone']['spectrogram'].to(device)
                dry_features = batch['dry_tone']['spectrogram'].to(device)
                label = batch['effects'].to(device)
                output = model(wet_features,dry_features)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                for i in range(output.shape[0]):
                    preds.append(torch.argmax(output[i], dim=0).detach().cpu().numpy())
                    labels.append(torch.argmax(label[i], dim=0).detach().cpu().numpy())
                    labels_.append(torch.nn.functional.one_hot(torch.argmax(label[i], dim=0), num_classes=5).detach().cpu().numpy())
                    logits.append(output[i].detach().cpu().numpy())
            print(f"Train: Epoch {epoch+1} | Accuracy: {accuracy_score(labels,preds)} | AUROC: {roc_auc_score(labels_,logits)} | Loss: {total_loss}")
            loss, accuracy, auroc = self.eval_cls(model, loss_fn, test_loader)
            lr_scheduler.step(loss)
            if accuracy > best_accuracy:
                print(f"saving model at epoch {epoch+1}")
                best_accuracy = accuracy
                torch.save(model.state_dict(), "saved_models/multiclass_model.pth")
        return
    def train_param_pred(self, model, train_loader, test_loader, loss_fn, optimizer, lr_scheduler,batch_size,effect_name,epochs):
        model.train()
        best_rmse = 9999999
        param_labels = []
        param_preds = []
        total_rmse = 0
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm.tqdm(train_loader):
                optimizer.zero_grad()
                wet_tone_feat = batch['wet_tone']
                dry_tone_feat = batch['dry_tone']
                # ,wet_tone_feat['loudness'].to(device),wet_tone_feat['f0'].to(device),dry_tone_feat['loudness'].to(device),dry_tone_feat['f0'].to(device)
                params = model(dry_tone_feat['spectrogram'].to(device),wet_tone_feat['spectrogram'].to(device))
                scaled_params = self.scaler.fit_transform(params)
                scaled_labels = self.scaler.fit_transform(batch['parameters'].to(device))
                params = params.squeeze(-1)
                loss = loss_fn(params,batch['parameters'].to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                for pred in params:
                    param_preds.append(pred.detach().cpu().numpy())
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
    def eval_param_pred(self,model, test_loader, loss_fn, batch_size,effect_name,epoch):
        param_labels = []
        param_preds = []
        total_loss = 0
        total_rmse = 0
        for batch in tqdm.tqdm(test_loader):
            wet_tone_feat = batch['wet_tone']
            dry_tone_feat = batch['dry_tone']
            params = model(dry_tone_feat['spectrogram'].to(device),wet_tone_feat['spectrogram'].to(device))
            scaled_labels = batch['parameters'].to(device)
            loss = loss_fn(params.squeeze(0),scaled_labels)
            total_loss += loss.item()
            for pred in params.squeeze(0):
                param_preds.append(pred.detach().cpu().numpy())
            for label in scaled_labels:
                param_labels.append(label.detach().cpu().numpy())
        r2 = r2_score(np.array(param_labels),np.array(param_preds))
        rmse_final = math.sqrt(total_loss / (len(test_loader) * batch_size))
        print(f"Test {effect_name}: Epoch {epoch+1} | Parameter R^2: {r2} | Parameter RMSE Loss: {rmse_final}")
        return rmse_final, r2
    @torch.no_grad
    def eval(self,model, test_loader, loss_fn_efct,loss_fn_params, epoch):
        model.eval()
        effect_labels = []
        effect_preds = []
        param_labels = []
        param_preds = []
        total_loss = 0
        total_rmse = 0
        for batch in tqdm.tqdm(test_loader):
            wet_tone_feat = batch['wet_tone']
            dry_tone_feat = batch['dry_tone']
            effect, params = model(wet_tone_feat['spectrogram'].to(device),dry_tone_feat['spectrogram'].to(device))
            loss,rmse = self.compute_loss(effect,params,batch['effects'].to(device),batch['parameters'].to(device),loss_fn_efct,loss_fn_params,self.lambda_)
            total_loss += loss.item()
            total_rmse += rmse
            for efct in batch['effects']:
                label = int(torch.argmax(efct).cpu())
                effect_labels.append(label)
            for efct in effect:
                label = int(torch.argmax(efct).cpu())
                effect_preds.append(label)
            for pred in params:
                param_preds.append(pred.detach().cpu().numpy())
            for label in batch['parameters']:
                param_labels.append(label.numpy())
        test_accuracy = accuracy_score(effect_labels,effect_preds)
        r2 = r2_score(np.array(param_labels),np.array(param_preds))
        rmse_final = math.sqrt(total_rmse / (len(test_loader) * model.batch_size))
        print(f"Test: Epoch {epoch+1} | Effect Accuracy: {test_accuracy} | Parameter R^2: {r2} | Parameter RMSE Loss: {rmse_final}")
        return rmse_final, test_accuracy, r2
    
    
    def compute_loss(self, output_effect, output_params, target_effect, target_params, loss_fn_efct, loss_fn_params, lambda_):
        
        output_params = torch.where(output_params != 0, self.sigmoid(output_params), output_params)
        target_params = torch.where(target_params != 0, self.sigmoid(target_params), target_params)
        loss_params = loss_fn_params(output_params,target_params)
        return loss_params
    
    def train(self, model, train_loader, test_loader, loss_fn_efct,loss_fn_params, optimizer, lr_scheduler, epochs):
        model.train()
        best_rmse = 9999999
        effect_labels = []
        effect_preds = []
        param_labels = []
        param_preds = []
        total_rmse = 0
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm.tqdm(train_loader):
                optimizer.zero_grad()
                wet_tone_feat = batch['wet_tone']
                dry_tone_feat = batch['dry_tone']
                # ,wet_tone_feat['loudness'].to(device),wet_tone_feat['f0'].to(device),dry_tone_feat['loudness'].to(device),dry_tone_feat['f0'].to(device)
                effect, params = model(wet_tone_feat['spectrogram'].to(device),dry_tone_feat['spectrogram'].to(device))
                
                loss, rmse = self.compute_loss(effect,params,batch['effects'].to(device),batch['parameters'].to(device),loss_fn_efct,loss_fn_params,self.lambda_)
                total_rmse += rmse
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                for efct in batch['effects']:
                    label = int(torch.argmax(efct).cpu())
                    effect_labels.append(label)
                for efct in effect:
                    label = int(torch.argmax(efct).cpu())
                    effect_preds.append(label)
                for pred in params:
                    param_preds.append(pred.detach().cpu().numpy())
                for label in batch['parameters']:
                    param_labels.append(label.numpy())
            r2 = r2_score(np.array(param_labels),np.array(param_preds))
            rmse_final = math.sqrt(total_rmse / (len(test_loader) * model.batch_size))
            print(f"Train: Epoch {epoch+1} | Effect Accuracy: {accuracy_score(effect_labels,effect_preds)} | Parameter R^2: {r2} | Parameter RMSE Loss: {rmse_final}")
            test_rmse, accuracy, r2 = self.eval(model, test_loader, loss_fn_efct,loss_fn_params, epoch)
            lr_scheduler.step(loss)
            if test_rmse < best_rmse:
                print(f"saving model at epoch {epoch+1}")
                best_rmse = test_rmse
                torch.save(model.state_dict(), "saved_models/parameter_prediction.pth")
        return

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