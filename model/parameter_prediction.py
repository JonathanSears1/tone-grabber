import torch
import auraloss
import tqdm
from pedalboard import Pedalboard
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from pedalboard.io import ReadableAudioFile
import numpy as np
from dataset.feature_extractor_torch import FeatureExtractorTorch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(512,1024, kernel_size=(3, 3), stride=(1, 1),padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.AdaptiveAvgPool2d((1,1))
        )
        self.batch_size = batch_size
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
    def __init__(self, num_effects : int, num_parameters : int, parameter_mask : dict,batch_size=4, num_heads = 2, dropout = .1, embed_dim=768):
        super(ParameterPrediction, self).__init__()
        self.parameter_mask = parameter_mask
        self.cnn = CNN(4)
        #self.loudness_embedding = MLP(257,128)
        #self.f0_embedding = F0_Embed(21912,embed_dim)
        #self.mlp_feat = MLP(embed_dim*4, embed_dim*2)
        self.mlp_1 = MLP(1024,embed_dim)
        self.self_attn = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=num_heads, dropout=dropout, batch_first=True)
        #self.cross_attn = torch.nn.MultiheadAttention(embed_dim * 2, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mlp_2 = MLP(embed_dim*2,embed_dim)
        self.cls = torch.nn.Linear(embed_dim, num_effects)
        self.mlp_3 = MLP(embed_dim*2,embed_dim)
        self.mlp_4 = MLP(embed_dim,embed_dim // 2)
        self.mlp_5 = MLP(embed_dim // 2, embed_dim // 4)
        self.parameter_pred = MLP(embed_dim // 4,num_parameters)
        self.batch_size = batch_size

    def forward(self, x_wet, x_dry, loudness_wet=None, f0_wet=None, loudness_dry=None, f0_dry=None):
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
        # loudness_embed_wet = self.loudness_embedding(loudness_wet).squeeze(1)
        # f0_embed_wet = self.f0_embedding(f0_wet)
        # features_wet = torch.cat([f0_embed_wet,loudness_embed_wet],dim=1)
        # # dry features
        # loudness_embed_dry = self.loudness_embedding(loudness_dry).squeeze(1)
        # f0_embed_dry = self.f0_embedding(f0_dry)
        # features_dry = torch.cat([f0_embed_dry,loudness_embed_dry],dim=1)
        # features = torch.cat([features_wet,features_dry],dim=1)
        # features = self.mlp_feat(features)
        # x, _ = self.cross_attn(x,features,x)
        effect = self.cls(self.mlp_2(x))
        effect_indices = torch.argmax(effect, dim=1)  # Get predicted effect index for each item in batch
        mask = torch.stack([torch.tensor(self.parameter_mask[idx.item()]) for idx in effect_indices]).to(device)
        params = self.parameter_pred(self.mlp_5(self.mlp_4(self.mlp_3(x))))
        params = params * mask
        return x, effect, params

class Trainer():
    def __init__(self, model, metadata, sample_rate=16000):
        self.model = model
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractorTorch()
        self.softmax = torch.nn.Softmax(dim=1)
        return
    
    @torch.no_grad
    def eval(self,model, test_loader, loss_fn_efct,loss_fn_params, epoch):
        model.train()
        effect_labels = []
        effect_preds = []
        total_loss = 0
        for batch in tqdm.tqdm(test_loader):
            wet_tone_feat = batch['wet_tone']
            dry_tone_feat = batch['dry_tone']
            # ,wet_tone_feat['loudness'].to(device),wet_tone_feat['f0'].to(device),dry_tone_feat['loudness'].to(device),dry_tone_feat['f0'].to(device)
            out, effect, params = model(wet_tone_feat['spectrogram'].to(device),dry_tone_feat['spectrogram'].to(device))
            # predicted_wet_tones = []
            # true_wet_tones = []
            # for i, (effect_, params_) in enumerate(zip(effect, params)):
            #     predicted_wet_tones.append(self.process_audio_from_outputs(effect_,params_,dry_tone_feat['path'][i]))
            # for i, (effect_, params_) in enumerate(zip(batch['effects'], batch['parameters'])):
            #     true_wet_tones.append(self.process_audio_from_outputs(effect_,params_,dry_tone_feat['path'][i]))
            # target_wet_tones = torch.stack(true_wet_tones).to(device)
            # predicted_wet_tones = torch.stack(predicted_wet_tones).to(device)
            loss = self.compute_loss(effect,params,batch['effects'].to(device),batch['parameters'].to(device),loss_fn_efct,loss_fn_params)
            total_loss += loss.item()
            for efct in batch['effects']:
                label = int(torch.argmax(efct).cpu())
                effect_labels.append(label)
            for efct in effect:
                label = int(torch.argmax(efct).cpu())
                effect_preds.append(label)
        test_accuracy = accuracy_score(effect_labels,effect_preds)
        print(f"Test: Epoch {epoch+1} | Effect Accuracy: {test_accuracy} | Parameter MSE Loss: {total_loss}")
        return total_loss, test_accuracy
    
    
    def compute_loss(self, output_effect, output_params, target_effect, target_params, loss_fn_efct, loss_fn_params, lambda_=.5):
        loss_efct = loss_fn_efct(output_effect,target_effect)
        output_params = self.softmax(output_params)
        target_params = self.softmax(target_params)
        loss_params = loss_fn_params(output_params,target_params)
        return lambda_ * loss_efct + (1-lambda_) * loss_params
    
    def train(self, model, train_loader, test_loader, loss_fn_efct,loss_fn_params, optimizer, lr_scheduler, epochs):
        # compile model for faster training speeds
        # try:
        #     model = torch.compile(model)
        # except Exception as e:
        #     print(f"Model compilation failed, using uncompiled model, speeds may be slower:\n {e}")
        model.train()
        best_loss = 9999999999
        effect_labels = []
        effect_preds = []
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm.tqdm(train_loader):
                optimizer.zero_grad()
                wet_tone_feat = batch['wet_tone']
                dry_tone_feat = batch['dry_tone']
                # ,wet_tone_feat['loudness'].to(device),wet_tone_feat['f0'].to(device),dry_tone_feat['loudness'].to(device),dry_tone_feat['f0'].to(device)
                out, effect, params = model(wet_tone_feat['spectrogram'].to(device),dry_tone_feat['spectrogram'].to(device))
                # predicted_wet_tones = []
                # true_wet_tones = []
                # for i, (effect_, params_) in enumerate(zip(effect, params)):
                #     predicted_wet_tones.append(self.process_audio_from_outputs(effect_,params_,dry_tone_feat['path'][i]))
                # for i, (effect_, params_) in enumerate(zip(batch['effects'], batch['parameters'])):
                #     true_wet_tones.append(self.process_audio_from_outputs(effect_,params_,dry_tone_feat['path'][i]))
                # target_wet_tones = torch.stack(true_wet_tones).to(device)
                # predicted_wet_tones = torch.stack(predicted_wet_tones).to(device)
                loss = self.compute_loss(effect,params,batch['effects'].to(device),batch['parameters'].to(device),loss_fn_efct,loss_fn_params)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                for efct in batch['effects']:
                    label = int(torch.argmax(efct).cpu())
                    effect_labels.append(label)
                for efct in effect:
                    label = int(torch.argmax(efct).cpu())
                    effect_preds.append(label)

            print(f"Train: Epoch {epoch+1} | Effect Accuracy: {accuracy_score(effect_labels,effect_preds)} | Parameter MSE Loss: {total_loss}")
            loss, accuracy = self.eval(model, test_loader, loss_fn_efct,loss_fn_params, epoch)
            lr_scheduler.step(loss)
            if loss < best_loss:
                print(f"saving model at epoch {epoch+1}")
                best_loss = loss
                torch.save(model.state_dict(), "saved_models/parameter_prediction.pth")
        return
    
def process_audio_from_outputs(self, effect, params, dry_tone_path):
        with ReadableAudioFile(dry_tone_path) as f:
            re_sampled = f.resampled_to(self.sample_rate)
            dry_tone = np.squeeze(re_sampled.read(int(self.sample_rate * f.duration)),axis=0)
            re_sampled.close()
            f.close()
        predicted_effect_pb = self.metadata['effects'][int(torch.argmax(effect))]
        predicted_params = [float(param) for param in list(params.detach()) if param != 0]
        param_names = self.metadata['effects_to_parameters'][self.metadata['index_to_effect'][int(torch.argmax(effect))]].keys()
        matched_params = {param_name:value for param_name,value in zip(param_names,predicted_params)}
        predicted_effect_with_params = predicted_effect_pb(**matched_params)
        
        predicted_wet = predicted_effect_with_params.process(dry_tone,self.sample_rate)
        return predicted_wet