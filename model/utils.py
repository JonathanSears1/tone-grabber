import torch
from dataset.feature_extractor_torch import FeatureExtractorTorch
import tqdm
from sklearn.metrics import r2_score
from pedalboard.io import ReadableAudioFile
import numpy as np
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, model, metadata, sample_rate=16000):
        self.model = model
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractorTorch()
        self.sigmoid = torch.nn.Sigmoid()
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