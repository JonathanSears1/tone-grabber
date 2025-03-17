import torch
import numpy as np
from torch.utils.data import Dataset

class Effect():
    def __init__(self, effect_idx, parameters, total_effects, effect_name, total_params, param_idxs,param_dict, order):
        # Create a representation of the effect
        # maximum amount of effects that can be used in the effects chain - also the length of the one hot encoding
        if effect_idx == -1:
            self.one_hot = torch.zeros(total_effects, dtype=torch.float32)
            self.param_repr = torch.zeros(total_params, dtype=torch.float32)
            self.effect_name = "None"
            return
        self.total_effects = total_effects
        # Create a one hot encoding of the effect
        one_hot = [0] * self.total_effects
        #effect_idx comes from a predefined 
        one_hot[effect_idx] = 1
        self.one_hot = torch.tensor(one_hot, dtype=torch.float32)
        # Create a representation of the parameters
        # Total number of parameters = sum of the number of parameters for each effect
        self.total_params = total_params
        self.param_repr = [0] * total_params
        # the location of each effect's parameters will be predefined with param_idxs
        self.param_repr[param_idxs[0]:param_idxs[-1]+1] = parameters
        self.param_repr = torch.tensor(self.param_repr, dtype=torch.float32)
        # Store the effect name
        self.effect_name = effect_name
        self.param_dict = param_dict
        self.order = order
        return
    
    def __dict__(self):
        return {"effect": self.effect_name,"one_hot":self.one_hot, "parameters": self.param_repr,"param_dict":self.param_dict,"order":self.order}
        
    
    
class EffectsChain():
    def __init__(self, effects, dry_tone_path, wet_tone_path, total_effects,max_chain_length,total_params):
        assert len(effects) <= total_effects, "The number of effects in the chain must be less than or equal to the maximum number of effects"
        # # pad effects chain to be the length of total_effects
        if len(effects) < max_chain_length:
            for i in range(max_chain_length-len(effects)):
                effects.append(Effect(-1, None, total_effects, "None", total_params, None,len(effects) + i))
        # Tensors of shape len(effects) X max_chain_length
        self.effects = torch.stack([effect.one_hot for effect in effects])
        self.parameters = torch.stack([effect.param_repr for effect in effects])
        self.param_dicts = [effect.param_dict for effect in effects]
        # Keep track of the names of each effect
        self.names = [effect.effect_name for effect in effects]
        self.dry_tone_path = dry_tone_path
        self.wet_tone_path = wet_tone_path
        return
    
    def __len__(self):
        return len(self.effects)
    def __dict__(self):
        return {"effects":self.effects, "parameters":self.parameters,"param_dict":self.param_dicts, "dry_tone_path":self.dry_tone_path, "wet_tone_path":self.wet_tone_path}

class EffectChainDataset(Dataset):
    def __init__(self, data,parameters=True, loudness_and_f0=False):
        '''
        Pass in a list of EffectsChain objects
        '''
        self.data = data
        self.loudness_and_f0=loudness_and_f0
        self.parameters = parameters
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Return the wet tone data at a given index
        '''
        entry = self.data[idx]
        dry_tone_path = entry['dry_tone_path']
        wet_tone_path = entry['wet_tone_path']
        wet_tone_spectrogram = entry['wet_tone_spectrogram']
        dry_tone_spectrogram = entry['dry_tone_spec']
        names = entry['names']
        effects = entry['effects']
        joint_spectrogram = np.concatenate((dry_tone_spectrogram,wet_tone_spectrogram),axis=0)
        
        if self.loudness_and_f0:
            wet_tone_loudness = entry['wet_tone_loudness']
            wet_tone_f0 = entry['wet_tone_f0']
            dry_tone_loudness = entry['dry_tone_loudness']
            dry_tone_f0 = entry['dry_tone_f0']

            dry_tone = {
                "spectrogram":dry_tone_spectrogram,
                "loudness":dry_tone_loudness,
                "f0":dry_tone_f0,
                "path":dry_tone_path
            }
            wet_tone = {
                "spectrogram":wet_tone_spectrogram,
                "loudness":wet_tone_loudness,
                "f0":wet_tone_f0
                }
        else:
            dry_tone = {
                "spectrogram":dry_tone_spectrogram,
                "path":dry_tone_path
                }
            wet_tone = {
                "spectrogram":wet_tone_spectrogram,
                }
        if self.parameters:
            parameters = entry['parameters']
            param_dict = entry['param_dict']
            return {"dry_tone":dry_tone,"wet_tone":wet_tone,"effect_names":names,"effects":effects,"parameters":parameters,"joint_spectrogram":joint_spectrogram,"param_dict":param_dict}
        
                
        return {"dry_tone":dry_tone,"wet_tone":wet_tone,"effect_names":names,"joint_spectrogram":joint_spectrogram,"effects":effects}
        