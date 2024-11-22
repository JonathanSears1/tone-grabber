import torch
import numpy as np
from torch.utils.data import Dataset

class Effect():
    def __init__(self, effect_idx, parameters, total_effects, effect_name, total_params, param_idxs, order):
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
        self.order = order
        return
    
    def __dict__(self):
        return {"effect": self.effect_name,"one_hot":self.one_hot, "parameters": self.param_repr,"order":self.order}
        
    
    
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
        # Keep track of the names of each effect
        self.names = [effect.effect_name for effect in effects]
        self.dry_tone_path = dry_tone_path
        self.wet_tone_path = wet_tone_path
        return
    
    def __len__(self):
        return len(self.effects)
    def __dict__(self):
        return {"effects":self.effects, "parameters":self.parameters, "dry_tone_path":self.dry_tone_path, "wet_tone_path":self.wet_tone_path}

class EffectChainDataset(Dataset):
    def __init__(self, data, dry_tone_features=True):
        '''
        Pass in a list of EffectsChain objects
        '''
        self.data = data
        self.dry_tone_features = dry_tone_features
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
        wet_tone_features = entry['wet_tone_features']
        names = entry['names']
        effects = entry['effects']
        parameters = entry['parameters']
        if self.dry_tone_features:
            dry_tone_feat = entry['dry_tone_features']
            return {"dry_tone_path":dry_tone_path,"wet_tone_path":wet_tone_path,"wet_tone_features":wet_tone_features,"dry_tone_features":dry_tone_feat,"effect_names":names,"effects":effects, "parameters":parameters, "index":idx}
        else:
            return {"dry_tone_path":dry_tone_path,"wet_tone_path":wet_tone_path,"wet_tone_features":wet_tone_features,"effect_names":names,"effects":effects, "parameters":parameters, "index":idx}