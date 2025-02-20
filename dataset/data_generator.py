from pedalboard import Pedalboard, Delay, Gain, Chorus, Reverb, Distortion, Compressor, Mix, Phaser, NoiseGate, PitchShift, PeakFilter, LowpassFilter, LowShelfFilter, Limiter, LadderFilter, IIRFilter, HighpassFilter, HighShelfFilter, GSMFullRateCompressor, Convolution, Clipping, Invert
from pedalboard.io import ReadableAudioFile
import random
import pickle
import tqdm
from transformers import AutoFeatureExtractor
import os
from dataset.dataset import Effect, EffectsChain, EffectChainDataset
from dataset.feature_extractor_torch import FeatureExtractorTorch
#from dataset.tf_dataset import TFEffectChainDataset, TFEffectsChain, TFEffect
#from model.feature_extractor import FeatureExtractor
import numpy as np
from scipy.stats import truncnorm
# Create a list of all the effects
# effects = [Delay, Gain, Chorus, Reverb, Distortion, Compressor, Phaser, NoiseGate, PitchShift, PeakFilter, LowpassFilter, LowShelfFilter, Limiter, LadderFilter, HighpassFilter, HighShelfFilter, Clipping]
#import tensorflow as tf
# Create -a mapping of effect names to their parameters and max/min values of said parameters
#TODO: adjust max min values to more accurate values

class DataGenerator():
    def __init__(self, 
                 effects_to_parameters: dict,
                 effects: list,
                 loudnees_and_f0 = False) -> None:
        self.effects_to_parameters = effects_to_parameters
        # calculate the total number of possible parameters
        total_parameters = 0
        for effect, params in effects_to_parameters.items():
            total_parameters += len(params)
        self.total_parameters = total_parameters
        # Create a dictionary to store the indices of the parameters for each effect
        effect_to_param_indices = {}
        current_index = 0
        param_mask = {}
        for effect, params in effects_to_parameters.items():
            num_params = len(params)
            if num_params > 0:
                idxs = list(range(current_index, current_index + num_params))
                effect_to_param_indices[effect] = idxs
                param_mask[effect] = [1 if i in idxs else 0 for i in range(total_parameters)]
                current_index += num_params
        self.effect_to_param_indices = effect_to_param_indices
        self.param_mask = param_mask
        # map each effect to a one hot encoding index
        self.effect_to_index = {effect.__name__: i for i, effect in enumerate(effects)}
        self.index_to_effect = {i: effect.__name__  for i, effect in enumerate(effects)}
        self.loudness_and_f0 = loudnees_and_f0
        self.effects = effects
        return
    def create_data(self,
                    num_samples: int, 
                    dry_tone_dir: str,
                    dry_tones: list, 
                    max_chain_length: int,
                    sample_rate=16000,
                    dry_tone_features=True
                    ) -> EffectChainDataset:
        # Create an empty list to store the data
        self.feature_extractor = FeatureExtractorTorch(loudness_and_f0=self.loudness_and_f0,sample_rate=sample_rate)
        data = []
        for dry_tone_path in tqdm.tqdm(dry_tones):
            name,ext = dry_tone_path.split('.')
            dry_tone_path = os.path.join(dry_tone_dir, dry_tone_path)
            with ReadableAudioFile(dry_tone_path) as f:
                # re sample the audio file to match the sample rate, pretrained model is sampled at 16000
                re_sampled = f.resampled_to(sample_rate)
                dry_tone = np.squeeze(re_sampled.read(int(sample_rate * f.duration)),axis=0)
                re_sampled.close()
                f.close()
            # Loop over the number of samples
            
            dry_tone_feat = self.feature_extractor.get_features(dry_tone)
            dry_tone_spec = dry_tone_feat['spectrogram']
            if self.loudness_and_f0:
                dry_tone_loudness = dry_tone_feat['loudness']
                dry_tone_f0 = dry_tone_feat['f0']
            for i in range(num_samples):
                wet_tone_data = {}
                wet_tone_data['dry_tone_path'] = dry_tone_path
                wet_tone_data['wet_tone_path'] = f'data/wet_tones/{name}_wet_{i}.{ext}'
                # Create a new pedalboard
                pedalboard = Pedalboard()
                # Randomly select a number of effects to add to the pedalboard
                num_effects = random.randint(1, max_chain_length)
                # Create a dictionary to the effects used and their parameters
                effect_list = []
                for j in range(num_effects):
                    # Randomly select an effect to add
                    effect = random.choice(self.effects)
                    # Get the effect name
                    effect_name = effect.__name__
                    # Get the effect parameters
                    parameters = self.effects_to_parameters[effect_name].keys()
                    # Create a dictionary to store the effect data (parameters and order)
                    effect_data = {}
                    # Loop over the parameters
                    parameter_values = []
                    for param in parameters:
                        # Randomly select a value for the parameter with truncated normal dsitribution
                        min_val,max_val = self.effects_to_parameters[effect_name][param]
                        mean = (min_val + max_val) / 2
                        std_dev = (max_val - min_val) / 4
                        # Calculate a and b
                        a, b = (min_val - mean) / std_dev, (max_val - mean) / std_dev
                        value = truncnorm.rvs(a,b,loc=mean,scale=std_dev)
                        # Add the parameter to the dictionary
                        effect_data[param] = value
                        parameter_values.append(value)
                    # Create a new effect with the parameters
                    new_effect = effect(**effect_data)
                    # Add the effect to the pedalboard
                    pedalboard.append(new_effect)
                    # Add the effect and corresponding params to the dictionary
                    effect_list.append(Effect(self.effect_to_index[effect_name],parameter_values,len(self.effects),effect_name,self.total_parameters,self.effect_to_param_indices[effect_name],effect_data,j))
                effect_chain = EffectsChain(effect_list, dry_tone_path, wet_tone_data['wet_tone_path'], len(self.effects),max_chain_length, self.total_parameters)
                wet_tone = pedalboard(dry_tone, sample_rate * f.duration)
                # we don't need to save the actual wet tone because it can be recreated with the dry tone + effect data
                wet_tone_features = self.feature_extractor.get_features(wet_tone)

                wet_tone_data['wet_tone_spectrogram'] = wet_tone_features['spectrogram']
                if self.loudness_and_f0:
                    wet_tone_data['wet_tone_loudness'] = wet_tone_features['loudness']
                    wet_tone_data['wet_tone_f0'] = wet_tone_features['f0']
                    wet_tone_data['dry_tone_loudness'] = dry_tone_loudness
                    wet_tone_data['dry_tone_f0'] = dry_tone_f0
                
                wet_tone_data['effects'] = effect_chain.effects.squeeze(0)
                wet_tone_data['parameters'] = effect_chain.parameters.squeeze(0)
                wet_tone_data['names'] = effect_chain.names
                wet_tone_data['dry_tone_spec'] = dry_tone_spec
                wet_tone_data['param_dict'] = effect_chain.param_dicts
                # Append the data to the list
                data.append(wet_tone_data)
        # Return the data
        dataset = EffectChainDataset(data)
        return dataset
    def get_metadata(self):
        param_mask_idx = {self.effect_to_index[k]:v for k,v in self.param_mask.items()}
        return {
                "parameter_mask_str": self.param_mask,
                "parameter_mask_idx":param_mask_idx,
                "effect_to_idx": self.effect_to_index,
                "index_to_effect":self.index_to_effect,
                "effects": self.effects,
                "total_parameters":self.total_parameters,
                "effects_to_parameters": self.effects_to_parameters
                }