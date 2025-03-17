from model.classifier import EffectClassifier
from model.parameter_prediction import ParameterPredictionResNet, PostProcessor
import torch

def predict(dry_tone, wet_tone, classifier, postprocessor,parameter_prediction_dict, metadata):
    effect = classifier(dry_tone, wet_tone)
    effect_idx = torch.argmax(effect)
    effect_name = metadata['index_to_effect'][effect_idx]
    param_model = parameter_prediction_dict[effect_name]
    param_model.eval()
    params = param_model(dry_tone, wet_tone)
    return effect_name, params

