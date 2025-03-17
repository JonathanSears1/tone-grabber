from model.parameter_prediction import Trainer, ParameterPredictionResNet
from dataset.data_generator import DataGenerator
from pedalboard import Reverb, Delay, Chorus, Distortion, Gain, PitchShift, LowpassFilter, HighpassFilter
import torch
from torch.nn import MSELoss, CrossEntropyLoss
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision.models as models
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('high')

# with open("effects_to_parameters.json") as f:
#     effects_to_parameters = json.load(f)

# List of effects
effects = [Distortion, Gain, PitchShift, LowpassFilter, HighpassFilter]
effects_to_parameters = {
        "Gain": {
            "gain_db": [-60, 24]
        },
        "Distortion": {
            "drive_db": [0, 60]
        },
        "PitchShift": {
        "semitones": [-12, 12]
        },
        "HighpassFilter": {
        "cutoff_frequency_hz": [20, 20000]
        },
        "LowpassFilter": {
        "cutoff_frequency_hz": [20, 20000]
        }
    }
import json
import pandas as pd
try:
    with open('data/nsynth-train.jsonwav/nsynth-train/examples.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError as e:
    print("You need to download the NSynth dataset first, or change the path to the examples.json file.")
    raise e
df = pd.DataFrame.from_records(data)
df = df.T
guitar_df = df[df['instrument_family_str'] == 'guitar']
elctric_guitar_df = guitar_df[guitar_df['instrument_source_str'] == "electronic"]
elctric_guitar_df = elctric_guitar_df.sample(2000)
dry_tones = [dry_tone + ".wav" for dry_tone in elctric_guitar_df['note_str'].tolist()]
# max_chain_length is the maximum number of effects applied to a sample
train_data, test_data = train_test_split(dry_tones, test_size=0.3)
test_data, val_data = train_test_split(test_data, test_size=0.5)

max_chain_length = 1
# create instance of data generator corresponding to effects
for effect_name,parameter_dict in effects_to_parameters.items():
    effect_to_params = {effect_name:parameter_dict}
    effect = None
    for effect_ in effects:
        if effect_.__name__ == effect_name:
            effect = [effect_]
    generator = DataGenerator(effect_to_params, effect)
        # Check if datasets already exist
    datasets_exist = all(os.path.exists(f'data/{effect_name}_{split}_dataset.pt') 
                            for split in ['train', 'test', 'val'])
    if datasets_exist:
        print("Loading existing datasets from pt files...")
        # Load existing datasets
        train_dataset = torch.load(f'data/{effect_name}_train_dataset.pt',weights_only=False)
        test_dataset = torch.load(f'data/{effect_name}_test_dataset.pt',weights_only=False) 
        val_dataset = torch.load(f'data/{effect_name}_val_dataset.pt',weights_only=False)
        print("Datasets loaded successfully")
    else:
        print("Generating new datasets...")
        train_dataset = generator.create_data(2, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=train_data,max_chain_length=1)
        test_dataset = generator.create_data(2, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=test_data,max_chain_length=1)
        val_dataset = generator.create_data(2, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=val_data,max_chain_length=1)
        print("Saving datasets to pt files...")
        # Save datasets
        torch.save(train_dataset, f'data/{effect_name}_train_dataset.pt')
        torch.save(test_dataset, f'data/{effect_name}_test_dataset.pt')
        torch.save(val_dataset, f'data/{effect_name}_val_dataset.pt')
        print("Datasets saved successfully")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
        #set the model
    metadata = generator.get_metadata()
    param_mask = metadata['parameter_mask_idx']
    num_parameters = metadata['total_parameters']
    num_effects = len(metadata['effect_to_idx'].keys())
    model = ParameterPredictionResNet(768,len(parameter_dict.values())).to(device)
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(),.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    trainer = Trainer(model, metadata,0,1)
    model = model.to(device)
    model.compile()
    trainer.train_param_pred(model,train_loader,test_loader,loss_fn,optimizer,scheduler,4,effect_name,epochs=20)
    trainer.eval_param_pred(model,val_loader,loss_fn,4,effect_name,epoch=0)

    del train_dataset
    del train_loader

    del test_dataset
    del test_loader

    del val_dataset
    del val_loader