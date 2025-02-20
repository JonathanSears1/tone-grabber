from dataset.data_generator import DataGenerator
from pedalboard import Reverb, Delay, Chorus, Distortion, Gain
import torch
from torch.nn import MSELoss, CrossEntropyLoss
import os
from model.parameter_prediction import ParameterPrediction
from model.parameter_prediction import Trainer
from auraloss.freq import MultiResolutionSTFTLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch._dynamo.config.capture_scalar_outputs = True

effects_to_parameters = {
    "Reverb": {
        "room_size": (0, 1),
        "damping": (0, 1), 
        "wet_level": (0, 1),
        "dry_level": (0, 1),
        "width": (0, 1),
        "freeze_mode": (0, 1)
    },
    "Delay": {
        "delay_seconds": (0, 2),
        "feedback": (0, 1),
        "mix": (0, 1)
    },
    "Chorus": {
        "rate_hz": (0, 100),
        "depth": (0, 1),
        "centre_delay_ms": (1, 30),
        "feedback": (0, 1),
        "mix": (0, 1)
    },
    "Distortion": {
        "drive_db": (0, 100)
    },
    "Gain": {
        "gain_db": (-12, 12)
    }
}
# List of effects
effects = [Reverb, Delay, Distortion, Gain, Chorus]

# create instance of data generator corresponding to effects
generator = DataGenerator(effects_to_parameters, effects)

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
num_samples = 4
# max_chain_length is the maximum number of effects applied to a sample
train_data, test_data = train_test_split(dry_tones, test_size=0.2)
test_data, val_data = train_test_split(test_data, test_size=0.5)

max_chain_length = 1
#set the model
metadata = generator.get_metadata()
param_mask = metadata['parameter_mask_idx']
num_parameters = metadata['total_parameters']
num_effects = len(metadata['effect_to_idx'].keys())
#model = ParameterPrediction(num_effects,num_parameters,param_mask,batch_size=8,num_heads=8)
model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(
    in_channels=1,  # Change from 3 to 1
    out_channels=model.conv1.out_channels,  # Keep the same number of output channels
    kernel_size=model.conv1.kernel_size,
    stride=model.conv1.stride,
    padding=model.conv1.padding,
    bias=model.conv1.bias is not None  # Keep bias settings the same
)
# Add adaptive pooling before the fully connected layer
model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
model.fc = torch.nn.Linear(model.fc.in_features,1)


# Check if datasets already exist
datasets_exist = all(os.path.exists(f'data/{split}_dataset.pt') 
                    for split in ['train', 'test', 'val'])

if datasets_exist:
    print("Loading existing datasets from pt files...")
    # Load existing datasets
    train_dataset = torch.load('data/train_dataset.pt',weights_only=False)
    test_dataset = torch.load('data/test_dataset.pt',weights_only=False) 
    val_dataset = torch.load('data/val_dataset.pt',weights_only=False)
    print("Datasets loaded successfully")
else:
    print("Generating new datasets...")
    train_dataset = generator.create_data(10, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=train_data,max_chain_length=1)
    test_dataset = generator.create_data(10, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=test_data,max_chain_length=1)
    val_dataset = generator.create_data(10, 'data/nsynth-train.jsonwav/nsynth-train/audio',dry_tones=val_data,max_chain_length=1)
    print("Saving datasets to pt files...")
    # Save datasets
    torch.save(train_dataset, 'data/train_dataset.pt')
    torch.save(test_dataset, 'data/test_dataset.pt')
    torch.save(val_dataset, 'data/val_dataset.pt')
    print("Datasets saved successfully")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
loss_fn_effect = CrossEntropyLoss()
loss_fn_params = MSELoss()
optimizer = Adam(model.parameters(),.000005)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
trainer = Trainer(model, metadata,lambda_=.5)
model = model.to(device)
#model.compile()
