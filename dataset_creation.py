#%%
import soundfile as sf
import pyrubberband as pyrb

import shutil
from glob import glob

# %%
import os

filepaths = glob("/media/nao/hd8t/Datasets/Audio/youtube-5m-loops/*/*.wav")
print(len(filepaths))

# dataset_dir = "/media/nao/hd8t/Datasets/Audio/youtube-5m-loops-flat"
# os.makedirs(dataset_dir, exist_ok=True)
# for path in filepaths:
#     filename = os.path.basename(path)
#     dest_path = os.path.join(dataset_dir, filename)
#     shutil.copy(path, dest_path)
# %%
import os
import librosa

SR = 16000
BPM = 120.0
NM_BEATS = 8

dataset_name = "youtube-5m-loops"
#%%
target_sample_num = int(60/BPM * NM_BEATS * SR)
print(target_sample_num)

for path in filepaths:
    dirname = path.split("/")[-2]
    filename = path.split("/")[-1]

    os.makedirs("./dataset/%s/%s" % (dataset_name, dirname), exist_ok=True)

    y, sr = librosa.load(path, sr=SR, mono=True)

    # time stretch
    stretch_rate = y.shape[0] / target_sample_num 
    y_stretch = pyrb.time_stretch(y, sr, stretch_rate)

    new_path = "./dataset/%s/%s/%s" % (dataset_name, dirname, filename)
    sf.write(new_path, y_stretch, sr)
    print(y_stretch.shape)

    break

# %%
import torch
import torchaudio
import numpy as np
import scipy
import os
from IPython.display import Audio, display
from openunmix import predict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
# %%

filepaths = glob("./dataset/%s/*/*.wav" % dataset_name)
print(len(filepaths))

 
# %%
from tqdm import tqdm

for path in tqdm(filepaths):
    y, sr = librosa.load(path, sr=44100, mono=True)

    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    dir_path = os.path.dirname(path)
    new_dir_path = os.path.join(dir_path, filename)
    os.makedirs(new_dir_path, exist_ok=True)

    sf.write(os.path.join(new_dir_path, "original.wav"), y, sr)

    estimates = predict.separate(
        torch.as_tensor(y).float(),
        rate = sr,
        device=device
    )   
    for target, estimate in estimates.items():
        indv_filename = target + ".wav"
        indv_filepath = os.path.join(dir_path, filename, indv_filename)
        audio = estimate.detach().cpu().numpy()[0]
        sf.write(indv_filepath, np.transpose(audio), sr)


# %%
