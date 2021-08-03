#%%
import soundfile as sf
import pyrubberband as pyrb

import os
import shutil
from glob import glob

# %%
filepaths = glob("/media/nao/hd8t/Datasets/Audio/Freesound/Freesound_caption/development/*.wav")
print(len(filepaths))

# dataset_dir = "/media/nao/hd8t/Datasets/Audio/youtube-5m-loops-flat"
# os.makedirs(dataset_dir, exist_ok=True)
# for path in filepaths:
#     filename = os.path.basename(path)
#     dest_path = os.path.join(dataset_dir, filename)
#     shutil.copy(path, dest_path)
# %%

SR = 44100
BPM = 120.0
NM_BEATS = 8

dataset_name = "freesound_caption_loop"

loop_output_path = "./dataset/%s/" % dataset_name
os.makedirs(loop_output_path, exist_ok=True)

target_sample_num = int(60/BPM * NM_BEATS * SR)
print(target_sample_num)

#%%
# extract loops
from loop_extraction.loopextractor import loopextractor
from tqdm import tqdm
from joblib import Parallel, delayed

# parallel process
def extract_loop(filepath):
    print(filepath)
    filename = os.path.basename(filepath)
    try:
        output_filepath = "%s/%s" % (loop_output_path, filename[:-4])
        loopextractor.run_algorithm(filepath, n_templates=[0, 0, 0], 
                            output_savename=output_filepath)
    except Exception as exp:
        print(exp)
Parallel(n_jobs=-1, verbose=4)(delayed(extract_loop)(path) for path in filepaths)

# #%%
# # trim silent parts in extracted loops 
# import soundfile as sf
# from glob import glob

# loop_output_path = "./dataset/%s_trim/" % dataset_name
# os.makedirs(loop_output_path, exist_ok=True)

# def trim_silience(filepath):
#     y, sr = librosa.load(filepath, sr=SR)
#     filename = os.path.basename(filepath)
#     new_filepath = os.path.join(loop_output_path, filename)
#     yt, index = librosa.effects.trim(y, top_db=40, frame_length=64, hop_length=32)
#     sf.write(new_filepath, yt, sr)   
# Parallel(n_jobs=-1, verbose=4)(delayed(trim_silience)(path) for path in filepaths)
    

#%%

# Timestetch to the target tempo and duration
import librosa

filepaths = glob(loop_output_path + "/*.wav")


def timestretch(path):
    y, sr = librosa.load(path, sr=SR) #, mono=True)

    # time stretch
    stretch_rate = y.shape[0] / target_sample_num 
    y_stretch = pyrb.time_stretch(y, sr, stretch_rate)

    # overwrite
    sf.write(path, y_stretch, sr)

Parallel(n_jobs=-1, verbose=4)(delayed(timestretch)(path) for path in filepaths)
#%%

from musicnn.tagger import top_tags

directories = {
    'rhythm': ['beat', 'drums', ],
    'melody': ['sitar', 'guitar', 'strings', 'flute', 'cello', 'harpsichord'],
    'eclectic': ['weird', 'techno', 'synth', 'new age'],
    'soundscape': ['ambient', 'slow', 'quiet'],
    'voice': ['vocal', 'male', 'female', 'man', 'choir']
}

def sort_files(path):
    tags = top_tags(path, topN = 3)

    for tag in tags:
        for k, v in directories.items():
            if tag in v:
                dirpath = os.path.join(loop_output_path, k)
                os.makedirs(dirpath, exist_ok=True)
                shutil.copy(path, dirpath)

Parallel(n_jobs=-1, verbose=4)(delayed(sort_files)(path) for path in filepaths)

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
