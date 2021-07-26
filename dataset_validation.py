#%%
import essentia
import essentia.standard as es

import os
from glob import glob
# %%

dataset_name = "youtube-5m-loops"

dirpaths = glob("./dataset/%s/*/*/" % dataset_name)
print(len(dirpaths))

# %%

def get_gain_and_bpm(filepath):
    features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                                rhythmStats=['mean', 'stdev'])(filepath)

    # print("Filename:", features['metadata.tags.file_name'])
    # print("-"*80)
    # print("Replay gain:", features['metadata.audio_properties.replay_gain'])
    # print("EBU128 integrated loudness:", features['lowlevel.loudness_ebu128.integrated'])
    # print("EBU128 loudness range:", features['lowlevel.loudness_ebu128.loudness_range'])
    # print("-"*80)
    # print("MFCC mean:", features['lowlevel.mfcc.mean'])
    # print("-"*80)
    # print("BPM:", features['rhythm.bpm'])
    # print("Beat positions (sec.)", features['rhythm.beats_position'])
    # print("-"*80)
    # print("Key/scale estimation (using a profile specifically suited for electronic music):",
    #     features['tonal.key_edma.key'], features['tonal.key_edma.scale'])
    return  features['lowlevel.loudness_ebu128.integrated'],  features['rhythm.bpm']
                                            

#%%
import shutil
from tqdm import tqdm 

selected_path = "./dataset/youtube-5m-loops-valid/"
os.makedirs(selected_path, exist_ok=True)

ignore_path = "./dataset/youtube-5m-loops-ignore/"
os.makedirs(ignore_path, exist_ok=True)

LOUDNESS_THRESHOLD = -32

for dirpath in tqdm(dirpaths):

    dir_name = os.path.dirname(dirpath)
    dir_name = os.path.basename(dir_name)

    try:
        drum_path = os.path.join(dirpath, 'drums.wav')
        bass_path = os.path.join(dirpath, 'bass.wav')
        other_path = os.path.join(dirpath, 'other.wav')
        org_path = os.path.join(dirpath, 'original.wav')

        if os.path.exists(drum_path) is False:
            continue

        drums_gain, _ = get_gain_and_bpm(drum_path)
        bass_gain, _ = get_gain_and_bpm(bass_path)
        other_gain, _ = get_gain_and_bpm(other_path)

        if drums_gain > LOUDNESS_THRESHOLD and bass_gain > LOUDNESS_THRESHOLD and other_gain > LOUDNESS_THRESHOLD:
            print(dirpath)
            new_dirpath = os.path.join(selected_path, dir_name)
            if os.path.exists(new_dirpath):
                shutil.rmtree(new_dirpath)
            shutil.move(dirpath, selected_path)
        else:            
            new_dirpath = os.path.join(ignore_path, dir_name)
            if os.path.exists(new_dirpath):
                shutil.rmtree(new_dirpath)
            shutil.move(dirpath, ignore_path)
    except Exception as exp:
        print("error", exp, dirpath)
        new_dirpath = os.path.join(ignore_path, dir_name)
        print(new_dirpath)
        if os.path.exists(new_dirpath):
            shutil.rmtree(new_dirpath)
        shutil.move(dirpath, ignore_path)

# %%
<<<<<<< HEAD
dirs = glob(selected_path+"/train/*/")
len(dirs)

# %%
import random 

VAL_SPLIT = 0.08 
NUM_VAL = int(len(dirs) * VAL_SPLIT)
random.shuffle(dirs)

for dirpath in dirs[:NUM_VAL]:
    shutil.move(dirpath, selected_path+'/val')
# %%
=======

# %%
>>>>>>> origin/master
