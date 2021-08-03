#%%

import loopextractor
# %%


from loopextractor import loopextractor
from glob import glob
import os
#from musicnn.tagger import top_tags
import librosa
import shutil
from tqdm import tqdm_notebook

# dirpath = '/Volumes/Berlin/_Sound_dataset/Freesound_caption/development'
# out_dirpath = '/Volumes/Berlin/_Sound_dataset/Freesound_caption/loops/'  

dirpath = '/media/nao/hd8t/Datasets/Audio/youtube-5m'
out_dirpath_root = '/media/nao/hd8t/Datasets/Audio/youtube-5m-loops'  
#categories_dirpath = '/media/nao/london/loop_remix/categories/'  

os.makedirs(out_dirpath_root, exist_ok=True)


# %%
for filepath in tqdm_notebook(glob("%s/*/*.wav" % dirpath)):
    filename = os.path.basename(filepath)
 
    dirname = filepath.split("/")[-2]    
   
    outpath = "%s/%s" % (out_dirpath_root, dirname)

    # if os.path.exists(outpath):
    #     continue

    os.makedirs(outpath, exist_ok=True)

    try:
        output_filepath = "%s/%s" % (outpath, filename[:-4])
        loopextractor.run_algorithm(filepath, n_templates=[0, 0, 0], 
                            output_savename=output_filepath)
    except Exception as exp:
        print(exp)
# %%
from loopextractor import loopextractor
loopextractor.run_algorithm("./loopextractor/jamming.wav", n_templates=[0, 0, 0], 
                            output_savename="./loopextractor/sample-loops")
# %%

# trim silence
import soundfile as sf
from glob import glob
import librosa

out_dirpath_root = '/media/nao/hd8t/Datasets/Audio/youtube-5m-loops'  
for filepath in glob("%s/*/*.wav" % out_dirpath):
    y, sr = librosa.load(filepath)
    filename = os.path.basename(filepath)
    yt, index = librosa.effects.trim(y, top_db=40, frame_length=64, hop_length=32)
    sf.write(out_dirpath2 + filename, yt, sr)
    print(filename, len(y), len(yt))
    