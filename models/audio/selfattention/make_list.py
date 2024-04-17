import os
import glob

root_path = '/home/pmri/WZQ/New_try/TAU_data_audio/'    ## the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.wav")))
normal = []
with open('/home/pmri/WZQ/self-attention/audiolist.txt', 'w+') as f:  ## the name of feature list
    for file in files:
        newline = file+'\n'
        f.write(newline)