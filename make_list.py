import os
import glob

root_path = '/home/pmri/WZQ/New_try/data/local/image/testing/'    ## the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.png")))
normal = []
with open('/home/pmri/WZQ/New_try/1vedio.txt', 'w+') as f:  ## the name of feature list
    for file in files:
        newline = file+'\n'
        f.write(newline)