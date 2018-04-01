import os
import numpy as np

path='Preproc/'
classname = os.listdir(path)[0]
files = os.listdir(path + classname)
infilename = files[0]
audio_path = path + classname + '/' + infilename