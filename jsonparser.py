#JSS
## what format are the rotation is in?
## right now parsing it and keeping it 
import numpy as np
import json 
from pathlib import Path
# root = '/home/orb/trajectory4' # need to pass in as an argument 
def parser(root):
    image_dir = root+"/useful_filenames.json"
    translation_dir = root+"/useful_tvecs.json"
    rotation_dir = root+"/useful_rvecs.json"
    with open(image_dir) as f:
        image_names = json.load(f)
    with open(translation_dir) as f:
        translation_values = json.load(f)
    with open(rotation_dir) as f:
        rotation_values = json.load(f)  
    names_trans_rot = {}
    for i in range(len(image_names)):
        tvec = translation_values[i]
        rvec = rotation_values[i]
        name = image_names[i]
        names_trans_rot[name]= {"tvec":tvec, "rvec":rvec}
    # print(len(names_trans_rot))
    return names_trans_rot
# parser()