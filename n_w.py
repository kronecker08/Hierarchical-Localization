# JSS
import numpy as np
from pathlib import Path
from tqdm import tqdm

## this the path to results file we get from colmap

path_to_results = Path("/home/orb/Downloads/c_l_c_i/results_D.txt")

path_to_save = Path("/home/orb/Downloads/c_l_c_i/ts.txt")


list_of_images = ["l10.jpg", "l40.jpg", "l80.jpg", "l120.jpg", "l150.jpg", "l220.jpg", "l250.jpg", "l280.jpg", "l310.jpg", "l350.jpg", "l380.jpg", "l415.jpg", "l450.jpg", "l485.jpg", "l530.jpg", "l555.jpg", "l597.jpg", "l640.jpg", "l680.jpg", "l720.jpg", "l760.jpg", "l810.jpg", "l845.jpg", "l880.jpg", "l910.jpg","l945.jpg","l980.jpg", "l1015.jpg", "l1131.jpg","11180.jpg"]

with open(path_to_results , "r") as f:
    Lines = f.readlines()
    trans = {}
    for line in Lines:
        line = line.split(" ")
        if line[0] in list_of_images:
            qvec = np.asarray(line[1:5], dtype=np.float32)
            tvec = np.asarray(line[5:8], dtype=np.float32)
            trans[line[0]] = tvec
# print(trans)
with open(path_to_save , 'w') as v:
    for i,j in enumerate(list_of_images):
        if i == len(trans)-1:
            break
        t1 = trans[list_of_images[i]]
        t2 = trans[list_of_images[i+1]]
        e = np.linalg.norm(t1-t2)
        st = str(e) + "\n"
        v.write(st)






