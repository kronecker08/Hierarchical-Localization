import argparse
import torch
from pathlib import Path
import h5py
import logging
from types import SimpleNamespace
import cv2
import numpy as np
from tqdm import tqdm
import pprint

from . import extractors
from .utils.base_model import dynamic_load
from .utils.tools import map_tensor

class ImageDataset():

    def __init__(self, conf):
        self.conf = conf 


    def processed(self, image):
#         print(self.conf.keys())

        if self.conf["grayscale"]:
#             print("TRUE")
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR

        if not self.conf["grayscale"]:
            image = self.image[:, :, ::-1]  # BGR to RGB
        if image is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        if self.conf["resize_max"] and max(w, h) > self.conf["resize_max"]:
            scale = self.conf["resize_max"]  / max(h, w)
            h_new, w_new = int(round(h*scale)), int(round(w*scale))
            image = cv2.resize(
                image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        if self.conf["grayscale"]:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.
#         print(image)

        data = {
#             'name': "query",    ## gave the image name query ;)input.unsqueeze(0)
            'image': (torch.from_numpy(image)).unsqueeze(0),
            'original_size': torch.from_numpy(np.array([size])),
        }
        return data



@torch.no_grad()
def main(conf, image, as_half=False):  ## what is this half stuff
#     logging.info('Extracting local features with configuration:'
#                  f'\n{pprint.pformat(conf)}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    loader = ImageDataset(conf['preprocessing'])
    data = loader.processed(image)
#     print(type(data))
#     print(data.keys())
    if True:
#         print(data)
        pred = model(map_tensor(data, lambda x: x.to(device)))
#         print(pred)
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5


        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
#         name = data['name'][0] ## name of the file
        pred["name"] = "query"
        

#     logging.info('Finished exporting features.')
    pred_copy = pred 
    del pred ## do not know why they are doing this 
    return pred_copy