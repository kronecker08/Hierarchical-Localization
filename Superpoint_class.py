#JSS
import torch
import cv2
import numpy as np
import os

from hloc.utils.base_model import dynamic_load
from hloc import extractors
from hloc.utils.tools import map_tensor
feature_config ={'model': {'max_keypoints': 4096, 'name': 'superpoint', 'nms_radius': 3},'preprocessing': {'grayscale': True, 'resize_max': 960}}

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Superpoint:
    def __init__(self):
        print("Initializing superpoint")
        self.load_superpoint()

    @torch.no_grad()
    def load_superpoint(self):
        self.device_sp = 'cuda' if torch.cuda.is_available() else 'cpu'
#         with torch.cuda.device(1):
#         print("Device for superpoint", self.device_sp)
        Model_sp = dynamic_load(extractors, feature_config['model']['name'])
        self.model_sp = Model_sp(feature_config['model']).eval().to(self.device_sp)

    def processed(self):
        if feature_config["preprocessing"]["grayscale"]:
#             print(np.shape(self.image))
            image_processed = cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY)
        if image_processed is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image_processed = image_processed.astype(np.float32)
        size = image_processed.shape[:2][::-1]
        w, h = size

        if feature_config["preprocessing"]["resize_max"] and max(w, h) > feature_config["preprocessing"]["resize_max"]:
            scale = feature_config["preprocessing"]["resize_max"]  / max(h, w)
            h_new, w_new = int(round(h*scale)), int(round(w*scale))
            image = cv2.resize(
                image_processed, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        if feature_config["preprocessing"]["grayscale"]:
            image_processed = image_processed[None]
        else:
            image_processed = image_processed.transpose((2, 0, 1))  # HxWxC to CxHxW
        image_processed = image_processed / 255.

        self.data = {
            'image': (torch.from_numpy(image_processed)).unsqueeze(0),
            'original_size': torch.from_numpy(np.array([size])),
        }
        return self.data
    
    @torch.no_grad()
    def key_point_and_descriptor(self, image):
        self.image = image
        self.processed()
#         print(self.data)
        pred_sp = self.model_sp(map_tensor(self.data, lambda x: x.to(self.device_sp)))
#         print(pred_sp)
        pred_sp = {k: v[0].cpu().numpy() for k, v in pred_sp.items()}
        pred_sp['image_size'] = original_size = self.data['original_size'][0].numpy()
        if 'keypoints' in pred_sp:
            size = np.array(self.data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred_sp['keypoints'] = (pred_sp['keypoints'] + .5) * scales[None] - .5
#         if as_half:
#             for k in pred:
#                 dt = pred[k].dtype
#                 if (dt == np.float32) and (dt != np.float16):
#                     pred[k] = pred[k].astype(np.float16)
# #         name = data['name'][0] ## name of the file
        pred_sp["name"] = "query"
        

#         logging.info('Finished exporting features.')
        self.pred_copy = pred_sp 
        del pred_sp ## do not know why they are doing this 
        return self.pred_copy
        
