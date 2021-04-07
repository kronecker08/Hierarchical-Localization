##JSS
import sys,time
import numpy as np
import cv2
import roslib
import rospy
import h5py
from sensor_msgs.msg import Image
import common_function as cf
from pathlib import Path
import pycolmap
from collections import defaultdict
import time

## superglue ka hai yeh
import torch
import tqdm
from hloc import matchers
from hloc.utils.base_model import dynamic_load
from hloc.utils.parsers import names_to_pair 

## localize sfm ka hai
from hloc.utils.read_write_model_apna import read_model


class image_pose:   
    @torch.no_grad()
    def __init__(self,config):
        self.subscriber = rospy.Subscriber("/camera/image_raw",
                                          Image, self.callback, queue_size=500)       
        self.config = config 
        self.hfnet = cf.HFNet(Path(self.config['model_path']), self.config['outputs'])
        self.load_global_descriptors()
        Model = dynamic_load(matchers, self.config["superglue_config"]["model"]["name"])
        ## superglue was jumping off too cuda:1 ???
        self.device = 'cuda:0'
        self.superglue_model = Model(self.config["superglue_config"]["model"]).eval().to(self.device)
        self.load_local_descriptors()
        self.load_map()
        camera_model, width, height, params = self.config["camera_params"]
        self.camera_params = {
                        'model': camera_model,
                        'width': width,
                        'height': height,
                        'params': params,
                    }
        
    def load_map(self):
        assert Path(self.config["sfm_dir"]).exists(), Path(self.config["sfm_dir"])
#         logging.info('Reading 3D model...')
        _, self.db_images, self.points3D = read_model(str(self.config["sfm_dir"]), '.bin')
        print("read model")

        

        
    def load_local_descriptors(self):
        local_feature_path = Path(self.config["output_path"])/"local_feature.h5"
        self.local_feature_file = h5py.File(str(local_feature_path),"r")        
        
        
    def load_global_descriptors(self):
        global_feature_path = Path(self.config["output_path"])/"global_features.h5"
        global_feature_file= h5py.File(str(global_feature_path),"r")
        self.global_feature_dict = {}
        for i,j in enumerate(list(global_feature_file.keys())):
            des = global_feature_file[j]["global_descriptor"].__array__()
            if i == 0:
                self.global_matrix = des
            else:
                self.global_matrix = np.vstack((self.global_matrix, des))
            self.global_feature_dict[i] = {"name":j, "descriptor": des}
    
    def callback(self, image_data):
        self.start_time = time.time()
        self.cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.descriptors()
    
    def descriptors(self):
        self.db = self.hfnet.inference((self.cv_image).astype(float))
        self.image_size = np.array(self.cv_image.shape[:2][::-1])
        self.global_descriptors = self.db["global_descriptor"]
        self.keypoints = self.db["keypoints"]
        self.local_descriptors = np.transpose(self.db["local_descriptors"])
        self.scores = self.db["scores"]
        self.find_global_matches()
    
    def find_global_matches(self):
        neighbours = cf.compute_distance(self.global_descriptors, self.global_matrix)
        global_matches = ((np.argsort(neighbours)[:self.config["global_matches"]]))
        self.global_matches_name = []
        for i in global_matches:
            name = self.global_feature_dict[i]["name"]
            self.global_matches_name.append(name)
        self.find_superglue_matches()
    
    @torch.no_grad()
    def find_superglue_matches(self):
        self.match_dict = {}
        batches = {}
        for i in range(0,self.config["global_matches"], self.config["batch_size"]):
            batches[i] = self.global_matches_name[i:i+self.config["batch_size"]]
        for i in batches:
            names = batches[i]
            kplist0 = []
            kplist1 = []
            desc0 = []
            desc1 = []
            sc0 = []
            sc1 = []
            for j in names:
                pair = names_to_pair("query", j)
                feats1 = self.local_feature_file[j]
                kplist0.append(self.keypoints)
                kplist1.append(feats1["keypoints"].__array__())
                desc0.append(self.local_descriptors)
                desc1.append(feats1["descriptors"].__array__())
                sc0.append(self.scores)
                sc1.append(feats1["scores"].__array__())
            # pad feature0 not necessary
            size_list=[n.shape[0] for n in kplist1]
            max_size = np.max(size_list)
            kplist1 = [ np.concatenate((n, np.zeros((max_size-n.shape[0], n.shape[1]))), axis=0) for n in kplist1]
            desc1 = [ np.concatenate((n, np.zeros((n.shape[0], max_size-n.shape[1]))), axis=1) for n in desc1]
            sc1 = [ np.concatenate((n, np.zeros((max_size-n.shape[0]))), axis=0) for n in sc1]
            data = {'keypoints0':kplist0, 'descriptors0':desc0, 'scores0':sc0,'keypoints1':kplist1, 'descriptors1':desc1, 'scores1':sc1}
            data = {k: torch.from_numpy(np.array(v)).float().to(self.device) for k, v in data.items()}
            # some matchers might expect an image but only use its size
            data['image0'] = torch.empty((len(sc0), 1,)+tuple(self.image_size)[::-1])
            data['image1'] = torch.empty((len(sc0), 1,)+tuple(feats1['image_size'])[::-1])
            pred = self.superglue_model(data)
            index = 0 
            for k in names:
#                 pair = names_to_pair("query", k)
                Matches = pred["matches0"][index].cpu().short().numpy()
                if 'matching_scores0' in pred:
                    Scores = pred['matching_scores0'][index].cpu().half().numpy()
                self.match_dict[k] = {"scores":Scores, "matches":Matches}
                index+=1 
        self.find_pose()
        
    def find_pose(self):
        kp_idx_to_3D = defaultdict(list)
        kp_idx_to_3D_to_db = defaultdict(lambda:defaultdict(list))
        num_matches = 0
        for i,db_name in enumerate(self.global_matches_name):
            db_id = self.db_images[db_name].id
            points3D_ids = self.db_images[db_name].point3D_ids
            matches = self.match_dict[db_name]['matches']
            valid = np.where(matches>-1)[0] # the ones which were matched
            valid = valid[points3D_ids[matches[valid]]!=-1]
            num_matches +=len(valid)
            
            for idx in valid:
                id_3D = points3D_ids[matches[idx]]
                kp_idx_to_3D_to_db[idx][id_3D].append(i)
                # avoid duplicate observations
                if id_3D not in kp_idx_to_3D[idx]:
                    kp_idx_to_3D[idx].append(id_3D)
                    
        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mkpq = self.keypoints[mkp_idxs]
        mkpq = mkpq.astype('float64')
        mkpq += 0.5  # COLMAP coordinates
        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
        mp3d = [self.points3D[j].xyz for j in mp3d_ids]
        mp3d = np.array(mp3d).reshape(-1, 3)
        ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, self.camera_params, self.config["ransac_thresh"])
        if ret["success"]:
            poses = (ret["qvec"], ret["tvec"])
            tvec = cf.colmap_to_global(ret["tvec"], ret["qvec"])
#             print(tvec)
            print(time.time()- self.start_time)
#             print(poses)
        else:
            print("Failed")

            
config = {'model_path':"/home/Hierarchical-Localization/hfnet/model/saved_models/hfnet",
          'outputs': ['global_descriptor', 'keypoints', 'local_descriptors', 'scores'],
          'output_path':"/home/Hierarchical-Localization/outputs/model_T3_localizedT4",
          "global_matches":5,
          "batch_size":1,
          'superglue_config':{'model': {'name': 'superglue', 'weights': 'outdoor', 'sinkhorn_iterations': 50}},
          "ransac_thresh":12,
          "sfm_dir":"/home/Hierarchical-Localization/outputs/model_T3_localizedT4/sfm_superpoint+superglue/geo_registered_model",
          "camera_params":("SIMPLE_RADIAL", 640, 480, (658.503, 320, 180, 0.0565491))}
def main(config):
    ic = image_pose(config)
    rospy.init_node("image_pose", anonymous=True)
    rospy.spin()
    
main(config)