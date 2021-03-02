# JSS
## TODO 
#using pca to reduce the dimensionality of the global descriptors
#using MobileVnet instead of netVLAD
#using hfnet instead of netVlAD and MobileVnet --that is not working 
# this will also help reduce the complexity 
#understand what superglue is outputting
# getting two models from SFM 
# Setting up the scale of the model -- this is done easy and simple
import numpy as np
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from pprint import pformat
import h5py
import json
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt



from hfnet.models import get_model
from hloc.utils.viz import plot_images, plot_keypoints
from hloc import feature_extractor_new
from hloc import match_feature_single
from hloc import localize_sfm_single
from hloc import extract_feature_single_masked
import jsonparser

def global_descriptors_parser(path_to_model):
    file_name = 'global_features.h5'
    hfile = h5py.File(str(path_to_model/file_name), 'r')
    image_names = list(hfile.keys())
    desc = [hfile[i]['global_descriptor'].__array__() for i in image_names]
    desc = np.stack(desc, 0)
    return image_names, desc
## Send in a RGB image
def global_descriptor_query(config_global, image):
    checkpoint_path = Path(config_global["checkpoint_path"])
    keys = ['global_descriptor']    
    with get_model(config_global['model']['name'])(
            data_shape={'image': [None, None, None, 3]},
            **config_global['model']) as net:
        if checkpoint_path is not None:
            net.load(str(checkpoint_path))
        data = {"image": image}
        predictions = net.predict(data, keys=keys)
    return predictions['global_descriptor']
def compute_distance(desc1, desc2):
    # For normalized descriptors, computing the distance is a simple matrix multiplication.
    return 2 * (1 - desc1 @ desc2.T)
def global_matches(path_to_model, config_global, image_query, number):
    image_names, global_matrix = global_descriptors_parser(path_to_model)
    GlobalDescriptorQuery = global_descriptor_query(config_global, image_query) 
    neighbours = compute_distance(GlobalDescriptorQuery, global_matrix)
    nearest_neighbours = np.argsort(neighbours)[:number]
    names_nearest = [image_names[i] for i in nearest_neighbours]
    return names_nearest, nearest_neighbours
def local_descriptor_and_matches(image, path_to_model, feature_conf_superpoint, matcher_conf, names_nearest):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    query_stuff = feature_extractor_new.main(feature_conf_superpoint, image_gray)
    plot_images([image])
    plot_keypoints([query_stuff["keypoints"]], ps =8)
    feature_file = path_to_model/"features_superpoint.h5"
    matches_dict = match_feature_single.main(matcher_conf, query_stuff, names_nearest, feature_file)
    return matches_dict, query_stuff
def local_descriptor_and_matches_mask(image, path_to_model, feature_conf_superpoint, matcher_conf, names_nearest, mask):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    query_stuff = extract_feature_single_masked.main(feature_conf_superpoint, image_gray, mask)
    plot_images([image])
    plot_keypoints([query_stuff["keypoints"]], ps =8)
    feature_file = path_to_model/"features_superpoint.h5"
    matches_dict = match_feature_single.main(matcher_conf, query_stuff, names_nearest, feature_file)
    return matches_dict, query_stuff
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
def colmap_to_global(_tvec, _qvec):
    return -np.matmul(qvec2rotmat(_qvec).transpose(), _tvec)
def localize(matches_dict, model, query_stuff, names_nearest):
    query_stuff["params"] = ("SIMPLE_RADIAL", 640, 480, (658.503, 320, 180, 0.0565491))
    pose = localize_sfm_single.main(model, query_stuff, names_nearest, matches_dict)
    pose_global = colmap_to_global(pose[1], pose[0])
    return pose_global
def error(tvec, CtG):
    return np.linalg.norm(tvec-CtG)