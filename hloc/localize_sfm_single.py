import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import h5py
from tqdm import tqdm
import pickle
import pycolmap

from .utils.read_write_model_apna import read_model
from .utils.parsers import (
    parse_image_lists_with_intrinsics, parse_retrieval, names_to_pair)





def pose_from_cluster(qname, qinfo, global_match_names_list, db_images, points3D, query_stuff, matches_dict, thresh):
    kpq = query_stuff['keypoints'].__array__() ## keypoints of the query #image
    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0


    for i, db_name in enumerate(global_match_names_list):
#         print(db_name)
        db_id = db_images[db_name].id  
        points3D_ids = db_images[db_name].point3D_ids                               
        pair = names_to_pair(qname, db_name)

        matches = matches_dict[pair]['matches'].__array__()  

        valid = np.where(matches > -1)[0]   ## indexes where it is greater
        valid = valid[points3D_ids[matches[valid]] != -1]   # finding the ones # #which matched and which are in the 3D model 
        num_matches += len(valid)

        for idx in valid:
            id_3D = points3D_ids[matches[idx]]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mkpq = kpq[mkp_idxs]
    mkpq += 0.5  # COLMAP coordinates

    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    mp3d = [points3D[j].xyz for j in mp3d_ids]
    mp3d = np.array(mp3d).reshape(-1, 3)

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                       for i in idxs for j in kp_idx_to_3D[i]]

    camera_model, width, height, params = qinfo
    
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }
    ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)
    ret['cfg'] = cfg
    return ret, mkpq, mp3d, mp3d_ids, num_matches, (mkp_idxs, mkp_to_3D_to_db)


def main(reference_sfm, query_stuff, global_match_names_list, matches_dict, ransac_thresh=12, covisibility_clustering=False):
    
    assert reference_sfm.exists(), reference_sfm   ## SFM Model       
    

#     logging.info('Reading 3D model...')    
    _, db_images, points3D = read_model(str(reference_sfm), '.bin')
    

#     feature_file = h5py.File(features, 'r')

    poses = {}
#     logging.info('Starting localization...')
    
    if True:
        qname = query_stuff["name"]
        qinfo = query_stuff["params"] 

        if True:
            ret, mkpq, mp3d, mp3d_ids, num_matches, map_ = pose_from_cluster(qname, qinfo, global_match_names_list, db_images, points3D, query_stuff, matches_dict, thresh=ransac_thresh)
            
            # qname == query name 
            # qinfo = camera parameters
            # db_ids == ids of matches in the db
            # db_images == name of matches in the db
            # points3D == what we get from the SFM model
            # feature file is after reading
            # match file is after reading 
            # logging.info(f'# inliers: {ret["num_inliers"]}')

            if ret['success']:
                poses = (ret['qvec'], ret['tvec'])
            else:
                closest = db_images[db_ids[0]]
                poses = (closest.qvec, closest.tvec)
#             logs['loc'] = {
#                 'db': db_ids,
#                 'PnP_ret': ret,
#                 'keypoints_query': mkpq,
#                 'points3D_xyz': mp3d,
#                 'points3D_ids': mp3d_ids,
#                 'num_matches': num_matches,
#                 'keypoint_index_to_db': map_,
#             }
        return poses

    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--covisibility_clustering', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)
