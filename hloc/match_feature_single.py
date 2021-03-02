import argparse
import torch
from pathlib import Path
import h5py
import logging
from tqdm import tqdm
import pprint

from . import matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superglue': {
        'output': 'matches-superglue_T3',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'NN': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'mutual_check': True,
            'distance_threshold': 0.7,
        },
    }
}


@torch.no_grad()
# def main(conf, pairs, features, export_dir, exhaustive=False):
## feature is the name of the file
## Used export dir let it be ??? -> dont think so // does not matter for now
## need to pass export_dir as it is being used to find the path 

def main(conf, query_stuff, list_of_image_from_global, feature_file):

#     logging.info('Matching local features with configuration:'
#                  f'\n{pprint.pformat(conf)}')

#     feature_path = Path(export_dir, features+'.h5')
#     assert feature_path.exists(), feature_path
    feature_file = h5py.File(str(feature_file), 'r')


    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    
    matches_dict = {}


    for name_from_global in (list_of_image_from_global):

        pair = names_to_pair(query_stuff["name"], name_from_global)

        data = {}

        feats1 = feature_file[name_from_global]
        

        for k in feats1.keys():
            data[k+'0'] = query_stuff[k].__array__()
        for k in feats1.keys():
            data[k+'1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(device)
                for k, v in data.items()}

        # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((1, 1,)+tuple(query_stuff['image_size'])[::-1])
#         print(tuple(query_stuff['image_size'])[::-1])
        
        data['image1'] = torch.empty((1, 1,)+tuple(feats1['image_size'])[::-1])

        pred = model(data)
#         grp = match_file.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
#         grp.create_dataset('matches0', data=matches)

        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            matches_dict[pair] = {"scores":scores, "matches": matches}
#             grp.create_dataset('matching_scores0', data=scores)
        else:
            matches_dict[pair] = {"matches": matches}
            
 

#         matched |= {(name0, name1), (name1, name0)}

#     match_file.close()
    
#     logging.info('Finished exporting matches.')
    return matches_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    parser.add_argument('--exhaustive', action='store_true')
    args = parser.parse_args()
    main(
        confs[args.conf], args.pairs, args.features, args.export_dir,
        exhaustive=args.exhaustive)
