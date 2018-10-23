from pprint import pprint
import os, json
import numpy as np
from copy import deepcopy

from run import run

def run_posetrack_v1(cfg, tf_var=None):
    json_fns = [pos_json for pos_json in os.listdir(cfg.dataset) if pos_json.endswith('.json')] 
    for json_fn in json_fns:
        json_path = os.path.join(cfg.dataset, json_fn)
        new_json_path = os.path.join(cfg.dir_json_pred, json_fn)
        with open(json_path, 'r') as f:
            seq_data = json.load(f)
            new_seq_data = deepcopy(seq_data)
            for i in range(len(seq_data['annolist'])):
                init_im_name = seq_data['annolist'][i]['image']
                if isinstance(init_im_name, list):
                    init_im_name = seq_data['annolist'][i]['image'][0]["name"]
                im_name = "/".join(init_im_name.split("/")[-3:])
                im_path = cfg.dataset_root + im_name
                print(im_path)
                elements_to_delete = []
                for j in range(len(seq_data['annolist'][i]["annorect"])):
                    point = seq_data['annolist'][i]["annorect"][j]['annopoints'][0]['point']
                    pose = []
                    cutoff = -500
                    for joint in point:
                        if joint["x"][0] > cutoff and joint["y"][0] > cutoff:
                            pose.append([joint["id"][0], joint["x"][0], joint["y"][0]])
                    if len(pose):
                        refined_pose = run(cfg, pose, im_path, tf_var)                  
                        new_point = []
                        for joint in refined_pose:
                            old_score = 0
                            for old_joint in point:
                                if old_joint["id"][0] == joint[0]:
                                    old_score = old_joint["score"][0]
                            new_joint = {"id": [joint[0]], "score": [old_score], "x": [joint[1]], "y": [joint[2]]}
                            new_point.append(new_joint)
                        new_seq_data['annolist'][i]["annorect"][j]['annopoints'][0]['point'] = new_point
                    else:
                        elements_to_delete.append(new_seq_data['annolist'][i]["annorect"][j])
                for el in elements_to_delete:
                    new_seq_data['annolist'][i]["annorect"].remove(el)
            with open(new_json_path, 'w') as outfile:
                json.dump(new_seq_data, outfile, indent=4) 
