from pprint import pprint
import os, json
import numpy as np
from copy import deepcopy

from run import run

def get_images_lookup(images):
    lookup = {}
    for image in images:
        lookup[image['id']] = image
    return lookup


def pt2_to_pt(pt2_id):
    corresp = {0: 13, 1: 12, 2: 14, 3: None, 4: None, 5: 9, 6: 8, 7: 10, 8: 7, 9: 11, 10: 6, 11: 3, 12: 2, 13: 4, 14: 1, 15: 5, 16: 0}
    return corresp[pt2_id]

def pt_to_pt2(pt_id):
    corresp = {0: 16, 1: 14, 2: 12, 3: 11, 4: 13, 5: 15, 6: 10, 7: 8, 8: 6, 9: 5, 10: 7, 11: 9, 12: 1, 13: 0, 14: 2}
    return corresp[pt_id]    

def run_posetrack_v2(cfg, tf_var=None):
    json_fns = [pos_json for pos_json in os.listdir(cfg.dataset) if pos_json.endswith('.json')] 
    for json_fn in json_fns:
        json_path = os.path.join(cfg.dataset, json_fn)
        new_json_path = os.path.join(cfg.dir_json_pred, json_fn)
        with open(json_path, 'r') as f:
            seq_data = json.load(f)
            new_seq_data = deepcopy(seq_data)
            im_lookup = get_images_lookup(seq_data['images'])
            for i in range(len(seq_data['annotations'])):
                # replace 'image_id' with 'id'
                im_name = im_lookup[seq_data['annotations'][i]['image_id']]['file_name']
                im_path = cfg.dataset_root + im_name
                print(im_path)
                 
                pose = []
                cutoff = -500
                kp = seq_data['annotations'][i]['keypoints']
                for joint_id in range(17):
                    pt2_joint_id = pt2_to_pt(joint_id)
                    if kp[3 * joint_id + 2] == 1 and pt2_joint_id is not None and kp[3 * joint_id] > cutoff and kp[3 * joint_id + 1] > cutoff:
                        pose.append([pt2_joint_id, kp[3 * joint_id], kp[3 * joint_id + 1]])
                if len(pose):
                    refined_pose = run(cfg, pose, im_path, tf_var)
                    new_kp = [0] * (3 * 17)
                    new_scores = [0] * 17
                    for joint in refined_pose:
                        new_joint_id = pt_to_pt2(joint[0])
                        new_kp[new_joint_id * 3] = joint[1]
                        new_kp[new_joint_id * 3 + 1] = joint[2]
                        new_kp[new_joint_id * 3 + 2] = 1
                        new_scores[new_joint_id] = seq_data['annotations'][i]['scores'][new_joint_id]
                    new_seq_data['annotations'][i]['keypoints'] = new_kp
                    new_seq_data['annotations'][i]['scores'] = new_scores

            with open(new_json_path, 'w') as outfile:
                json.dump(new_seq_data, outfile, indent=4) 
