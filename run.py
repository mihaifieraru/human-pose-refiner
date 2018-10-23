from pprint import pprint
from scipy.misc import imread, imresize
import numpy as np
from numpy import concatenate as cat
from numpy import array as arr

from nnet.predict import setup_pose_prediction, run_pose_prediction


def get_scale(cfg, pose):
    chin = None
    head_top = None
    for joint in pose:
        if joint[0] == 12:
            chin = [int(joint[1]), int(joint[2])]
        if joint[0] == 14:
            head_top = [int(joint[1]), int(joint[2])] 
    if chin and head_top:
        stick = np.array(chin) - np.array(head_top)
        diag_len = np.linalg.norm(stick) * cfg.mean_diag_to_stick
    else:
        diag_len = cfg.mean_hbbox_diag      
    if diag_len == 0:
            diag_len = cfg.mean_hbbox_diag
   
    # Rewrite the following lines 
    SC_BIAS = 0.6  # 0.8*0.75
    headSize = SC_BIAS * diag_len

    ref_height = 200
    HEAD_HEIGHT_RATIO = 1/8
    refHeight = 400

    sc = ref_height * HEAD_HEIGHT_RATIO / headSize
    assert(sc<100 and sc>0.01)
    scale = 1 / sc
    new_sc = scale*200/refHeight

    final_sc = cfg.global_scale / new_sc

    return final_sc
    

def crop_input(cfg, rgb, coords):
    coords_for_crop = coords

    if coords_for_crop.shape[0] != 0:
        minX, minY = coords_for_crop.min(axis=0).tolist()
        maxX, maxY = coords_for_crop.max(axis=0).tolist()
    else:
        minX = 0
        minY = 0
        maxX = rgb.shape[1]
        maxY = rgb.shape[0]

    x1 = round(max(0, (minX - cfg.delta_crop)))
    y1 = round(max(0, (minY - cfg.delta_crop)))
    x2 = round(min(rgb.shape[1], (maxX + cfg.delta_crop)))
    y2 = round(min(rgb.shape[0], (maxY + cfg.delta_crop)))

    # major hack to deal with corner case in which pose gets out of the image
    if y2 - y1 < cfg.stride or x2 - x1 < cfg.stride:
        return np.copy(rgb), np.copy(coords), [0, 0]

    rgb_cropped = np.copy(rgb[y1:y2, x1:x2, :])
    coords_cropped = np.copy(coords)
    if coords_cropped.size:
        coords_cropped[:, 0] = coords_cropped[:, 0] - x1
        coords_cropped[:, 1] = coords_cropped[:, 1] - y1

    delta = [x1, y1]

    return rgb_cropped, [coords_cropped], delta


def generate_scmap(cfg, joint_id, coords, size):
    dist_thresh = cfg.pos_dist_thresh * cfg.global_scale
    num_joints = cfg.num_joints

    scmap = np.zeros(cat([size, arr([num_joints])]))

    dist_thresh_sq = dist_thresh ** 2

    width = size[1]
    height = size[0]

    for person_id in range(len(coords)):
        for k, j_id in enumerate(joint_id[person_id]):
            joint_pt = coords[person_id][k, :]
            j_x = np.asscalar(joint_pt[0])
            j_y = np.asscalar(joint_pt[1])
            
            # don't loop over entire heatmap, but just relevant locations
            min_x = int(round(max(j_x - dist_thresh - 1, 0)))
            max_x = int(round(min(j_x + dist_thresh + 1, width - 1)))
            min_y = int(round(max(j_y - dist_thresh - 1, 0)))
            max_y = int(round(min(j_y + dist_thresh + 1, height - 1)))

            for j in range(min_y, max_y + 1):  # range(height):
                pt_y = j
                for i in range(min_x, max_x + 1):  # range(width):
                    pt_x = i
                    dx = j_x - pt_x
                    dy = j_y - pt_y
                    dist = dx ** 2 + dy ** 2

                    if dist <= dist_thresh_sq:
                        scmap[j, i, j_id] = 1

    return scmap * 255


def normalize_input(cfg, pose, im_path):
    scale = get_scale(cfg, pose)

    rgb_input = imresize(imread(im_path, mode='RGB'), scale)
    
    pose = [[joint[0], joint[1] * scale, joint[2] * scale] for joint in pose] 
    coords = np.array([[joint[1], joint[2]] for joint in pose])
    joint_ids = [[joint[0] for joint in pose]]

    rgb_input, coords_cropped, crop_delta = crop_input(cfg, rgb_input, coords)
    
    scmap_input = generate_scmap(cfg, joint_ids, coords_cropped, rgb_input.shape[0:2]) 
    
    input_data = [rgb_input, scmap_input]

    return input_data, crop_delta, scale


def run_inference(cfg, input_data, tf_var=None):
    if tf_var is None:
        sess, batch_inputs, outputs = setup_pose_prediction(cfg)
    else:
        sess = tf_var[0]
        batch_inputs = tf_var[1]
        outputs = tf_var[2]
    pose = run_pose_prediction(cfg, input_data, sess, batch_inputs, outputs)    
    return pose


def adjust_pose_to_input(cfg, cropped_pose, crop_delta, scale):
    new_pose = [[joint[0], (joint[1] + crop_delta[0]) / scale, 
                (joint[2] + crop_delta[1]) / scale, joint[3]] 
                for joint in cropped_pose]
    return new_pose


def run(cfg, pose, im_path, tf_var=None):
    input_data, crop_delta, scale = normalize_input(cfg, pose, im_path) 
    if input_data[0].shape[0] * input_data[0].shape[1] > 3000000:
        return pose
    cropped_pose = run_inference(cfg, input_data, tf_var)
    pose = adjust_pose_to_input(cfg, cropped_pose, crop_delta, scale)
    return pose

