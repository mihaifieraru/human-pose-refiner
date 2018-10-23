import argparse
from copy import deepcopy
import os, sys

from config import load_config
from nnet.predict import setup_pose_prediction

def run_dataset():
    cfg = deepcopy(load_config())

    if not os.path.exists(cfg.dir_json_pred):
        os.makedirs(cfg.dir_json_pred)

    sess, batch_inputs, outputs = setup_pose_prediction(cfg)
    tf_var = [sess, batch_inputs, outputs]    

    if cfg.dataset_type == "posetrack_v1":
        from dataset.posetrack_v1 import run_posetrack_v1
        run_posetrack_v1(cfg, tf_var)
    if cfg.dataset_type == "posetrack_v2":
        from dataset.posetrack_v2 import run_posetrack_v2
        run_posetrack_v2(cfg, tf_var)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, unparsed = parser.parse_known_args()
    
    return_code = run_dataset()
    sys.exit(return_code)
