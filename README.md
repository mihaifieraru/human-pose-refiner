The code is tested with python3 and Tensorflow v1.5.
We recommend using virtualenv to create an environment and install the rest of the packages (including TF) using the associated pip. Official documentation https://www.tensorflow.org/install/pip .
```
pip install -U https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl
pip install -U easydict numpy scipy scikit-image init_weights: /BS/mihai/work/experiments/ddpose_track_train/exp20cc/snapshot-515000
```


Choose a path where you'd like to clone the code and then set the environment variable `HUMAN_POSE_REFINER` accordingly.
```
# HUMAN_POSE_REFINER=/BS/mihai/work/src2/human-pose-refiner
git clone #todo $HUMAN_POSE_REFINER
```

Download the weights of the model used for refining predictions on the PoseTrack dataset.
```
cd $HUMAN_POSE_REFINER/model/
# download snapshot-515000.zip from https://drive.google.com/open?id=1BEDB_zX8-0XrpdBimYAN4GwaAVAUrrBJ
unzip snapshot-515000.zip
rm snapshot-515000.zip
```

Download the PoseTrack 2018 Dataset (images and annotations) from https://posetrack.net/users/download.php .
It should be organized in this way:
```
$HUMAN_POSE_REFINER/
    --data
        --posetrack_data_v2
            --annotations
                --test
                --train
                --val
            --images
                --test
                --train
                --val
```
Set path to posetrack-v2 initial predictions
```
vim $HUMAN_POSE_REFINER/exp/improve-mota-posetrack-v2/pose_cfg.yaml
# set the parameter dataset:
```

Refine predictions such that the MOTA score is improved:
```
cd $HUMAN_POSE_REFINER/exp/improve-mota-posetrack-v2/
CUDNN_USE_AUTOTUNE=0 CUDA_VISIBLE_DEVICES=0 /BS/mihai/work/software/hpr-env/bin/python $HUMAN_POSE_REFINER/run_dataset.py
```


