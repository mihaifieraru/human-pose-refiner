The code is tested with python3 and Tensorflow v1.5.
We recommend using virtualenv to create an environment and install the required packages (including TF) using the associated pip. See official documentation here: https://www.tensorflow.org/install/pip .

```
# install tensorflow v1.5
pip install -U https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp35-cp35m-linux_x86_64.whl

# install required packages
pip install -U easydict numpy scipy scikit-image init_weights: /BS/mihai/work/experiments/ddpose_track_train/exp20cc/snapshot-515000
```


Choose a path where you'd like to clone the code and then set the environment variable `HUMAN_POSE_REFINER` accordingly.
```
# HUMAN_POSE_REFINER=/path/to/human-pose-refiner
git clone git@github.com:mihaifieraru/human-pose-refiner.git $HUMAN_POSE_REFINER
```

Download the weights of the model used for refining predictions on the PoseTrack dataset.
```
# download snapshot-515000.zip from https://drive.google.com/open?id=1BEDB_zX8-0XrpdBimYAN4GwaAVAUrrBJ
cp snapshot-515000.zip $HUMAN_POSE_REFINER/model/
cd $HUMAN_POSE_REFINER/model/
unzip snapshot-515000.zip
rm snapshot-515000.zip
```

Download the PoseTrack 2018 Dataset (images and annotations) from https://posetrack.net/users/download.php .
It should be organized in the following way:
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
Set path to the posetrack-v2 predictions that you want to refine. This should be a folder containing the json files that are in the same format required by the evaluation https://github.com/leonid-pishchulin/poseval .
```
vim $HUMAN_POSE_REFINER/exp/improve-mota-posetrack-v2/pose_cfg.yaml
# set the parameter dataset:
```

Refine predictions such that the MOTA score is improved:
```
# activate the environment containing TF 1.5 and the required packages
# set gpu ID in CUDA_VISIBLE_DEVICES. 
cd $HUMAN_POSE_REFINER/exp/improve-mota-posetrack-v2/
CUDNN_USE_AUTOTUNE=0 CUDA_VISIBLE_DEVICES=0 python3 $HUMAN_POSE_REFINER/run_dataset.py
```

Refine predictions such that the MAP score is improved:
```
# activate the environment containing TF 1.5 and the required packages
# set gpu ID in CUDA_VISIBLE_DEVICES. 
cd $HUMAN_POSE_REFINER/exp/improve-map-posetrack-v2/
CUDNN_USE_AUTOTUNE=0 CUDA_VISIBLE_DEVICES=0 python3 $HUMAN_POSE_REFINER/run_dataset.py
```

A significant amount of the code is borrowed from the following repository https://github.com/eldar/pose-tensorflow.

## Citation
If Human Pose Refiner helps your research, please cite it in your respective publications:
```
@InProceedings{Fieraru_2018_CVPR_Workshops,
author = {Fieraru, Mihai and Khoreva, Anna and Pishchulin, Leonid and Schiele, Bernt},
title = {Learning to Refine Human Pose Estimation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2018}
}
```
