# Two-Stage Object Detection on Scanning Radars with Feature Enrichment

## Installation
```bash
# create conda env
conda create -n sungting-handover python=3.7

# switch to cuda version 10.2
# check https://github.com/phohenecker/switch-cuda for detail
source sungting/switch-cuda/switch-cuda.sh 10.2 

# clone this repo to install dependancy
git clone https://github.com/scheckmedia/centernet-uda.git
cd centernet-uda

conda activate sungting-handover
pip install numpy
pip install dotsi
pip install tensorboard
pip install -r requirements.txt
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
cd libs/DCNv2 && ./make.sh


# downloading pretrained weight
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_amd64.tar.gz

tar zxvf gdrive_2.1.1_linux_amd64.tar.gz

# download this folder on google drive https://drive.google.com/drive/folders/1f9IMqXP-nu3xlHRKE-QhmqhZ-RdW9G6z
gdrive download --recursive 1f9IMqXP-nu3xlHRKE-QhmqhZ-RdW9G6z
# you will have to visit the link in the browser and log in your own google account => get the key => paste it to the terminal
# for more details, please check this https://unix.stackexchange.com/a/429457


# clone my repo from gpal github: https://github.com/geometrical-edu/two-stage-scanning-radar-object-detector
git clone https://github.com/geometrical-edu/two-stage-scanning-radar-object-detector

# cd into your target repo
```

## Code Structure
Each folder in the repo corresponds to the settings in my thesis.
The name of the folders is constructed in this way: [dataset]-[single-stage(1)/two-stage(2)]-[with(5)/without(1) feature-enrichment]
So there are total eight settings in this repo.

My code structure use [hydra](https://github.com/facebookresearch/hydra) as configuration framework.
In each code folder, run the following command:

```
python train.py experiment=itri_train
```
to run the configuration in the folder of \$folder\$/configs/experiment/itri_train.yaml

Some key args in config hw2itri_em_dla.yaml:
- pretrained: network pretrained weight used to train the model or do inferences on the test set
- image_folder & annotation_file 
	- image_folder is the root path of the dataset
	- annotation_file is the path for the annotations. The file path inside the annotation_file is based on the root path of image_folder
- input_size: CenterNet (baseline used in my thesis) downsampled images 4 times. So use input_size to first adjust your image files to the closest number that can be devided by 4
- test_only: if not test only, the training process will be run when `python train.py experiment=hw2itri_em_dla' is triggered. Otherwise, only the inference process will be run.
- experiment: the experiment name of this config file
- is_vis_output: save the inference image results of object detection

In each experiment, the config used for that experiment will be stored in `$folder$/outputs/$exp_name$/$exp_folder$/.hydra/config.yaml`

`$exp_name$` is set in the arg experiment inside each config file. `$exp_folder$` is set in side hydra->run->dir inside each config file.

Your probably only need to focus on directories radiate-2-5 & itri-2-5 
(cd to the root of your target repo)

radiate-2-5

train: `python train.py experiment=radiate_train`

test: `python train.py experiment=radiate_test`

itri-2-5

train: `python train.py experiment=itri_train`

test: `python train.py experiment=itri_test`


# Utility Code

## utility code for separating each single message from bag files
Please check `util/code/save_single_sensor/rosbag_save_rav4_single_sensor.py` for saving each message from different topics to folders on your own machine.

## utility code for generating bbox using livox_detection:
Please follow: https://github.com/Livox-SDK/livox_detection to install your env

use `util/livox_det/bimo_centernet_20201007/baraja_livox_detection_0_200_timelapsed_with_class.py` to generate bbox from lidar pointclouds
