# MvP-TI
This repository is a fork of the [Multi-view Pose Transformer (MvP)](https://github.com/sail-sg/mvp).

It extends the original implementation to investigate texture-independent multi-view 3D pose estimation for birds. As part of this work, the [3D-POP](https://github.com/alexhang212/dataset-3dpop) dataset has been added, together with evaluation code to evaluate identical to [3D-MuPPET-TI](https://github.com/basilisk4/3D-MuPPET-TI).



## 0. System
The easiest way to run this project is by using the official NVIDIA PyTorch Docker image.
Make sure you have [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed so the container can access your GPU.


```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/pytorch:20.07-py3
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:20.07-py3
```



## 1. Installation
1. Set the project root directory as ${POSE_ROOT}.
2. Install all the required python packages (with requirements.txt).
3. compile deformable operation for projective attention.
```bash
cd ./models/ops
sh ./make.sh
cd ../..
```

## 2. Data and Pre-trained Model Preparation
To use MvP on CMU Panoptic, Shelf/Campus, Human3.6M datasets refer to the original [MvP repository](https://github.com/sail-sg/mvp).

### 2.1 Model
Please follow [VoxelPose](https://github.com/microsoft/voxelpose-pytorch?tab=readme-ov-file#cmu-panoptic-dataset) to download the PoseResNet-50 pre-trained model. Also download the [SAM ViT-H](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)  model checkpoint. For evaluation donwload the [YOLO_Barn model](https://zenodo.org/records/10453890) used by [3D-MuPPET](https://github.com/alexhang212/3D-MuPPET)

### 2.2 3D-POP

Download the [3D-POP dataset](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.HPBBC7) anywhere on your system. You can only download the N6000 folder as well as Sequences 1, 2, 5 and 11. After that clone the 3D-POP-Dataset repository and activate the SAM conda enviroment. The SAM conda enviroment is only needed for dataset creation. The dataset creation can also be performed on another machine if that is more convinient.
```
git clone https://github.com/alexhang212/Dataset-3DPOP.git ./Utils
conda env create -f sam.yml
conda activate SAM
```
Create the training datasets:
```
python ./data/createPop3d.py --path=3D-POP-PATH --out=./data/pop3d
python ./data/createPop3d.py --path=3D-POP-PATH --out=./data/pop3d-seg --seg=True
```
Create the evaluation datasets:
```
python ./data/createPop3d-eval.py --path=3D-POP-PATH --out=./data/pop3d-eval
python ./data/createPop3d-eval.py --path=3D-POP-PATH --out=./data/pop3d-eval-seg --seg=True
```
The directory tree should look like this:
```
${POSE_ROOT}
|-- models
|   |-- pose_resnet50_panoptic.pth.tar
|   |-- sam_vit_h_4b8939.pth
|   |-- YOLO_Barn.pt
|-- data
|   |-- pop3d
|   |-- pop3d-seg
|   |-- pop3d-eval
|   |-- pop3d-eval-seg
```



## 3. Training and Evaluation
For Training and evauluation first  set the correct python path:

```
export PYTHONPATH="${PYTHONPYTH}:path_to_project/MvP-TI"
```

For training on the 3D-POP dataset run:

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env run/train_3d.py --cfg configs/pop3d/best_model_config.yaml
```
And for the segmented 3D-POP dataset run:

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env run/train_3d.py --cfg configs/pop3d-seg/best_model_config.yaml
```

The evaluation result will be printed after every epoch, all result can additionaly be found in the log file. The best checkpoint is saved as model_best.pth.tar. The results in the log files can be plotted with:

```
python Utils/plot.py path_to_log_file
```


## 4. Evaluation

To evaluate a trained model on the evaluation dataset, pass the name, config, the path to the evaluation dataset(data) and the path to the 3D-POP dataset. For our set-up that results in this command:

```
python ./evaluation/evaluation.py --name=MvP --cfg=configs/pop3d/best_model_config.yaml --data=data/pop3d-eval  --dataset=3D-POP-PATH  

```
And for the segmented version:
```
python ./evaluation/evaluation.py --name=MvP-seg --cfg=configs/pop3d-seg/best_model_config.yaml --data=data/pop3d-eval-seg  --dataset=3D-POP-PATH 

```

## Reference
```
@article{wang2021mvp,
  title={Direct Multi-view Multi-person 3D Human Pose Estimation},
  author={Tao Wang and Jianfeng Zhang and Yujun Cai and Shuicheng Yan and Jiashi Feng},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## LICENSE
This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
