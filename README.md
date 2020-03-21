# Arbitrary-Oriented Object Detection with Circular Smooth Label

## Abstract
This repo is based on [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf), and it is completed by [YangXue](https://yangxue0827.github.io/).

## Pipeline
![2](pipeline.png)

## Circular Smooth Label
![5](CSL.png)

## Performance
More results and trained models are available in the [MODEL_ZOO.md](MODEL_ZOO.md).
### DOTA1.0 (Task1)
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Anchor | Label Mode | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| [CSL](https://arxiv.org/abs/2003.05597) | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 39.52 | - | H | **Pulse** | smooth L1 | 90 | 1x | × | 1X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v20.py |
| [CSL](https://arxiv.org/abs/2003.05597) | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 58.86 | - | H | **Rectangular** | smooth L1 | 90 | 1x | × | 1X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v21.py |
| [CSL](https://arxiv.org/abs/2003.05597) | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 60.15 | - | H | **Triangle** | smooth L1 | 90 | 1x | × | 1X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v22.py |
| [CSL](https://arxiv.org/abs/2003.05597) | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 63.51 | - | H | **Gaussian** | smooth L1 | 90 | 2x | × | 2X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v18.py |
| [CSL](https://arxiv.org/abs/2003.05597) | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 42.06 | - | H | **Pulse** | smooth L1 | **180** | 2x | × | 4X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v28.py |
| [CSL](https://arxiv.org/abs/2003.05597) | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 61.98 | - | H | **Rectangular** | smooth L1 | **180** | 2x | × | 2X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v23.py |
| [CSL](https://arxiv.org/abs/2003.05597) | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 57.94 | - | H | **Triangle** | smooth L1 | **180** | 2x | × | 4X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v26.py |
| [CSL](https://arxiv.org/abs/2003.05597) | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 64.50 | - | H | **Gaussian** | smooth L1 | **180** | 2x | × | 2X Quadro RTX 8000 | 1 | cfgs_res50_dota_v27.py |
| [CSL](https://arxiv.org/abs/2003.05597) | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 65.09 | - | H | **Gaussian** | smooth L1 + **atan(theta)**  | **180** | 2x | × | 2X Quadro RTX 8000 | 1 | cfgs_res50_dota_v31.py |
| [CSL](https://arxiv.org/abs/2003.05597) | ResNet152_v1 **MS** | DOTA1.0 trainval | DOTA1.0 test | 70.29 | [model](https://drive.google.com/file/d/1em9_GgRn0OdNel286gYJvF8R5e8sz9ed/view?usp=sharing) | H | **Gaussian** | smooth L1 + atan(theta)  | **180** | 2x | **√** | 2X Quadro RTX 8000 | 1 | cfgs_res152_dota_v36.py |

### Visualization
![1](demo1.png)

## My Development Environment
**docker images: docker pull yangxue2docker/yx-tf-det:tensorflow1.13.1-cuda10-gpu-py3**      
1、python3.5 (anaconda recommend)               
2、cuda 10.0                     
3、[opencv(cv2)](https://pypi.org/project/opencv-python/)       
4、[tfplot 0.2.0](https://github.com/wookayin/tensorflow-plot) (optional)            
5、tensorflow 1.13                      

## Download Model
### Pretrain weights
1、Please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz), [resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) pre-trained models on Imagenet, put it to data/pretrained_weights.       
2、**(Recommend)** Or you can choose to use a better backbone, refer to [gluon2TF](https://github.com/yangJirui/gluon2TF).    
* [Baidu Drive](https://pan.baidu.com/s/1GpqKg0dOaaWmwshvv1qWGg), password: 5ht9.          
* [Google Drive](https://drive.google.com/drive/folders/1BM8ffn1WnsRRb5RcuAcyJAHX8NS2M1Gz?usp=sharing)      

## Compile
```  
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace (or make)

cd $PATH_ROOT/libs/box_utils/
python setup.py build_ext --inplace
```

## Train

1、If you want to train your own data, please note:  
```     
(1) Modify parameters (such as CLASS_NUM, DATASET_NAME, VERSION, etc.) in $PATH_ROOT/libs/configs/cfgs.py
(2) Add category information in $PATH_ROOT/libs/label_name_dict/lable_dict.py     
(3) Add data_name to $PATH_ROOT/data/io/read_tfrecord.py 
```     

2、Make tfrecord     
For DOTA dataset:      
```  
cd $PATH_ROOT\data\io\DOTA
python data_crop.py
```  

```  
cd $PATH_ROOT/data/io/  
python convert_data_to_tfrecord.py --VOC_dir='/PATH/TO/DOTA/' 
                                   --xml_dir='labeltxt'
                                   --image_dir='images'
                                   --save_name='train' 
                                   --img_format='.png' 
                                   --dataset='DOTA'
```      

3、Multi-gpu train
```  
cd $PATH_ROOT/tools
python multi_gpu_train.py
```

## Eval
```  
cd $PATH_ROOT/tools
python test_dota.py --test_dir='/PATH/TO/IMAGES/'  
                    --gpus=0,1,2,3,4,5,6,7  
                    --s (visualization, optional)
                    --ms (multi-scale test, optional)
``` 

## Tensorboard
```  
cd $PATH_ROOT/output/summary
tensorboard --logdir=.
``` 

![3](images.png)

![4](scalars.png)

## Citation

If this is useful for your research, please consider cite.

```
@article{yang2020arbitrary,
    title={Arbitrary-Oriented Object Detection with Circular Smooth Label},
    author={Yang, Xue and Yan, Junchi},
    journal={arXiv preprint arXiv:2003.05597},
    year={2020}
}

@inproceedings{xia2018dota,
    title={DOTA: A large-scale dataset for object detection in aerial images},
    author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={3974--3983},
    year={2018}
}
```


## Reference
1、https://github.com/endernewton/tf-faster-rcnn   
2、https://github.com/zengarden/light_head_rcnn   
3、https://github.com/tensorflow/models/tree/master/research/object_detection    
4、https://github.com/fizyr/keras-retinanet     
