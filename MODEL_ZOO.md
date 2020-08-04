# Arbitrary-Oriented Object Detection with Circular Smooth Label

## Performance（deprecated）

**Due to the improvement of the code, the performance of this repo is gradually improving, so the experimental results in this file are for reference only.**

### Window Function
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Anchor | Label Mode | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 39.52 | - | H | **Pulse** | smooth L1 | 90 | 1x | × | 1X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v20.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 58.86 | - | H | **Rectangular** | smooth L1 | 90 | 1x | × | 1X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v21.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 60.15 | - | H | **Triangle** | smooth L1 | 90 | 1x | × | 1X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v22.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 63.51 | - | H | **Gaussian** | smooth L1 | 90 | 2x | × | 2X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v18.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 42.06 | - | H | **Pulse** | smooth L1 | **180** | 2x | × | 4X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v28.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 61.98 | - | H | **Rectangular** | smooth L1 | **180** | 2x | × | 2X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v23.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 57.94 | - | H | **Triangle** | smooth L1 | **180** | 2x | × | 4X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v26.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 64.50 | - | H | **Gaussian** | smooth L1 | **180** | 2x | × | 2X Quadro RTX 8000 | 1 | cfgs_res50_dota_v27.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 65.09 | - | H | **Gaussian** | smooth L1 + **atan(theta)**  | **180** | 2x | × | 2X Quadro RTX 8000 | 1 | cfgs_res50_dota_v31.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 65.44 | - | H | Gaussian | smooth L1 + atan(theta)  | 180 | 2x | × | 2X Quadro RTX 8000 | 1 | cfgs_res50_dota_v37.py |
| CSL | ResNet152_v1 **MS** | DOTA1.0 trainval | DOTA1.0 test | 70.29 | [model](https://drive.google.com/file/d/1em9_GgRn0OdNel286gYJvF8R5e8sz9ed/view?usp=sharing) | H | **Gaussian** | smooth L1 + atan(theta)  | **180** | 2x | **√** | 2X Quadro RTX 8000 | 1 | cfgs_res152_dota_v36.py |

### CSL VS Baseline
| Based Method | Angle Range | Angle Pred. | Label Mode | BR | SV | LV | SH | HA | 5-mAP | 
|:------------:|:-----------:|:-----------:|:----------:|:---:|:---:|:---:|:---:|:---:|:---:|
| RetinaNet-H | 90 | Reg. Five-Param. | - | 41.15 | 53.75 | 48.30 | 55.92 | 55.77 | 50.98 |
| RetinaNet-R | 90 | Reg. Five-Param. | - | 32.27 | 64.64 | 71.01 | 68.62 | 53.52 | 58.01 |
| RetinaNet-H | 180 | Reg. Five-Param. | - | 38.47 | 54.15 | 47.89 | 60.87 | 53.63 | 51.00 |
| RetinaNet-H | - | Reg. Eight-Param. | - | 43.97 | 58.50 | 54.79 | 65.55 | 55.65 | 55.69 |
| RetinaNet-R | 90 | Cls. | Gaussian | 35.14 | 63.21 | 73.92 | 69.49 | 55.53 | 59.46 |
| RetinaNet-H | 90 | Cls. | Pulse | 9.80 | 28.04 | 11.42 | 18.43 | 23.35 | 18.21 |
| RetinaNet-H | 90 | Cls. | Rectangular | 37.62 | 54.28 | 48.97 | 62.59 | 50.26 | 50.74 |
| RetinaNet-H | 90 | Cls. | Triangle | 37.25 | 54.45 | 44.01 | 60.03 | 52.20 | 49.59 |
| RetinaNet-H | 90 | Cls. | Gaussian | 41.03 | 59.63 | 52.57 | 64.56 | 54.64 | 54.49 |
| RetinaNet-H | 180 | Cls. | Pulse | 13.95 | 16.79 | 6.5 | 16.80 | 22.48 | 15.30 |
| RetinaNet-H | 180 | Cls. | Rectangular | 36.14 | 60.80 | 50.01 | 65.75 | 53.17 | 53.17 |
| RetinaNet-H | 180 | Cls. | Triangle | 32.69 | 47.25 | 44.39 | 54.11 | 41.90 | 44.07 |
| RetinaNet-H | 180 | Cls. | Gaussian | 41.16 | 63.68 | 55.44 | 65.85 | 55.23 | 56.21 |

## Radius
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Anchor | Label Mode | Raduius/Sigma | Reg. Loss | Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 63.51 | - | H | **Gaussian** | 4 | smooth L1  | 90 | 2x | × | 2X GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v18.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 65.45 | - | R | **Gaussian** | 4 | smooth L1  | 90 | 2x | × | 2X Quadro RTX 8000 | 1 | cfgs_res50_dota_v33.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 40.78 | - | H | **Gaussian** | 0.1 | smooth L1 | **180** | 2x | × | 2x GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v35.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 59.23 | - | H | **Gaussian** | 2 | smooth L1  | **180** | 2x | × | 2x GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v32.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.12 | - | H | **Gaussian** | 4 | smooth L1  | **180** | 2x | × | 4x GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v30.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 64.50 | - | H | **Gaussian** | 6 | smooth L1  | **180** | 2x | × | 2X Quadro RTX 8000 | 1 | cfgs_res50_dota_v27.py |
| CSL | ResNet50_v1 600->800 | DOTA1.0 trainval | DOTA1.0 test | 63.99 | - | H | **Gaussian** | 8 | smooth L1  | **180** | 2x | × | 4x GeForce RTX 2080 Ti | 1 | cfgs_res50_dota_v29.py |
| CSL | ResNet50_v1 **800->1024** | DOTA1.0 trainval | DOTA1.0 test | 63.68 | - | H | **Gaussian** | 6 | smooth L1  | **180** | 2x | × | 2X Quadro RTX 8000 | 1 | cfgs_res50_dota_v25.py |

## Scene Text Dataset
|    Backbone    |    Training data    |    Val data    |   Performance (RetinaNet-H)   |    Performance (CSL)   |GPU | Configs | 
|:------------:|:------------:|:---------:|:------------:|:---------:|:---------:|:---------:|
| ResNet101_v1 MS | ICDAR2015 train | ICDAR2015 test | 72.12 / 74.90 / 73.49 | 75.78 / 79.78 / 77.73| 2X Quadro RTX 8000 | cfgs_res101_icdar2015_v1.py |
| ResNet101_v1 MS | MLT trainval | MLT test | | | 2X Quadro RTX 8000 | |
| ResNet101_v1 MS | MLT trainval + ICDAR2015 train | ICDAR2015 test | | | 2X Quadro RTX 8000 | |
| ResNet101_v1 MS | HRSC2016 train | HRSC2016 test | | | 2X Quadro RTX 8000 | |
