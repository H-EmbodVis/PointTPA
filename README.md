<div align="center">
  <h1>PointTPA: Dynamic Network Parameter Adaptation <br>
    for 3D Scene Understanding </h1>


  <a href="https://arxiv.org/abs/2604.04933"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=Arxiv"></a> 
  <a href="https://this-yq.github.io/PointTPA/"><img src="https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"></a>


[Siyuan Liu](syliu@hust.edu.cn)</sup>\*</sup>, [Chaoqun Zheng](cqzheng@hust.edu.cn)</sup>\*</sup>, [Xin Zhou](https://lmd0311.github.io/), [Tianrui Feng](tianruifeng@hust.edu.cn), [Dingkang Liang](https://dk-liang.github.io/)</sup>†</sup> and [Xiang Bai](https://scholar.google.com/citations?user=UeltiQ4AAAAJ&hl=en)

 


  Huazhong University of Science and Technology, Wuhan, China

(*) equal contribution, (​†​) corresponding author.
</div>

---

## News
- **[06/Apr/2026]** ✨ Release the code. 😊😊
- **[21/Feb/2026]** 🎉 Our paper PointTPA is accepted by **CVPR 2026**! 🥳🥳

---
## Abstract
<p align="justify">
Scene-level point cloud understanding remains challenging due to diverse geometries, imbalanced category distributions, and highly varied spatial layouts. Existing methods improve object-level performance but rely on static network parameters during inference, limiting their adaptability to dynamic scene data. We propose PointTPA, a <b>T</b>est-time <b>P</b>arameter <b>A</b>daptation framework that generates input-aware network parameters for scene-level point clouds. PointTPA adopts a Serialization-based Neighborhood Grouping (SNG) to form locally coherent patches and a Dynamic Parameter Projector (DPP) to produce patch-wise adaptive weights, enabling the backbone to adjust its behavior according to scene-specific variations while maintaining a low parameter overhead. Integrated into the PTv3 structure, PointTPA demonstrates strong parameter efficiency by introducing two lightweight modules of less than 2% of the backbone's parameters. Despite this minimal parameter overhead, PointTPA achieves 78.4% mIoU on ScanNet validation, surpassing existing parameter-efficient fine-tuning (PEFT) methods across multiple benchmarks, highlighting the efficacy of our test-time dynamic network parameter adaptation mechanism in enhancing 3D scene understanding.
</p>

---

- [📌 Overview](#1-overview)
- [📊 Results](#2-results)
- [⚙️ Get Started](#3-get-started)
- [🙏 Acknowledgement](#4-acknowledgement)
- [📖 Citation](#5-citation)

---

## 1. Overview

<div align="center">
<p align="center">
  <img src="assets/pipline-1.png" alt="Overview" style="max-width:80%; height:auto;" />
</p>

<p align="center">

</div>

---

## 2. Results
### 2.1 Main Results
<p align="center">
  <img src="assets/Results.png" alt="Over" style="max-width:40%; height:auto;" />
</p>

### 2.2 Time Comparison
<p align="center">
  <img src="assets/time_comp_01.png" alt="Overview" style="max-width:40%; height:auto;" />
</p>

### 2.3 Visualization
<div align="center" style="display:flex; width:100%; gap:2%; justify-content:center;">
  <img src="assets/comp_01.png" alt="vs" style="width:49%; height:280px; object-fit:contain; object-position:center;" />
  <img src="assets/supp_comp_01(1).png" alt="v2" style="width:49%; height:280px; object-fit:contain; object-position:center;" />
</div>



---

## 3. Get Started
### 3.1 Installation
We recommend using Anaconda to set up the environment for this project. You may also refer to [Pointcept](https://github.com/pointcept/pointcept#installation) for additional environment configuration details.
```shell
$ git clone https://github.com/Ykzzldx2435/PointTPA.git
$ cd PointTPA
# Create virtual env 
$ conda env create -f environment.yml
$ conda activate PointTPA
```
### 3.2 Train <br>
The required datasets should be prepared in the `data` directory in advance. Detailed preparation instructions are provided in the [Pointcept documentation](https://github.com/Pointcept/Pointcept#data-preparation).<br>

#### 3.2.1 Link the Processed Datasets
```shell
$mkdir data
$ln -s ${PROCESSED_SCANNET_DIR} ${CODEBASE_DIR}/data/{DATASET NAME} # (e.g. scannet s3dis scannetpp)
```
#### 3.2.2 Project Structure
With the datasets linked properly, the project directory is expected to have the following structure:
```
PointTPA/
├── data/                                          # symlink to processed datasets
│   ├── scannet/
│   ├── s3dis/
│   └── scannetpp/
├── configs/
|   ├── _base_/
|   └── ptv3-pointtpa/                             # training and testing configuration            
├── pointcept/
|   ├── ...
|   ├── datasets/
|   ├── engines/
|   └── models/
|      ├── ...
|      ├── peft/
|      |   ├── __init__.py
|      |   └── pointtpa.py                         # core modules
|      └── point_transformer_v3/
|          ├── __init__.py
|          ├── point_transformer_v3.py             # backbone
|          └── point_transformer_v3_pointtpa.py    # PointTPA       
├── scripts/                            
├── tools/
├── libs/                  
├── environment.yml
├── LICENSE
└── README.md
```

#### 3.2.3 Quick Start<br>
Use the following commands to train or evaluate the model from the terminal.<br>
##### ScanNet
```shell
# Linear Probing
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannet-lin -n semseg-ptv3-scannet-lin -w /path/to/pretrain_weight
# Lin + PointTPA
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannet-pointtpa -n semseg-ptv3-scannet-pointtpa -w /path/to/pretrain_weight
# Decoder Probing
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannet-dec -n semseg-ptv3-scannet-dec -w /path/to/pretrain_weight
# Dec with PointTPA
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannet-pointtpa-dec -n semseg-ptv3-scannet-pointtpa-dec -w /path/to/pretrain_weight
# FFT
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannet-ft -n semseg-ptv3-scannet-ft -w /path/to/pretrain_weight
```

##### ScanNet200
```shell
# Linear Probing
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannet200-lin -n semseg-ptv3-scannet200-lin -w /path/to/pretrain_weight
# Lin + PointTPA
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannet200-pointtpa -n semseg-ptv3-scannet200-pointtpa -w /path/to/pretrain_weight
# Decoder Probing
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannet200-dec -n semseg-ptv3-scannet200-dec -w /path/to/pretrain_weight
# Dec with PointTPA
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannet200-pointtpa-dec -n semseg-ptv3-scannet200-pointtpa-dec -w /path/to/pretrain_weight
# FFT
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannet200-ft -n semseg-ptv3-scannet200-ft -w /path/to/pretrain_weight
```

##### S3DIS
```shell
# Linear Probing
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-s3dis-lin -n semseg-ptv3-s3dis-lin -w /path/to/pretrain_weight
# Lin + PointTPA
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-s3dis-pointtpa -n semseg-ptv3-s3dis-pointtpa -w /path/to/pretrain_weight
# Decoder Probing
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-s3dis-dec -n semseg-ptv3-s3dis-dec -w /path/to/pretrain_weight
# Dec with PointTPA
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-s3dis-pointtpa-dec -n semseg-ptv3-s3dis-pointtpa-dec -w /path/to/pretrain_weight
# FFT
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-s3dis-ft -n semseg-ptv3-s3dis-ft -w /path/to/pretrain_weight
```

##### ScanNet++
```shell
# Linear Probing
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannetpp-lin -n semseg-ptv3-scannetpp-lin -w /path/to/pretrain_weight
# Lin + PointTPA
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannetpp-pointtpa -n semseg-ptv3-scannetpp-pointtpa -w /path/to/pretrain_weight
# Decoder Probing
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannetpp-dec -n semseg-ptv3-scannetpp-dec -w /path/to/pretrain_weight
# Dec with PointTPA
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannetpp-pointtpa-dec -n semseg-ptv3-scannetpp-pointtpa-dec -w /path/to/pretrain_weight
# FFT
$sh scripts/train.sh -m 1 -g 2 -d ptv3-pointtpa -c semseg-ptv3-scannetpp-ft -n semseg-ptv3-scannetpp-ft -w /path/to/pretrain_weight
```
The pretrained Sonata weights are available for download [HERE](https://huggingface.co/facebook/sonata/blob/main/pretrain-sonata-v1m1-0-base.pth).

---

## 4. Acknowledgement
This project is based on [Sonata](https://arxiv.org/abs/2503.16429) , [PTv3](https://arxiv.org/abs/2312.10035), and also references DAPT ([paper](https://arxiv.org/abs/2403.01439), [code](https://github.com/LMD0311/DAPT)), PointGST ([paper](https://arxiv.org/abs/2410.08114), [code](https://github.com/jerryfeng2003/pointgst)), IDPT ([paper](https://arxiv.org/abs/2304.07221), [code](https://github.com/zyh16143998882/ICCV23-IDPT)) and VeRA ([paper](https://arxiv.org/abs/2310.11454), [code](https://huggingface.co/docs/peft/main/en/package_reference/vera)). Our code organization follows the Pointcept repository ( [Pointcept](https://github.com/Pointcept/Pointcept) ). We thank the authors for their excellent work.


---

## 5. Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation.
```
@inproceedings{liu2026pointtpa,
        title={PointTPA: Dynamic Network Parameter Adaptation for 3D Scene Understanding},
        author={Liu, Siyuan and Zheng, Chaoqun and Zhou, Xin and Feng, Tianrui and Liang, Dingkang and Bai, Xiang},
        booktitle={Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition},
        year={2026}
}
```
