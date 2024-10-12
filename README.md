<p align="center">
  <img src="web-img/task_illustration_v5.png" width="700">
  <h1 align="center">From Cognition to Precognition: A Future-Aware Framework for Social Navigation</h1>
  <h3 align="center">
    <a href="https://zeying-gong.github.io/">Zeying Gong</a>, <a href="https://hutslib.github.io/">Tianshuai Hu</a>, <a href="https://precognition.team/">Ronghe Qiu</a>, <a href="https://junweiliang.me/">Junwei Liang</a>
  </h3>
  <p align="center">
    <a href="https://zeying-gong.github.io/projects/falcon/">Project Website</a> , <a href="https://arxiv.org/abs/2409.13244">Paper (ArXiv)</a>
  </p>
  <p align="center">
    <a href="https://github.com/Zeying-Gong/habitat-lab">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
    </a>
    <a href="https://zeying-gong.github.io/projects/falcon/">
      <img src="https://img.shields.io/badge/Falcon-Link-42ba94.svg">
    </a>
    <a href="https://arxiv.org/abs/2409.13244">
      <img src="https://img.shields.io/badge/arXiv-2409.13244-red.svg" />
    </a>
    <a href="https://github.com/facebookresearch/habitat-sim">
      <img src="https://img.shields.io/static/v1?label=supports&message=Habitat%20Sim&color=informational&link=https://github.com/facebookresearch/habitat-sim">
    </a>
  </p>
</p>

## :sparkles: Overview

To navigate safely and efficiently in crowded spaces, robots should not only perceive the current state of the environment but also anticipate future human movements. 
In this paper, we propose a reinforcement learning architecture, namely **Falcon**, to tackle socially-aware navigation by explicitly predicting human trajectories and penalizing actions that block future human paths. 
To facilitate realistic evaluation, we introduce a novel SocialNav benchmark containing two new datasets, Social-HM3D and Social-MP3D. 
This benchmark offers large-scale photo-realistic indoor scenes populated with a reasonable amount of human agents based on scene area size, incorporating natural human movements and trajectory patterns. 
We conduct a detailed experimental analysis with the state-of-the-art learning-based method and two classic rule-based path-planning algorithms on the new benchmark. 
The results demonstrate the importance of future prediction and our method achieves the best task success rate of 55% while maintaining about 90% personal space compliance.

## :hammer_and_wrench: Installation

### Getting Started

#### 1. **Preparing conda env**

Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
```
conda_env_name=falcon
conda create -n $conda_env_name python=3.9 cmake=3.14.0
conda activate $conda_env_name
```

#### 2. **conda install habitat-sim & habitat-lab**
Following [Habitat-lab](https://github.com/facebookresearch/habitat-lab.git)'s instruction:
```
conda install habitat-sim withbullet -c conda-forge -c aihabitat
```

Then, assuming you have [this repositories](https://github.com/Zeying-Gong/habitat-lab) cloned (forked from Habitat 3.0), install necessary dependencies of Habitat.
```
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

#### 3. **Downloading the Social-HM3D & Social-MP3D datasets**

- Download Scene Datasets

Following the instructions for **HM3D** and **MatterPort3D** in Habitat-lab's [Datasets.md](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).

- Download Episode Datasets

Download social navigation (SocialNav) episodes for the test scenes, which can be found here: [Link](https://drive.google.com/drive/folders/1V0a8PYeMZimFcHgoJGMMTkvscLhZeKzD?usp=drive_link).

After downloading, unzip and place the datasets in the default location:
```
unzip -d data/datasets/pointnav
```

The file structure should look like this:
```
data
└── datasets
    └── pointnav
        ├── social-hm3d
        │   ├── train
        │   │   ├── content
        │   │   └── train.json.gz
        │   └── val
        │       ├── content
        │       └── val.json.gz
        └── social-mp3d
            ├── train
            │   ├── content
            │   └── train.json.gz
            └── val
                ├── content
                └── val.json.gz
```

Note that here the definition of SocialNav is different from the original task in [Habitat 3.0](https://arxiv.org/abs/2310.13724).



## :arrow_forward: Evaluation 

### Two classic rule-based methods (ASTAR & ORCA)

In this paper, two rule-based methods are used for evaluation:

- **[ASTAR](https://ieeexplore.ieee.org/document/4082128)**: A well-known pathfinding algorithm that finds the shortest path using a heuristic to estimate the cost.

- **[ORCA](https://gamma.cs.unc.edu/ORCA/publications/ORCA.pdf)**: A multi-agent navigation algorithm designed for collision-free movement through reciprocal avoidance.

You can evaluate Astar or ORCA on the Social-HM3D or Social-MP3D datasets using the following template:

```
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=social_nav_v2/<algorithm>_<dataset>.sh
```

For example, to run Astar on the Social-HM3D dataset:

```
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=social_nav_v2/astar_hm3d.sh
```

If you wish to generate videos, simply add the `habitat_baselines.eval.video_option=["disk"]` to the end of the command. For instance, to run Astar on the Social-HM3D dataset and record videos:

```
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=social_nav_v2/astar_hm3d.sh \
habitat_baselines.eval.video_option=["disk"]
```

### Two RL-based methods (Proximity & *Falcon(ours)*)

**TODO: this section is work in progress.** The code of Proximity can be found in [this link](https://github.com/EnricoCancelli/ProximitySocialNav).

## :black_nib: Citation

If you find this repository useful in your research, please consider citing our paper:

```
@misc{gong2024cognitionprecognitionfutureawareframework,
  title={From Cognition to Precognition: A Future-Aware Framework for Social Navigation}, 
  author={Zeying Gong and Tianshuai Hu and Ronghe Qiu and Junwei Liang},
  year={2024},
  eprint={2409.13244},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2409.13244},  
}
```