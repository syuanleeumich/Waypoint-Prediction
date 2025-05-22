## Install Guideline
### Prepare Data
```bash
mkdir data
```
#### Adaptive Connectivity Graph

使用 gdown 下载数据：
```bash
# 安装 gdown
pip install gdown

# 下载数据到 data 目录并命名为adapted_mp3d_connectivity_graphs
cd data
gdown https://drive.google.com/drive/folders/1wpuGAO-rRalPKt8m1-QIvlb_Pv1rYJ4x --folder --output adapted_mp3d_connectivity_graphs
```
#### scene_datasets
```bash
# 下载场景数据集
gdown 1f5icTuonvwW12tQg55FXR3Y1bCsL-9r6 -O scene_datasets.zip
unzip scene_datasets.zip
```
#### ResNet-50 Depth Encoder pretrained weights
```bash
# 下载预训练权重
mkdir -p ddppo-models
gdown https://zenodo.org/record/6634113/files/gibson-2plus-resnet50.pth -O ddppo-models/gibson-2plus-resnet50.pth
cd ..
```

#### 
### Python 3.8
原论文使用的是python3.6, 但是python3.6不支持2.0及以上版本的pytorch, 不能使用cuda 12.1以上的版本，在运行时会warning cuda 11.X与H100或者L40不兼容，所以改用python3.8.

```bash
conda create -n  test_dcvln python=3.8
conda activate test_dcvln
```

### habitat-sim
可能因为版本过旧，使用原文指令并不能正确安装gpu版本的headless habitat-sim. 
```bash
#不要用这个
conda install habitat-sim=0.1.7 headless -c conda-forge -c aihabitat
```
参考<https://github.com/jzhzhang/NaVid-VLN-CE/issues/6>使用conda官网提供的包，下载<https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2>,然后conda install
```bash
gdown https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2 -O habitat-sim-0.1.7-headless.tar.bz2
conda install habitat-sim-0.1.7-headless.tar.bz2
```

### Requirements
由于从python3.6转到了python3.8, requirements.txt需要改动. 由于3.8的setuptools和pip更新，需要先降级单独安装gym==0.21.0. 3.8不再支持tensorflow==1.13.1和tb-nightly，改为tensorboardX>=2.0. opencv使用预编译版本，否则编译太慢.
```bash
pip install 'setuptools<60' 'pip<23'
pip install gym==0.21.0
```
这里可能有关于opencv的warning,不用管,只是gym里面的setup写的不规范。然后安装其他包。
```bash
pip install --upgrade setuptools pip
pip install -r requirements38.txt
```
### Pytorch
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### Habitat lab
```bash
# 克隆特定版本的 habitat-lab
cd ..
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.1.7
python setup.py develop --all
cd ..
cd Waypoint-Prediction
```
可能会有关于tensorflow和tb-nightly的warning,不用管

检查是否安装成功
```bash
python -c "import habitat_sim; print('habitat-sim version:', habitat_sim.__version__)"
python -c "import habitat; print
('habitat-lab version:', habitat.__version__)"
```

### reinstall numpy
强制把numpy改为1.19.5版本，否则float会不兼容
```bash
pip install numpy==1.19.5 matplotlib==3.3.4 numba==0.53.1 gitpython==3.1.44
pip install absl-py braceexpand objectio simplejson scikit-image==0.17.2
```

## Generate Training Data
### Generate Training Image
Modify SPLIT in gen_training_data/get_images_inputs.py to 'train'
```bash
python gen_training_data/get_images_inputs.py
```
Modify SPLIT in gen_training_data/get_images_inputs.py to 'val_unseen'
```bash
python gen_training_data/get_images_inputs.py
```

### Generate Nav Dict
Modify SPLIT in gen_training_data/get_nav_dict.py to 'train'
```bash
python gen_training_data/get_nav_dict.py
```
Modify SPLIT in gen_training_data/get_nav_dict.py to 'val_unseen'
```bash
python gen_training_data/get_nav_dict.py
```

### Generate Ground Truth Dict
```bash
python gen_training_data/test_twm0.2_obstacle_first.py
```

## Train
Modify EXP_ID in run_waypoints.bash
```bash
bash run_waypoint.bash
```