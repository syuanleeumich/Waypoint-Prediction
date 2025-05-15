import glob
import numpy as np
from PIL import Image
import pickle as pkl

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 数据加载器和变换
class RGBDepthPano(Dataset):
    def __init__(self, args, img_dir, navigability_dict):
        # 设置输入维度常量
        # self.IMG_WIDTH = 256
        # self.IMG_HEIGHT = 256
        self.RGB_INPUT_DIM = 224  # RGB图像的输入维度
        self.DEPTH_INPUT_DIM = 256  # 深度图像的输入维度
        self.NUM_IMGS = args.NUM_IMGS  # 每个样本的图像数量
        self.navigability_dict = navigability_dict  # 可导航性字典，用于筛选有效路径点

        # RGB图像变换：转换为浮点型并进行标准化
        self.rgb_transform = torch.nn.Sequential(
            # [transforms.Resize((256,341)),
            #  transforms.CenterCrop(self.RGB_INPUT_DIM),
            #  transforms.ToTensor(),]
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            )
        # 深度图像变换（已注释）
        # self.depth_transform = transforms.Compose(
        #     # [transforms.Resize((self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)),
        #     [transforms.ToTensor(),
        #     ])

        # 获取所有图像目录
        self.img_dirs = glob.glob(img_dir)

        # 筛选有效的路径点图像
        for img_dir in glob.glob(img_dir):
            scan_id = img_dir.split('/')[-1][:11]  # 提取扫描ID
            waypoint_id = img_dir.split('/')[-1][12:-14]  # 提取路径点ID
            if waypoint_id not in self.navigability_dict[scan_id]:
                self.img_dirs.remove(img_dir)  # 移除不在可导航字典中的路径点

    def __len__(self):  # 返回数据集长度
        return len(self.img_dirs)

    def __getitem__(self, idx):  # 获取单个样本数据
        # 获取样本基本信息
        img_dir = self.img_dirs[idx]
        sample_id = str(idx)
        scan_id = img_dir.split('/')[-1][:11]  # 提取扫描ID
        waypoint_id = img_dir.split('/')[-1][12:-14]  # 提取路径点ID

        # 加载RGB和深度图像
        rgb_depth_img = pkl.load(open(img_dir, "rb"))
        rgb_img = torch.from_numpy(rgb_depth_img['rgb']).permute(0, 3, 1, 2)  # 调整通道顺序为(B,C,H,W)
        depth_img = torch.from_numpy(rgb_depth_img['depth']).permute(0, 3, 1, 2)  # 调整通道顺序为(B,C,H,W)

        # 初始化张量用于存储变换后的图像
        trans_rgb_imgs = torch.zeros(self.NUM_IMGS, 3, self.RGB_INPUT_DIM, self.RGB_INPUT_DIM)
        trans_depth_imgs = torch.zeros(self.NUM_IMGS, self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)

        # 初始化张量用于存储未变换的图像（已注释掉）
        no_trans_rgb = torch.zeros(self.NUM_IMGS, 3, self.RGB_INPUT_DIM, self.RGB_INPUT_DIM, dtype=torch.uint8)
        no_trans_depth = torch.zeros(self.NUM_IMGS, self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)

        # 对每个图像进行变换
        for ix in range(self.NUM_IMGS):
            trans_rgb_imgs[ix] = self.rgb_transform(rgb_img[ix])  # 应用RGB变换
            # no_trans_rgb[ix] = rgb_img[ix]
            trans_depth_imgs[ix] = depth_img[ix][0]  # 提取深度图第一个通道
            # no_trans_depth[ix] = depth_img[ix][0]

        # 创建样本字典，包含所有必要信息
        sample = {'sample_id': sample_id,
                  'scan_id': scan_id,
                  'waypoint_id': waypoint_id,
                  'rgb': trans_rgb_imgs,  # 变换后的RGB图像
                  'depth': trans_depth_imgs.unsqueeze(-1),  # 变换后的深度图像，增加一个维度
                #   'no_trans_rgb': no_trans_rgb,
                #   'no_trans_depth': no_trans_depth,
                  }

        # 调试打印代码（已注释掉）
        # print('------------------------')
        # print(trans_rgb_imgs[0][0])
        # print(rgb_img[0].shape, rgb_img[0])
        # anivlrb

        return sample  # 返回处理后的样本
