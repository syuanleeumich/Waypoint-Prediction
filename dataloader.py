import glob
import numpy as np
from PIL import Image
import pickle as pkl

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Data loader and transformations
class RGBDepthPano(Dataset):
    def __init__(self, args, img_dir, navigability_dict):
        # Set input dimension constants
        # self.IMG_WIDTH = 256
        # self.IMG_HEIGHT = 256
        self.RGB_INPUT_DIM = 224  # Input dimension for RGB images
        self.DEPTH_INPUT_DIM = 256  # Input dimension for depth images
        self.NUM_IMGS = args.NUM_IMGS  # Number of images per sample
        self.navigability_dict = navigability_dict  # Navigability dictionary for filtering valid waypoints

        # RGB image transformations: convert to float and normalize
        self.rgb_transform = torch.nn.Sequential(
            # [transforms.Resize((256,341)),
            #  transforms.CenterCrop(self.RGB_INPUT_DIM),
            #  transforms.ToTensor(),]
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            )
        # Depth image transformations (commented out)
        # self.depth_transform = transforms.Compose(
        #     # [transforms.Resize((self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)),
        #     [transforms.ToTensor(),
        #     ])

        # Get all image directories
        self.img_dirs = glob.glob(img_dir)

        # Filter valid waypoint images
        for img_dir in glob.glob(img_dir):
            scan_id = img_dir.split('/')[-1][:11]  # Extract scan ID
            waypoint_id = img_dir.split('/')[-1][12:-14]  # Extract waypoint ID
            if waypoint_id not in self.navigability_dict[scan_id]:
                self.img_dirs.remove(img_dir)  # Remove waypoints not in navigability dictionary

    def __len__(self):  # Return dataset length
        return len(self.img_dirs)

    def __getitem__(self, idx):  # Get single sample data
        # Get basic sample information
        img_dir = self.img_dirs[idx]
        sample_id = str(idx)
        scan_id = img_dir.split('/')[-1][:11]  # Extract scan ID
        waypoint_id = img_dir.split('/')[-1][12:-14]  # Extract waypoint ID

        # Load RGB and depth images
        rgb_depth_img = pkl.load(open(img_dir, "rb"))
        rgb_img = torch.from_numpy(rgb_depth_img['rgb']).permute(0, 3, 1, 2)  # Adjust channel order to (B,C,H,W)
        depth_img = torch.from_numpy(rgb_depth_img['depth']).permute(0, 3, 1, 2)  # Adjust channel order to (B,C,H,W)

        # Initialize tensors for storing transformed images
        trans_rgb_imgs = torch.zeros(self.NUM_IMGS, 3, self.RGB_INPUT_DIM, self.RGB_INPUT_DIM)
        trans_depth_imgs = torch.zeros(self.NUM_IMGS, self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)

        # Initialize tensors for storing untransformed images (commented out)
        no_trans_rgb = torch.zeros(self.NUM_IMGS, 3, self.RGB_INPUT_DIM, self.RGB_INPUT_DIM, dtype=torch.uint8)
        no_trans_depth = torch.zeros(self.NUM_IMGS, self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)

        # Transform each image
        for ix in range(self.NUM_IMGS):
            trans_rgb_imgs[ix] = self.rgb_transform(rgb_img[ix])  # Apply RGB transformation
            # no_trans_rgb[ix] = rgb_img[ix]
            trans_depth_imgs[ix] = depth_img[ix][0]  # Extract first channel of depth image
            # no_trans_depth[ix] = depth_img[ix][0]

        # Create sample dictionary containing all necessary information
        sample = {'sample_id': sample_id,
                  'scan_id': scan_id,
                  'waypoint_id': waypoint_id,
                  'rgb': trans_rgb_imgs,  # Transformed RGB images
                  'depth': trans_depth_imgs.unsqueeze(-1),  # Transformed depth images with added dimension
                #   'no_trans_rgb': no_trans_rgb,
                #   'no_trans_depth': no_trans_depth,
                  }

        # Debug print code (commented out)
        # print('------------------------')
        # print(trans_rgb_imgs[0][0])
        # print(rgb_img[0].shape, rgb_img[0])
        # anivlrb

        return sample  # Return processed sample
