import torch
import torch.nn as nn
import torchvision
import numpy as np

from ddppo_resnet.resnet_policy import PNResnetDepthEncoder

class RGBEncoder(nn.Module):
    """RGB image encoder class based on ResNet50 architecture"""
    def __init__(self, resnet_pretrain=True, trainable=False):
        """
        Initialize RGB encoder
        Args:
            resnet_pretrain: Whether to use pre-trained ResNet50 model
            trainable: Whether to allow encoder parameters to be updated during training
        """
        super(RGBEncoder, self).__init__()
        if resnet_pretrain:
            print('\nLoading Torchvision pre-trained Resnet50 for RGB ...')
        # Load ResNet50 model
        rgb_resnet = torchvision.models.resnet50(pretrained=resnet_pretrain)
        # Remove last two layers (fully connected and pooling layers), keep only feature extraction part
        rgb_modules = list(rgb_resnet.children())[:-2]
        rgb_net = torch.nn.Sequential(*rgb_modules)
        self.rgb_net = rgb_net
        # Set whether parameters are trainable
        for param in self.rgb_net.parameters():
            param.requires_grad_(trainable)

        # self.scale = 0.5  # Scale factor (commented out)

    def forward(self, rgb_imgs):
        """
        Forward propagation function
        Args:
            rgb_imgs: RGB image tensor with shape [B, N, C, H, W], where B is batch size and N is number of images per sample
        Returns:
            Processed RGB features
        """
        # Get input shape
        rgb_shape = rgb_imgs.size()
        # Reshape tensor to [B*N, C, H, W] for batch processing
        rgb_imgs = rgb_imgs.reshape(rgb_shape[0]*rgb_shape[1],
                                    rgb_shape[2], rgb_shape[3], rgb_shape[4])
        # Extract features through ResNet network
        rgb_feats = self.rgb_net(rgb_imgs)  # * self.scale

        # Debug print (commented out)
        # print('rgb_imgs', rgb_imgs.shape)
        # print('rgb_feats', rgb_feats.shape)

        # Return compressed features
        return rgb_feats.squeeze()


class DepthEncoder(nn.Module):
    """Depth image encoder class based on PointNav's pre-trained ResNet"""
    def __init__(self, resnet_pretrain=True, trainable=False):
        """
        Initialize depth encoder
        Args:
            resnet_pretrain: Whether to use pre-trained ResNet model
            trainable: Whether to allow encoder parameters to be updated during training
        """
        super(DepthEncoder, self).__init__()

        # Create PointNav depth encoder instance
        self.depth_net = PNResnetDepthEncoder()
        if resnet_pretrain:
            print('Loading PointNav pre-trained Resnet50 for Depth ...')
            # Load PointNav pre-trained weights
            ddppo_pn_depth_encoder_weights = torch.load('./data/ddppo-models/gibson-2plus-resnet50.pth')
            # Extract weights related to visual encoder
            weights_dict = {}
            for k, v in ddppo_pn_depth_encoder_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue
                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v
            # Release original weights variable to save memory
            del ddppo_pn_depth_encoder_weights
            # Load filtered weights to model
            self.depth_net.load_state_dict(weights_dict, strict=True)
        # Set whether parameters are trainable
        for param in self.depth_net.parameters():
            param.requires_grad_(trainable)

    def forward(self, depth_imgs):
        """
        Forward propagation function
        Args:
            depth_imgs: Depth image tensor with shape [B, N, H, W, 1], where B is batch size and N is number of images per sample
        Returns:
            Processed depth features
        """
        # Get input shape
        depth_shape = depth_imgs.size()
        # Reshape tensor to [B*N, H, W, 1] for batch processing
        depth_imgs = depth_imgs.reshape(depth_shape[0]*depth_shape[1],
                                    depth_shape[2], depth_shape[3], depth_shape[4])
        # Extract features through depth network
        depth_feats = self.depth_net(depth_imgs)

        # Debug print and breakpoint (commented out)
        # print('depth_imgs', depth_imgs.shape)
        # print('depth_feats', depth_feats.shape)
        #
        # import pdb; pdb.set_trace()

        return depth_feats
