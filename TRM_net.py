import torch
import torch.nn as nn
import numpy as np
import utils

from transformer.waypoint_bert import WaypointBert
from pytorch_transformers import BertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def TRM_predict(mode, args, predictor, rgb_feats, depth_feats):
    ''' Predict waypoint probabilities '''
    # Use predictor model to get visual logits
    vis_logits = predictor(rgb_feats, depth_feats)
    # Element-wise probability (use sigmoid activation to convert logits to probabilities between 0-1)
    vis_probs = torch.sigmoid(vis_logits)

    # Return different results based on mode
    if mode == 'train':
        return vis_logits  # Training mode returns logits for loss calculation
    elif mode == 'eval':
        return vis_probs, vis_logits  # Evaluation mode returns both probabilities and logits


class BinaryDistPredictor_TRM(nn.Module):
    """Transformer-based binary distribution predictor"""
    def __init__(self, args=None, hidden_dim=768, n_classes=12):
        """
        Initialize predictor
        Args:
            args: Configuration parameters
            hidden_dim: Hidden layer dimension, default 768
            n_classes: Number of output classes, default 12
        """
        super(BinaryDistPredictor_TRM, self).__init__()
        self.args = args
        self.batchsize = args.BATCH_SIZE
        self.num_angles = args.ANGLES  # Number of angles
        self.num_imgs = args.NUM_IMGS  # Number of images
        self.n_classes = n_classes  # Number of output classes
        
        # RGB feature processing network (map ResNet features to hidden_dim dimension)
        # self.visual_1by1conv_rgb = nn.Conv2d(
        #     in_channels=2048, out_channels=512, kernel_size=1)
        self.visual_fc_rgb = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod([2048,7,7]), hidden_dim),  # Flatten 2048x7x7 features and project to hidden_dim
            nn.ReLU(True),
        )
        
        # Depth feature processing network (map depth features to hidden_dim dimension)
        # self.visual_1by1conv_depth = nn.Conv2d(
        #     in_channels=128, out_channels=512, kernel_size=1)
        self.visual_fc_depth = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod([128,4,4]), hidden_dim),  # Flatten 128x4x4 features and project to hidden_dim
            nn.ReLU(True),
        )
        
        # Network for fusing RGB and depth features
        self.visual_merge = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),  # Reduce concatenated features back to hidden_dim
            nn.ReLU(True),
        )

        # Configure Transformer model
        config = BertConfig()
        config.model_type = 'visual'
        config.finetuning_task = 'waypoint_predictor'
        config.hidden_dropout_prob = 0.3
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = args.TRM_LAYER  # Number of Transformer layers
        self.waypoint_TRM = WaypointBert(config=config)  # Initialize WaypointBert model

        layer_norm_eps = config.layer_norm_eps
        # Layer normalization (commented out)
        # self.mergefeats_LayerNorm = BertLayerNorm(
        #     hidden_dim,
        #     eps=layer_norm_eps
        # )

        # Create attention mask to control information flow in Transformer
        self.mask = utils.get_attention_mask(
            num_imgs=self.num_imgs,
            neighbor=args.TRM_NEIGHBOR).to(device)

        # Visual classifier to map Transformer output to final predictions
        self.vis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,
                int(n_classes*(self.num_angles/self.num_imgs))),  # Output dimension adapted to angles and classes
        )

    def forward(self, rgb_feats, depth_feats):
        """
        Forward propagation function
        Args:
            rgb_feats: RGB features from RGB encoder
            depth_feats: Depth features from depth encoder
        Returns:
            vis_logits: Visual logits with shape [batchsize, num_angles, n_classes]
        """
        # Calculate actual number of samples in each batch
        bsi = rgb_feats.size(0) // self.num_imgs

        # Process RGB features
        # rgb_x = self.visual_1by1conv_rgb(rgb_feats)
        rgb_x = self.visual_fc_rgb(rgb_feats).reshape(
            bsi, self.num_imgs, -1)  # Reshape to [bsi, num_imgs, hidden_dim]

        # Process depth features
        # depth_x = self.visual_1by1conv_depth(depth_feats)
        depth_x = self.visual_fc_depth(depth_feats).reshape(
            bsi, self.num_imgs, -1)  # Reshape to [bsi, num_imgs, hidden_dim]

        # Fuse RGB and depth features
        vis_x = self.visual_merge(
            torch.cat((rgb_x, depth_x), dim=-1)  # Concatenate along last dimension
        )
        # Layer normalization (commented out)
        # vis_x = self.mergefeats_LayerNorm(vis_x)

        # Replicate attention mask for each batch
        attention_mask = self.mask.repeat(bsi,1,1,1)
        # Process fused features through Transformer
        vis_rel_x = self.waypoint_TRM(
            vis_x, attention_mask=attention_mask
        )

        # Apply visual classifier to get logits
        vis_logits = self.vis_classifier(vis_rel_x)
        # Reshape to [bsi, num_angles, n_classes]
        vis_logits = vis_logits.reshape(
            bsi, self.num_angles, self.n_classes)

        # Heatmap offset (make each image point to the center)
        # Concatenate latter half and first half to achieve circular shift
        vis_logits = torch.cat(
            (vis_logits[:,self.args.HEATMAP_OFFSET:,:], vis_logits[:,:self.args.HEATMAP_OFFSET,:]),
            dim=1)

        return vis_logits


class BertLayerNorm(nn.Module):
    """BERT-style layer normalization module"""
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct TensorFlow-style layer normalization module (epsilon inside square root)
        Args:
            hidden_size: Hidden layer size
            eps: Small constant for numerical stability
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Scale parameter
        self.bias = nn.Parameter(torch.zeros(hidden_size))  # Bias parameter
        self.variance_epsilon = eps  # Small constant added to variance to prevent division by zero

    def forward(self, x):
        """
        Forward propagation
        Args:
            x: Input tensor
        Returns:
            Normalized tensor
        """
        u = x.mean(-1, keepdim=True)  # Calculate mean
        s = (x - u).pow(2).mean(-1, keepdim=True)  # Calculate variance
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)  # Normalize
        return self.weight * x + self.bias  # Apply scale and bias
