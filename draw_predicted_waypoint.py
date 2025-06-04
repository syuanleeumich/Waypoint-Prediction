import torch
import argparse
from dataloader import RGBDepthPano

from image_encoders import RGBEncoder, DepthEncoder
from TRM_net import BinaryDistPredictor_TRM, TRM_predict

from eval import waypoint_eval

import os
import glob
import utils
import random
from utils import nms
from utils import print_progress
from tensorboardX import SummaryWriter

import numpy as np
from gen_training_data.utils import Simulator
from test_gtmap import create_simulator
import draw
from habitat.utils.visualizations import maps as habitat_maps
import json
from collections import defaultdict
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Args:
    def __init__(self):
        self.EXP_ID = 'eval_ipynb'
        self.TRAINEVAL = 'eval'
        self.VIS = 0
        self.ANGLES = 120
        self.NUM_IMGS = 12
        self.NUM_CLASSES = 12
        self.MAX_NUM_CANDIDATES = 5
        self.PREDICTOR_NET = 'TRM'
        self.EPOCH = 300
        self.BATCH_SIZE = 8
        self.LEARNING_RATE = 1e-6
        self.WEIGHT = 0
        self.TRM_LAYER = 2
        self.TRM_NEIGHBOR = 1
        self.HEATMAP_OFFSET = 5
        self.HIDDEN_DIM = 768

args = Args()

rgb_encoder = RGBEncoder(resnet_pretrain=True, trainable=False).to(device)
depth_encoder = DepthEncoder(resnet_pretrain=True, trainable=False).to(device)
predictor = BinaryDistPredictor_TRM(args=args, hidden_dim=args.HIDDEN_DIM, n_classes=args.NUM_CLASSES).to(device)

nav_dict_path = './training_data/%s_*_mp3d_waypoint_twm0.2_obstacle_first_withpos.json'%(args.ANGLES)
navigability_dict = utils.load_gt_navigability(nav_dict_path)
eval_img_dir = './training_data/rgbd_fov90/val_unseen/*/*.pkl'  # Evaluation image directory
evaldataloader = RGBDepthPano(args, eval_img_dir, navigability_dict)  # Evaluation data loader
evalloader = torch.utils.data.DataLoader(evaldataloader, 
        batch_size=args.BATCH_SIZE, shuffle=False, num_workers=4) 
raw_graph_data = {}

for split in ['train', 'val_unseen']:
    path = f'./data/adapted_mp3d_connectivity_graphs/{split}.json'
    with open(path) as f:
        data = json.load(f)
        raw_graph_data.update(data) 

criterion_mse = torch.nn.MSELoss(reduction='none')  # Mean squared error loss
params = list(predictor.parameters())  # Get predictor parameters
optimizer = torch.optim.AdamW(params, lr=args.LEARNING_RATE)  # Use AdamW optimizer

checkpoint_load_path = './checkpoints/wp-train/snap/check_val_best_avg_pred_distance'
epoch, predictor, optimizer = utils.load_checkpoint(
                        predictor, optimizer, checkpoint_load_path)
rgb_encoder.eval()
depth_encoder.eval()
predictor.eval()

config_path = 'gen_training_data/config.yaml'
scene_path = './data/scene_datasets/mp3d/{scan}/{scan}.glb'

cur_scan_id = None

for _, data in enumerate(evalloader):
    scan_ids = data['scan_id']
    waypoint_ids = data['waypoint_id']
    sample_id = data['sample_id']
    rgb_imgs = data['rgb'].to(device)
    depth_imgs = data['depth'].to(device)

    target, obstacle, weight, \
    source_pos, target_pos = utils.get_gt_nav_map(
                args.ANGLES, navigability_dict, scan_ids, waypoint_ids)
    
    target = target.to(device)
    obstacle = obstacle.to(device)
    weight = weight.to(device)

    rgb_feats = rgb_encoder(rgb_imgs)        # Extract RGB features
    depth_feats = depth_encoder(depth_imgs)  # Extract depth features
    vis_probs, vis_logits = TRM_predict('eval', args, predictor, rgb_feats, depth_feats)
    
    if cur_scan_id is None or scan_ids[0] != cur_scan_id:
        cur_scan_id = scan_ids[0]
        sim = create_simulator(config_path, scene_path, cur_scan_id)
    for i in range(len(scan_ids)):
        scan_id = scan_ids[i]

        if scan_id != cur_scan_id:
            sim.close()
            cur_scan_id = scan_id
            sim = create_simulator(config_path, scene_path, cur_scan_id)

        waypoint_id = waypoint_ids[i]
        cur_source_pos = source_pos[i]
        cur_target = target[i]
        cur_vis_probs = vis_probs[i]
        cur_vis_logits = vis_logits[i]
        print(f'Processing sample: scan={scan_id}, waypoint={waypoint_id}')
        
        target_peak = (cur_target > 9).to(torch.uint8)
        vis_logits_batch = cur_vis_logits.unsqueeze(0)  # Add batch dimension while preserving original shape
        vis_logits_batch_wrap = torch.cat(
            (vis_logits_batch[:,-1:,:], vis_logits_batch, vis_logits_batch[:,:1,:]), 
            dim=1)
        predict_nms = utils.nms(
            vis_logits_batch_wrap.unsqueeze(1), max_predictions=5,
            sigma=(7.0,5.0))
        predict_nms = predict_nms.squeeze()[1:-1,:]
        num_waypoints = (predict_nms > 0).sum().item()

        node_height = raw_graph_data[scan_id]['nodes'][waypoint_id][1]
        source_pos_complete = np.array([cur_source_pos[0], node_height, cur_source_pos[1]])
        top_down_map = draw.get_top_down_map(sim, base_height=node_height)
        source_pos_index = habitat_maps.to_grid(cur_source_pos[1], cur_source_pos[0], top_down_map.shape[0:2], sim)
        draw.draw_source_new(top_down_map, cur_source_pos, 0.05, sim)

        current_heading = 0.0
        predict_waypoints_radial = draw.get_waypoints_from_radial_map(predict_nms, 120, 12)
        for r, theta in predict_waypoints_radial:
            pos = draw._rtheta_to_global_coordinates(r, theta, cur_source_pos, current_heading)
            draw.draw_waypoint_prediction_new(top_down_map, pos, 0.05, sim)

        target_radial = draw.get_waypoints_from_radial_map(target_peak, 120, 12)
        for r, theta in target_radial:
            pos = draw._rtheta_to_global_coordinates(r, theta, cur_source_pos, current_heading)
            draw.draw_oracle_waypoint_new(top_down_map, pos, 0.05, sim)

        color_map = draw.colorize_topdown_map(top_down_map)
        crop_image = draw.crop_around_point(color_map, source_pos_index[0], source_pos_index[1], padding=200)

        save_dir = f'waypoints_visualize/{scan_id}/{waypoint_id}'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'crop_image.png')
        plt.imsave(save_path, crop_image)


