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
import quaternion
import open3d as o3d
import math 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_full_scene_pcd(depth, hfov):
    height, width = depth.shape

    cx = (width - 1.) / 2.
    cy = (height - 1.) / 2.
    fx = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    # fy = (height / 2.) / np.tan(np.deg2rad(self.args.hfov / 2.))

    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)
    
    # Apply the mask, and unprojection is done only on the valid points
    valid_mask = depth > 0
    masked_depth = depth[valid_mask]
    u = u[valid_mask]
    v = v[valid_mask]

    # Convert to 3D coordinates
    x = (u - cx) * masked_depth / fx
    y = (v - cy) * masked_depth / fx
    z = masked_depth

    # Stack x, y, z coordinates into a 3D point cloud
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)
    
    # Perturb the points a bit to avoid colinearity
    points += np.random.normal(0, 4e-3, points.shape)

    color_mask = np.repeat(valid_mask[:, :, np.newaxis], 3, axis=2)
    # image_flat = image[color_mask].reshape(-1, 3)  # Flatten the image array for easier indexing
    # colors = image_flat / 255.0  # Normalize the colors
    
    # TODO: we can get per pixel label for stair here as well.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    
    camera_object_pcd = pcd.voxel_down_sample(0.05)

    return camera_object_pcd

def get_transform_matrices(sensor_position, sensor_rotation, init_sim_position, init_sim_rotation):
    """
    transform the habitat-lab space to Open3D space (initial pose in habitat)
    habitat-lab space need to rotate camera from x,y,z to  x, -y, -z
    Returns Pose_diff, R_diff change of the agent relative to the initial timestep
    """


    camera_position = sensor_position#agent_state.sensor_states["depth_90"].position
    camera_rotation = quaternion.as_rotation_matrix(sensor_rotation)#agent_state.sensor_states["depth_90"].rotation)

    h_camera_matrix = np.eye(4)
    h_camera_matrix[:3, :3] = camera_rotation
    h_camera_matrix[:3, 3] = camera_position

    habitat_camera_self = np.eye(4)
    habitat_camera_self[:3, :3] = np.array([[1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]])
    habitat_camera_self_aj = np.eye(4)
    habitat_camera_self_aj[:3, :3] = np.array([[0, 0, -1],
                [0, 1, 0],
                [1, 0, 0]])
    
    R_habitat2open3d = np.eye(4)
    R_habitat2open3d[:3, :3] = quaternion.as_rotation_matrix(init_sim_rotation)
    R_habitat2open3d[:3, 3] = init_sim_position

    camera_pose = habitat_camera_self_aj @ np.linalg.inv(R_habitat2open3d) @ h_camera_matrix
    O_camera_matrix = habitat_camera_self_aj @ np.linalg.inv(R_habitat2open3d) @ h_camera_matrix @ habitat_camera_self

    return O_camera_matrix

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
evaldataloader = RGBDepthPano(args.NUM_IMGS, eval_img_dir, navigability_dict)  # Evaluation data loader
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

checkpoint_load_path = './checkpoints/train_0/snap/check_val_best_avg_pred_distance'
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
    if scan_ids[0] == '2azQ1b91cZZ':
        continue
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
    
    # rgb_feats = rgb_encoder(rgb_imgs)        # Extract RGB features
    # depth_feats = depth_encoder(depth_imgs)  # Extract depth features
    
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
        cur_obstacle = obstacle[i]
        cur_source_pos = source_pos[i]
        print(f'Processing sample: scan={scan_id}, waypoint={waypoint_id}')
        
        node_height = raw_graph_data[scan_id]['nodes'][waypoint_id][1]
        source_pos_complete = np.array([cur_source_pos[0], node_height, cur_source_pos[1]])
        top_down_map = draw.get_top_down_map(sim, base_height=node_height)
        source_pos_index = habitat_maps.to_grid(cur_source_pos[1], cur_source_pos[0], top_down_map.shape[0:2], sim)
        draw.draw_source_new(top_down_map, cur_source_pos, 0.05, sim)

        current_heading = 0.0
        obstacle_radial = draw.get_obstacles_from_radial_map(cur_obstacle, 120, 12)
        last_pos_index = None
        for r, theta in obstacle_radial:
            pos = draw._rtheta_to_global_coordinates(r, theta, cur_source_pos, current_heading)
            pos_index = habitat_maps.to_grid(pos[1], pos[0], top_down_map.shape[0:2], sim)
            if last_pos_index != None:
                draw.drawline(top_down_map, [last_pos_index[1], last_pos_index[0]], [pos_index[1], pos_index[0]], thickness=1, style="filled", color=6 ,gap=1)
            #draw.draw_obstacle_new(top_down_map_2, pos, 0.05, sim)
            last_pos_index = pos_index

        ############## use depth map to creat point cloud ##############
        number = 12
        k = 0
        heading = k * 2 * math.pi / number
        theta = -(heading - np.pi) / 2 
        init_agent_position = np.array(raw_graph_data[cur_scan_id]['nodes'][waypoint_id]) # this is the agent position
        camera_height = 0.38 #1.25 # notice I do not think the robot is this tall, but this is the height of the camera
        init_camera_position = init_agent_position + np.array([0, camera_height, 0])
        init_agent_rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        init_camera_rotation = init_agent_rotation
        all_camera_positions, all_camera_rotations = [], []
        all_depths = []
        for k in range(number):
            heading = k * 2 * math.pi / number
            theta = -(heading - np.pi) / 2 
            rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
            obs = sim.get_observations_at(init_agent_position, rotation) # this function need agent position as input not camera position
            all_depths.append(obs['depth'])
            all_camera_positions.append(init_camera_position)
            all_camera_rotations.append(rotation)
        point_np_sum = []
        for v in range(number):
            camera_matrix_T = get_transform_matrices(all_camera_positions[v], all_camera_rotations[v], init_camera_position, init_camera_rotation)
            full_scene_pcd = build_full_scene_pcd(all_depths[v][:,:,0], hfov=90)
            full_scene_pcd.transform(camera_matrix_T)
            points_np = np.asarray(full_scene_pcd.points)
            point_np_sum.append(points_np)
        point_np_sum = np.concatenate(point_np_sum, axis=0)
        num_angles = 120
        num_distances = 12
        max_dist = 3
        angle_resolution = 2 * np.pi / num_angles
        dist_resolution = max_dist / num_distances
        max_unit_climb = 0.21
        robot_height = 0.38  # height of the robot
        slopes = np.zeros((num_angles, num_distances))
        robot_radius = 0.2
        xys = np.zeros((num_angles,  num_distances, 2))
        for angle_idx in range(num_angles):
            robot_floor = -robot_height
            for dist_idx in range(num_distances):
                angle = angle_idx * angle_resolution
                dist = (dist_idx + 1) * dist_resolution
                x = dist * np.cos(angle)
                y = dist * np.sin(angle)
                xys[angle_idx, dist_idx, 0] = x
                xys[angle_idx, dist_idx, 1] = y
                # find all points with in 0.4 m from (x,y) 
                distances = np.sqrt((point_np_sum[:, 0] - x) ** 2 + (point_np_sum[:, 2] - y) ** 2)
                close_points = point_np_sum[distances < robot_radius] # robot_radius this is based on the robot size
                if len(close_points) == 0:
                    # no points in this direction at this distance 
                    continue
                else:
                    # find the maximum allowed height at this distance
                    # carmera position is at (0, 0, 0) and the camera height is 0.38
                    z_max = robot_floor + robot_height + 0.5 #0.5
                    z_min = robot_floor #-robot_height
                    lower_points = close_points[(close_points[:, 1] <= z_max)]#&(close_points[:, 1] >= z_min)]
                    if len(lower_points) == 0:
                        # no points below the maximum height
                        continue
                    max_height = lower_points[:, 1].max()
                    slopes[angle_idx, dist_idx] = max_height - robot_floor
                robot_floor += slopes[angle_idx, dist_idx]
                            
                # ax.scatter(x, y, s=1.0, c='red', alpha=0.5)
        # find the first slope in each direction that is larger than the maximum allowed slope
        max_slope = 0.25 # # maximum allowed slope now it is 45 degree
        cur_obs_preds = np.zeros((num_angles, num_distances))
        for angle_idx in range(num_angles):
            for dist_idx in range(num_distances):
                if slopes[angle_idx, dist_idx] > max_slope:# or slopes[angle_idx, dist_idx] < -max_slope:
                    cur_obs_preds[angle_idx, dist_idx:] = 1
                    break


        current_heading = 0.0  
        obs_pred_radial = draw.get_obstacles_from_radial_map(cur_obs_preds, 120, 12)
        last_pos_index = None
        for r, theta in obs_pred_radial:
            pos = draw._rtheta_to_global_coordinates(r, theta, cur_source_pos, current_heading)
            pos_index = habitat_maps.to_grid(pos[1], pos[0], top_down_map.shape[0:2], sim)
            if last_pos_index != None:
                draw.drawline(top_down_map, [last_pos_index[1], last_pos_index[0]], [pos_index[1], pos_index[0]], thickness=1, style="filled", color=13 ,gap=1)
            #draw.draw_obstacle_new(top_down_map_2, pos, 0.05, sim)
            last_pos_index = pos_index

        color_map = draw.colorize_topdown_map(top_down_map)
        crop_image = draw.crop_around_point(color_map, source_pos_index[0], source_pos_index[1], padding=200)

        save_dir = f'waypoints_visualize_pc/{scan_id}/{waypoint_id}'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'obstacle.png')
        plt.imsave(save_path, crop_image)

        n_rows = 2
        n_cols = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6))
        panorama_d = np.concatenate(depth_imgs[i].cpu().numpy(), axis=1)
        panorama_d = np.fliplr(panorama_d)  # Flip depth panorama horizontally
        axes[0].imshow(panorama_d, cmap='gray')  # or 'gray', 'viridis', etc.
        panorama_rgb = np.concatenate(rgb_imgs[i].cpu().numpy().transpose(0,2,3,1), axis=1)
        panorama_rgb = np.fliplr(panorama_rgb)
        axes[1].imshow(panorama_rgb)
        save_path = os.path.join(save_dir, 'sensor.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close(fig)  
        
        



