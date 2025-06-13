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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup(args):
    """
    Set random seeds and create experiment directories
    Args:
        args: Command line arguments
    """
    torch.manual_seed(0)  # Set PyTorch random seed
    random.seed(0)  # Set Python random seed
    exp_log_path = './checkpoints/%s/'%(args.EXP_ID)  # Experiment log path
    os.makedirs(exp_log_path, exist_ok=True)  # Create experiment directory
    exp_log_path = './checkpoints/%s/snap/'%(args.EXP_ID)  # Model snapshot path
    os.makedirs(exp_log_path, exist_ok=True)  # Create model snapshot directory

def find_obstacle_indices(obstacle):
    """
    Find the first indices of obstacle pixels in the obstacle map
    Args:
        obstacle: Obstacle map tensor
    Returns:
        List of indices where obstacles are present
    """
    num_classes = obstacle.size(-1)
    non_zero_mask = obstacle != 0  # Create a mask for non-zero elements

    # Convert boolean mask to int 
    non_zero_int = non_zero_mask.int()

    # use argmax to find the index of the first non-zero value 
    first_non_zero_idx = non_zero_int.argmax(dim=-1)  # Get the index of the first non-zero value along the last dimension
    has_non_zero = non_zero_mask.any(dim=-1)  # Check if there are any non-zero values along the last dimension
    first_non_zero_idx[~has_non_zero] = num_classes  # If no non-zero values, set index to 0
    return first_non_zero_idx

class Param():
    """Parameters class, handles command line arguments"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train waypoint predictor')

        # Experiment settings
        self.parser.add_argument('--EXP_ID', type=str, default='test_0')  # Experiment ID
        self.parser.add_argument('--TRAINEVAL', type=str, default='train', help='trian or eval mode')  # Training or evaluation mode
        self.parser.add_argument('--VIS', type=int, default=0, help='visualize predicted hearmaps')  # Whether to visualize predicted heatmaps
        # self.parser.add_argument('--LOAD_EPOCH', type=int, default=None, help='specific an epoch to load for eval')

        # Model structure parameters
        self.parser.add_argument('--ANGLES', type=int, default=24)  # Number of angle divisions
        self.parser.add_argument('--NUM_IMGS', type=int, default=24)  # Number of images
        self.parser.add_argument('--NUM_CLASSES', type=int, default=12)  # Number of classes
        self.parser.add_argument('--MAX_NUM_CANDIDATES', type=int, default=5)  # Maximum number of candidate points

        self.parser.add_argument('--PREDICTOR_NET', type=str, default='TRM', help='TRM only')  # Predictor network type

        # Training parameters
        self.parser.add_argument('--EPOCH', type=int, default=10)  # Number of training epochs
        self.parser.add_argument('--BATCH_SIZE', type=int, default=2)  # Batch size
        self.parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)  # Learning rate
        self.parser.add_argument('--WEIGHT', type=int, default=0, help='weight the target map')  # Whether to weight the target map

        # Transformer model parameters
        self.parser.add_argument('--TRM_LAYER', default=2, type=int, help='number of TRM hidden layers')  # Number of TRM hidden layers
        self.parser.add_argument('--TRM_NEIGHBOR', default=2, type=int, help='number of attention mask neighbor')  # Number of attention mask neighbors
        self.parser.add_argument('--HEATMAP_OFFSET', default=2, type=int, help='an offset determined by image FoV and number of images')  # Heatmap offset
        self.parser.add_argument('--HIDDEN_DIM', default=768, type=int)  # Hidden dimension size

        self.args = self.parser.parse_args()

def predict_waypoints(args):
    """
    Main function for waypoint prediction
    Args:
        args: Command line arguments
    """
    print('\nArguments', args)
    log_dir = './checkpoints/%s/tensorboard/'%(args.EXP_ID)  # TensorBoard log directory
    writer = SummaryWriter(log_dir=log_dir)  # Create TensorBoard writer

    ''' Initialize network models '''
    # Initialize RGB encoder, using pretrained weights, without fine-tuning
    rgb_encoder = RGBEncoder(resnet_pretrain=True, trainable=False).to(device)
    # Initialize depth encoder, using pretrained weights, without fine-tuning
    depth_encoder = DepthEncoder(resnet_pretrain=True, trainable=False).to(device)
    if args.PREDICTOR_NET == 'TRM':
        print('\nUsing TRM predictor')
        print('HIDDEN_DIM default to 768')
        args.HIDDEN_DIM = 768
        # Initialize Transformer-based binary distribution predictor
        predictor = BinaryDistPredictor_TRM(args=args,
            hidden_dim=args.HIDDEN_DIM, n_classes=args.NUM_CLASSES).to(device)

    ''' Load navigability data (ground truth waypoints, obstacles, and weights) '''
    nav_dict_path = './training_data/%s_*_mp3d_waypoint_twm0.2_obstacle_first_withpos.json'%(args.ANGLES)

    #navigability_dict = utils.load_gt_navigability(
    #    './training_data/%s_*_mp3d_waypoint_twm0.2_obstacle_first_withpos.json'%(args.ANGLES))
    navigability_dict = utils.load_gt_navigability(nav_dict_path)
    #print("\nNavigability dictionary keys:")
    #for key in navigability_dict.keys():
        #print(key)

    ''' Create data loaders for RGB and depth images '''
    train_img_dir = './training_data/rgbd_fov90/train/*/*.pkl'  # Training image directory
    traindataloader = RGBDepthPano(args.NUM_IMGS, train_img_dir, navigability_dict)  # Training data loader
    eval_img_dir = './training_data/rgbd_fov90/val_unseen/*/*.pkl'  # Evaluation image directory
    evaldataloader = RGBDepthPano(args.NUM_IMGS, eval_img_dir, navigability_dict)  # Evaluation data loader
    if args.TRAINEVAL == 'train':
        trainloader = torch.utils.data.DataLoader(traindataloader, 
        batch_size=args.BATCH_SIZE, shuffle=True, num_workers=4)  # Training batch data loader
    evalloader = torch.utils.data.DataLoader(evaldataloader, 
        batch_size=args.BATCH_SIZE, shuffle=False, num_workers=4)  # Evaluation batch data loader

    ''' Define loss functions and optimizer '''
    criterion_bcel = torch.nn.BCEWithLogitsLoss(reduction='none')  # Binary cross entropy loss
    criterion_mse = torch.nn.MSELoss(reduction='none')  # Mean squared error loss
    criterion_cls = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')  # Cross entropy loss

    params = list(predictor.parameters())  # Get predictor parameters
    optimizer = torch.optim.AdamW(params, lr=args.LEARNING_RATE)  # Use AdamW optimizer

    ''' Training loop '''
    if args.TRAINEVAL == 'train':
        print('\nTraining starts')
        # Record best validation results
        best_val_1 = {"avg_wayscore": 0.0, "log_string": '', "update":False}  # Best result based on average waypoint score
        best_val_2 = {"avg_pred_distance": 10.0, "log_string": '', "update":False}  # Best result based on average prediction distance

        for epoch in range(args.EPOCH):  # Loop through each training epoch
            sum_loss = 0.0  # Accumulated loss

            # Set encoders to evaluation mode (parameters not updated)
            rgb_encoder.eval()
            depth_encoder.eval()
            # Set predictor to training mode
            predictor.train()

            # Iterate through training data
            for i, data in enumerate(trainloader):
                scan_ids = data['scan_id']  # Scene IDs
                waypoint_ids = data['waypoint_id']  # Waypoint IDs
                rgb_imgs = data['rgb'].to(device)  # RGB images
                depth_imgs = data['depth'].to(device)  # Depth images

                ''' Check image orientation (commented code) '''
                # from PIL import Image
                # from matplotlib import pyplot
                # import numpy as np
                # # import pdb; pdb.set_trace()
                # out_img = np.swapaxes(
                #     np.swapaxes(
                #         data['no_trans_rgb'][0].cpu().numpy(), 1,2),
                #     2, 3)
                # for kk, out_img_i in enumerate(out_img):
                #     im = Image.fromarray(out_img_i)
                #     im.save("./play/%s.png"%(kk))
                #     pyplot.imsave("./play/mpl_%s.png"%(kk), out_img_i)
                # out_depth = data['no_trans_depth'][0].cpu().numpy() * 255
                # out_depth = out_depth.astype(np.uint8)
                # for kk, out_depth_i in enumerate(out_depth):
                #     im = Image.fromarray(out_depth_i)
                #     im.save("./play/depth_%s.png"%(kk))

                ''' Process observation data '''
                rgb_feats = rgb_encoder(rgb_imgs)        # Extract RGB features (BATCH_SIZE*ANGLES, 2048)
                depth_feats = depth_encoder(depth_imgs)  # Extract depth features (BATCH_SIZE*ANGLES, 128, 4, 4)

                ''' Get learning targets '''
                # Get ground truth navigation maps (target, obstacle, weight)
                target, obstacle, weight, _, _ = utils.get_gt_nav_map(
                    args.ANGLES, navigability_dict, scan_ids, waypoint_ids)
                target = target.to(device)
                obstacle = obstacle.to(device) # [batch_size, num_angles, num_distances]
                weight = weight.to(device)
                obstacle_indices = find_obstacle_indices(obstacle)  # Find obstacle indices
                # Use TRM predictor for prediction
                if args.PREDICTOR_NET == 'TRM':
                    vis_logits, obs_logits = TRM_predict('train', args,
                        predictor, rgb_feats, depth_feats)

                    # Calculate loss
                    loss_vis = criterion_mse(vis_logits, target)
                    if args.WEIGHT:
                        loss_vis = loss_vis * weight  # Apply weights if enabled
                    num_classes = obs_logits.size(-1)  # Number of classes in obstacle logits
                    loss_obs = criterion_cls(obs_logits.reshape(-1, num_classes), obstacle_indices.reshape(-1))  # Obstacle loss
                    total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES  # Calculate total loss
                    total_loss += 10 * loss_obs.sum() / obs_logits.size(0) / args.ANGLES  # Add obstacle loss to total loss

                # Backpropagation and optimization
                optimizer.zero_grad()  # Clear gradients
                total_loss.backward()  # Backpropagation
                optimizer.step()  # Update parameters
                sum_loss += total_loss.item()  # Accumulate loss

                # Print training progress
                print_progress(i+1, len(trainloader), prefix='Epoch: %d/%d'%((epoch+1),args.EPOCH))
            
            # Record training loss to TensorBoard
            writer.add_scalar("Train/Loss", sum_loss/(i+1), epoch)
            print('Train Loss: %.5f' % (sum_loss/(i+1)))  # Print average training loss

            ''' Evaluation phase '''
            # print('Evaluation ...')
            sum_loss = 0.0  # Evaluation loss
            # Store prediction results
            predictions = {'sample_id': [], 
                'source_pos': [], 'target_pos': [],
                'probs': [], 'logits': [],
                'target': [], 'obstacle': [], 'sample_loss': []}

            # Set all networks to evaluation mode
            rgb_encoder.eval()
            depth_encoder.eval()
            predictor.eval()

            # Iterate through evaluation data
            for i, data in enumerate(evalloader):
                scan_ids = data['scan_id']
                waypoint_ids = data['waypoint_id']
                sample_id = data['sample_id']
                rgb_imgs = data['rgb'].to(device)
                depth_imgs = data['depth'].to(device)

                # Get ground truth navigation maps
                target, obstacle, weight, \
                source_pos, target_pos = utils.get_gt_nav_map(
                    args.ANGLES, navigability_dict, scan_ids, waypoint_ids)
                target = target.to(device)
                obstacle = obstacle.to(device)
                obstacle_indices = find_obstacle_indices(obstacle)  # Find obstacle indices
                weight = weight.to(device)

                ''' Process observation data '''
                rgb_feats = rgb_encoder(rgb_imgs)        # Extract RGB features
                depth_feats = depth_encoder(depth_imgs)  # Extract depth features

                # Use TRM predictor for prediction
                if args.PREDICTOR_NET == 'TRM':
                    vis_probs, vis_logits, obs_probs, obs_logits = TRM_predict('eval', args,
                        predictor, rgb_feats, depth_feats)
                    overall_probs = vis_probs  # Overall probabilities
                    overall_logits = vis_logits  # Overall logits
                    
                    # Calculate loss
                    loss_vis = criterion_mse(vis_logits, target)
                    loss_obs = criterion_cls(obs_logits.reshape(-1, obs_logits.size(-1)), 
                        obstacle_indices.reshape(-1))
                    if args.WEIGHT:
                        loss_vis = loss_vis * weight  # Apply weights if enabled
                    sample_loss = loss_vis.sum(-1).sum(-1) / args.ANGLES  # Sample loss
                    total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES  # Total loss
                    total_loss += 10 * loss_obs.sum() / obs_logits.size(0) / args.ANGLES  # Add obstacle loss to total loss

                # Accumulate loss and store prediction results
                sum_loss += total_loss.item()
                predictions['sample_id'].append(sample_id)
                predictions['source_pos'].append(source_pos)
                predictions['target_pos'].append(target_pos)
                predictions['probs'].append(overall_probs.tolist())
                predictions['logits'].append((overall_logits.tolist()))
                predictions['target'].append(target.tolist())
                predictions['obstacle'].append(obstacle.tolist())
                predictions['sample_loss'].append(target.tolist())

            # Print evaluation loss
            print('Eval Loss: %.5f' % (sum_loss/(i+1)))
            # Evaluate prediction results
            results = waypoint_eval(args, predictions)
            
            # Record evaluation metrics to TensorBoard
            writer.add_scalar("Evaluation/Loss", sum_loss/(i+1), epoch)
            writer.add_scalar("Evaluation/p_waypoint_openspace", results['p_waypoint_openspace'], epoch)
            writer.add_scalar("Evaluation/p_waypoint_obstacle", results['p_waypoint_obstacle'], epoch)
            writer.add_scalar("Evaluation/avg_wayscore", results['avg_wayscore'], epoch)
            writer.add_scalar("Evaluation/avg_pred_distance", results['avg_pred_distance'], epoch)
            
            # Build log string
            log_string = 'Epoch %s '%(epoch)
            for key, value in results.items():
                if key != 'candidates': 
                    log_string += '{} {:.5f} | '.format(str(key), value)
            print(log_string)  # Print evaluation results

            # Save checkpoint - based on average waypoint score
            if results['avg_wayscore'] > best_val_1['avg_wayscore']:
                checkpoint_save_path = './checkpoints/%s/snap/check_val_best_avg_wayscore'%(args.EXP_ID)
                utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_save_path)
                print('New best avg_wayscore result found, checkpoint saved to %s'%(checkpoint_save_path))
                best_val_1['avg_wayscore'] = results['avg_wayscore']
                best_val_1['log_string'] = log_string
            
            # Save latest checkpoint
            checkpoint_reg_save_path = './checkpoints/%s/snap/check_latest'%(args.EXP_ID)
            utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_reg_save_path)
            print('Best avg_wayscore result til now: ', best_val_1['log_string'])

            # Save checkpoint - based on average prediction distance
            if results['avg_pred_distance'] < best_val_2['avg_pred_distance']:
                checkpoint_save_path = './checkpoints/%s/snap/check_val_best_avg_pred_distance'%(args.EXP_ID)
                utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_save_path)
                print('New best avg_pred_distance result found, checkpoint saved to %s'%(checkpoint_save_path))
                best_val_2['avg_pred_distance'] = results['avg_pred_distance']
                best_val_2['log_string'] = log_string
            
            # Save latest checkpoint again (redundant operation)
            checkpoint_reg_save_path = './checkpoints/%s/snap/check_latest'%(args.EXP_ID)
            utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_reg_save_path)
            print('Best avg_pred_distance result til now: ', best_val_2['log_string'])

    elif args.TRAINEVAL == 'eval':
        ''' Evaluation mode - inference (with a bit of expert mixing) '''
        print('\nEvaluation mode, please doublecheck EXP_ID and LOAD_EPOCH')
        # Load best checkpoint
        checkpoint_load_path = './checkpoints/%s/snap/check_cwp_bestdist_hfov90'%(args.EXP_ID)
        epoch, predictor, optimizer = utils.load_checkpoint(
                        predictor, optimizer, checkpoint_load_path)

        sum_loss = 0.0
        # Store prediction results
        predictions = {'sample_id': [], 
            'source_pos': [], 'target_pos': [],
            'probs': [], 'logits': [],
            'target': [], 'obstacle': [], 'sample_loss': []}

        # Set all networks to evaluation mode
        rgb_encoder.eval()
        depth_encoder.eval()
        predictor.eval()

        # Iterate through evaluation data
        for i, data in enumerate(evalloader):
            # If visualization is enabled and 5 samples have been processed, break the loop
            if args.VIS and i == 5:
                break

            scan_ids = data['scan_id']
            waypoint_ids = data['waypoint_id']
            sample_id = data['sample_id']
            rgb_imgs = data['rgb'].to(device)
            depth_imgs = data['depth'].to(device)

            # Get ground truth navigation maps
            target, obstacle, weight, \
            source_pos, target_pos = utils.get_gt_nav_map(
                args.ANGLES, navigability_dict, scan_ids, waypoint_ids)
            target = target.to(device)
            obstacle = obstacle.to(device)
            weight = weight.to(device)

            ''' Process observation data '''
            rgb_feats = rgb_encoder(rgb_imgs)        # Extract RGB features
            depth_feats = depth_encoder(depth_imgs)  # Extract depth features

            ''' Predict waypoint probabilities '''
            if args.PREDICTOR_NET == 'TRM':
                vis_probs, vis_logits = TRM_predict('eval', args,
                    predictor, rgb_feats, depth_feats)
                overall_probs = vis_probs
                overall_logits = vis_logits
                loss_vis = criterion_mse(vis_logits, target)

                if args.WEIGHT:
                    loss_vis = loss_vis * weight
                sample_loss = loss_vis.sum(-1).sum(-1) / args.ANGLES
                total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES

            # Accumulate loss and store prediction results
            sum_loss += total_loss.item()
            predictions['sample_id'].append(sample_id)
            predictions['source_pos'].append(source_pos)
            predictions['target_pos'].append(target_pos)
            predictions['probs'].append(overall_probs.tolist())
            predictions['logits'].append(overall_logits.tolist())
            predictions['target'].append(target.tolist())
            predictions['obstacle'].append(obstacle.tolist())
            predictions['sample_loss'].append(target.tolist())

        # Print evaluation loss
        print('Eval Loss: %.5f' % (sum_loss/(i+1)))
        # Evaluate prediction results
        results = waypoint_eval(args, predictions)
        
        # Build log string
        log_string = 'Epoch %s '%(epoch)
        for key, value in results.items():
            if key != 'candidates':
                log_string += '{} {:.5f} | '.format(str(key), value)
        print(log_string)
        print('Evaluation Done')

    else:
        RunningModeError  # Running mode error

if __name__ == "__main__":
    param = Param()  # Create parameter object
    args = param.args  # Get command line arguments
    setup(args)  # Setup experiment environment

    # If visualization is enabled, ensure it's in evaluation mode
    if args.VIS:
        assert args.TRAINEVAL == 'eval'

    predict_waypoints(args)  # Execute waypoint prediction
