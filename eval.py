import torch
import numpy as np
import math
import utils
import copy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage.transform import resize
import os

def waypoint_eval(args, predictions):
    ''' Evaluation of the predicted waypoint map,
        notice that the number of candidates is cap at args.MAX_NUM_CANDIDATES,
        but the number of GT waypoints could be any value in range [1,args.ANGLES].

        The preprocessed data is constraining each angle sector has at most
        one GT waypoint.
    '''

    # Extract data from predictions dictionary
    sample_id = predictions['sample_id']         # Unique identifier for each sample
    source_pos = predictions['source_pos']       # Starting position coordinates
    target_pos = predictions['target_pos']       # Ground truth waypoint positions
    probs = predictions['probs']                 # Probability scores
    logits = predictions['logits']               # Raw model outputs
    target = predictions['target']               # Ground truth waypoint map
    obstacle = predictions['obstacle']           # Obstacle map (1 for obstacle, 0 for open space)
    sample_loss = predictions['sample_loss']     # Loss for each sample

    # Initialize results dictionary with evaluation metrics
    results = {
        'candidates': {},                # Dictionary to store predicted waypoints for each sample
        'p_waypoint_openspace': 0.0,     # Percentage of waypoints in open space
        'p_waypoint_obstacle': 0.0,      # Percentage of waypoints in obstacles
        'avg_wayscore': 0.0,             # Average score of waypoints on target map
        'avg_pred_distance': 0.0,        # Average distance from targets to predictions
        'avg_chamfer_distance': 0.0,     # Average Chamfer distance between predictions and targets
        'avg_hausdorff_distance': 0.0,   # Average Hausdorff distance between predictions and targets
        'avg_num_delta': 0.0,            # Average difference between number of predictions and targets
    }

    # Lists to store metrics for each sample
    num_candidate = []          # Number of candidates (capped at args.MAX_NUM_CANDIDATES)
    num_waypoint_openspace = [] # Count of waypoints in open space
    num_waypoint_obstacle = []  # Count of waypoints in obstacles
    waypoint_score = []         # Scores on target map collected by predictions
    pred_distance = []          # Distance from targets to predictions
    chamfer_distance_all = []   # Chamfer distances for all samples
    hausdorff_distance_all = [] # Hausdorff distances for all samples
    num_delta_all = []          # Difference in number of predictions vs targets for all samples

    ''' Process each prediction batch '''
    for i, batch_x in enumerate(logits):
        # Extract batch data
        batch_sample_id = sample_id[i]
        batch_source_pos = source_pos[i]
        batch_target_pos = target_pos[i]
        batch_target = target[i]
        batch_obstacle = obstacle[i]
        batch_sample_loss = sample_loss[i]

        # Convert to tensor and apply softmax to get normalized probabilities
        batch_x = torch.tensor(batch_x)
        batch_x_norm = torch.softmax(
            batch_x.reshape(
                batch_x.size(0), args.ANGLES*args.NUM_CLASSES
                ), dim=1
            )
        batch_x_norm = batch_x_norm.reshape(batch_x.size(0), args.ANGLES, args.NUM_CLASSES)
        # batch_x_norm = torch.sigmoid(batch_x)

        # Wrap around the first and last columns to handle circular data
        batch_x_norm_wrap = torch.cat(
            (batch_x_norm[:,-1:,:], batch_x_norm, batch_x_norm[:,:1,:]), 
            dim=1)
        
        # Apply non-maximum suppression to find peak waypoint candidates
        # sigma controls the suppression window size
        batch_output_map = utils.nms(
            batch_x_norm_wrap.unsqueeze(1), max_predictions=5,
            sigma=(7.0,5.0))
        batch_output_map = batch_output_map.squeeze()[:,1:-1,:]

        # Additional NMS with different sigma values for visualization purposes
        if args.VIS:
            # NMS with different sigma values for comparison
            batch_output_map_sig4 = utils.nms(
                batch_x_norm_wrap.unsqueeze(1), max_predictions=args.MAX_NUM_CANDIDATES,
                sigma=(4.0,4.0))
            batch_output_map_sig4 = batch_output_map_sig4.squeeze()[:,1:-1,:]
            batch_output_map_sig5 = utils.nms(
                batch_x_norm_wrap.unsqueeze(1), max_predictions=args.MAX_NUM_CANDIDATES,
                sigma=(5.0,5.0))
            batch_output_map_sig5 = batch_output_map_sig5.squeeze()[:,1:-1,:]
            batch_output_map_sig7_5 = utils.nms(
                batch_x_norm_wrap.unsqueeze(1), max_predictions=args.MAX_NUM_CANDIDATES,
                sigma=(7.0,5.0))
            batch_output_map_sig7_5 = batch_output_map_sig7_5.squeeze()[:,1:-1,:]

        # Process each sample in the batch
        for j, id in enumerate(batch_sample_id):
            # Initialize data structures for this sample
            candidates = {}         # Dictionary to store angle-distance pairs
            c_openspace = 0         # Counter for waypoints in open space
            c_obstacle = 0          # Counter for waypoints in obstacles
            candidates_pos = []     # List to store polar coordinates of candidates

            ''' Gather predicted candidates and check if candidates are in openspace '''
            for jdx, angle_view in enumerate(batch_output_map[j]):
                # If there's a non-zero prediction at this angle
                if angle_view.sum() != 0:
                    # Store the distance index with highest probability
                    candidates[jdx] = angle_view.argmax().item()
                    
                    # Convert to polar coordinates [angle, distance]
                    candidates_pos.append(
                        [jdx * 2 * math.pi / args.ANGLES,  # Convert index to angle in radians
                        (candidates[jdx]+1) * 0.25])      # Convert index to distance in meters
                    
                    # Check if waypoint is in open space or obstacle
                    if batch_obstacle[j][jdx][candidates[jdx]] == 0:
                        c_openspace += 1
                    else:
                        c_obstacle += 1

            # Store the candidates for this sample
            results['candidates'][id] = {
                # 'loss': batch_sample_loss[j],
                'angle_dist': candidates,
            }
            
            # Update metrics
            num_candidate.append(len(candidates))
            num_waypoint_openspace.append(c_openspace)
            num_waypoint_obstacle.append(c_obstacle)

            ''' Calculate score collected over the target heatmap by predictions '''
            score_map = torch.tensor(batch_target[j])
            # Calculate average score of predictions on the target map
            score = (score_map[batch_output_map[j] != 0]
                ).sum() / (len(candidates))
            waypoint_score.append(score.item())

            ''' Measure target to prediction distance metrics '''
            # Convert positions to Cartesian coordinates
            bsp = np.array(batch_source_pos[j])  # Source position
            btp = np.array(batch_target_pos[j])  # Target positions
            cp = np.array(candidates_pos)        # Candidate positions (polar)
            
            # Convert polar to Cartesian coordinates
            cp_x = np.sin(cp[:,0]) * cp[:,1] + bsp[0]  # x = sin(angle) * distance + source_x
            cp_y = np.cos(cp[:,0]) * cp[:,1] + bsp[1]  # y = cos(angle) * distance + source_y
            cp = np.concatenate(
                (np.expand_dims(cp_x, axis=1),
                np.expand_dims(cp_y, axis=1)), axis=1)
            
            # Calculate distance matrix between targets and predictions
            tp_dists = cdist(btp, cp)  # Pairwise distances between all targets and predictions
            
            # Calculate the mean of minimum distances from targets to predictions
            tp_dist_min = tp_dists.min(1).mean()
            pred_distance.append(tp_dist_min)

            # Calculate Chamfer distance (bidirectional mean of minimum distances)
            predict_to_gt_0 = tp_dists.min(0).mean()  # Mean of minimum distances from predictions to targets
            gt_to_predict_0 = tp_dists.min(1).mean()  # Mean of minimum distances from targets to predictions
            chamfer_distance = 0.5 * (
                predict_to_gt_0 + gt_to_predict_0)    # Average of both directions
            chamfer_distance_all.append(chamfer_distance)

            # Calculate Hausdorff distance (maximum of minimum distances)
            predict_to_gt_1 = tp_dists.min(0).max()   # Max of minimum distances from predictions to targets
            gt_to_predict_1 = tp_dists.min(1).max()   # Max of minimum distances from targets to predictions
            hausdorff_distance = max(
                predict_to_gt_1, gt_to_predict_1)     # Maximum of both directions
            hausdorff_distance_all.append(hausdorff_distance)

            # Calculate difference between number of predictions and targets
            num_target = len(batch_target_pos[j])
            num_predict = len(candidates_pos)
            num_delta = num_predict - num_target
            num_delta_all.append(num_delta)

            # Visualization code
            if args.VIS:
                import pdb; pdb.set_trace()
                save_img_dir = './visualize/%s-best_avg_wayscore'%(args.EXP_ID.split('-')[1])
                if not os.path.exists(save_img_dir):
                    os.makedirs(save_img_dir)
                
                # Prepare visualization images
                im1 = (np.array(batch_target[j])/np.array(batch_target[j]).max()*255).astype('uint8')
                batch_x_pos = copy.deepcopy(batch_x[j].numpy())
                batch_x_pos[batch_x_pos<0]=0.0
                im2 = (batch_x_pos/batch_x_pos.max()*255).astype('uint8')
                im6 = (batch_output_map_sig4[j].numpy()/batch_output_map_sig4[j].numpy().max()*255).astype('uint8')
                im7 = (batch_output_map_sig5[j].numpy()/batch_output_map_sig5[j].numpy().max()*255).astype('uint8')
                im8 = (batch_output_map_sig7_5[j].numpy()/batch_output_map_sig7_5[j].numpy().max()*255).astype('uint8')
                
                # Create figure with multiple subplots
                fig = plt.figure(figsize=(10,14))
                fig.add_subplot(1, 5, 1); plt.imshow(im6); plt.axis('off')
                fig.add_subplot(1, 5, 2); plt.imshow(im7); plt.axis('off')
                fig.add_subplot(1, 5, 3); plt.imshow(im8); plt.axis('off')
                fig.add_subplot(1, 5, 4); plt.imshow(im2); plt.axis('off')
                fig.add_subplot(1, 5, 5); plt.imshow(im1); plt.axis('off')
                plt.savefig(save_img_dir+'/predict-target-%s-%s.jpeg'%(i,j),
                    bbox_inches='tight')
                plt.close()

    # Calculate overall metrics from collected sample-level metrics
    p_waypoint_openspace = sum(num_waypoint_openspace) / sum(num_candidate)  # Percentage in open space
    p_waypoint_obstacle = sum(num_waypoint_obstacle) / sum(num_candidate)    # Percentage in obstacles
    avg_wayscore = np.mean(waypoint_score).item()                           # Average waypoint score
    avg_pred_distance = np.mean(pred_distance).item()                       # Average prediction distance
    avg_chamfer_distance = np.mean(chamfer_distance_all).item()             # Average Chamfer distance
    avg_hausdorff_distance = np.mean(hausdorff_distance_all).item()         # Average Hausdorff distance
    avg_num_delta = np.mean(num_delta_all).item()                           # Average number difference

    # Populate the results dictionary with final metrics
    results['p_waypoint_openspace'] = p_waypoint_openspace
    results['p_waypoint_obstacle'] = p_waypoint_obstacle
    results['avg_wayscore'] = avg_wayscore
    results['avg_pred_distance'] = avg_pred_distance
    results['avg_chamfer_distance'] = avg_chamfer_distance
    results['avg_hausdorff_distance'] = avg_hausdorff_distance
    results['avg_num_delta'] = avg_num_delta

    return results
