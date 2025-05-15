import torch
import numpy as np
import sys
import glob
import json

def neighborhoods(mu, x_range, y_range, sigma, circular_x=True, gaussian=False):
    """ 
    Generate a mask centered at mu, given x and y ranges, with origin at the center of the output
    Args:
        mu: Tensor (N, 2), center point coordinates
        x_range: x-axis range
        y_range: y-axis range
        sigma: size of the mask or standard deviation of Gaussian distribution
        circular_x: whether to connect circularly in x direction (like panoramic images)
        gaussian: whether to use Gaussian distribution instead of binary mask
    Returns:
        Tensor (N, y_range, x_range), generated mask
    """
    # Extract x and y coordinates of center points and expand dimensions for broadcasting
    x_mu = mu[:,0].unsqueeze(1).unsqueeze(1)
    y_mu = mu[:,1].unsqueeze(1).unsqueeze(1)

    # Generate coordinate grid centered at mu
    x = torch.arange(start=0,end=x_range, device=mu.device, dtype=mu.dtype).unsqueeze(0).unsqueeze(0)
    y = torch.arange(start=0,end=y_range, device=mu.device, dtype=mu.dtype).unsqueeze(1).unsqueeze(0)

    # Calculate distance from each point to center
    y_diff = y - y_mu
    x_diff = x - x_mu
    if circular_x:
        # If x-axis is circular (like panoramic), take minimum distance from both sides
        x_diff = torch.min(torch.abs(x_diff), torch.abs(x_diff + x_range))
    
    # Closer to center means closer to 1
    if gaussian:
        # Generate Gaussian distribution mask
        output = torch.exp(-0.5 * ((x_diff/sigma[0])**2 + (y_diff/sigma[1])**2 ))
    else:
        # Generate binary mask (rectangular region)
        output = torch.logical_and(
            torch.abs(x_diff) <= sigma[0], torch.abs(y_diff) <= sigma[1]
        ).type(mu.dtype)

    return output


def nms(pred, max_predictions=10, sigma=(1.0,1.0), gaussian=False):
    ''' 
    Non-maximum suppression function to find local maxima in prediction maps
    Args:
        pred: Input prediction map, shape (batch_size, 1, height, width)
        max_predictions: Maximum number of predictions to keep per sample
        sigma: Size of suppression region
        gaussian: Whether to use Gaussian suppression instead of rectangular suppression
    Returns:
        Prediction map after non-maximum suppression
    '''

    shape = pred.shape

    # Initialize output and prediction copy for suppression
    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0],-1))  # (BATCH_SIZE, height*width)
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0],-1))  # (BATCH_SIZE, height*width)

    # Iteratively find maximum prediction points
    for i in range(max_predictions):
        # Find global maximum in current prediction map
        flat_supp_pred = supp_pred.reshape((shape[0],-1))
        val, ix = torch.max(flat_supp_pred, dim=1)
        indices = torch.arange(0,shape[0])
        # Save maximum value to output map
        flat_output[indices,ix] = flat_pred[indices,ix]

        # Calculate suppression region
        # Convert 1D index to 2D coordinates
        y = ix / shape[-1]
        x = ix % shape[-1]
        mu = torch.stack([x,y], dim=1).float() # Stack x,y coordinates into tensor of shape (batch_size, 2)

        # Generate suppression mask centered at maximum
        g = neighborhoods(mu, shape[-1], shape[-2], sigma, gaussian=gaussian)

        # Suppress regions near center maximum
        supp_pred *= (1-g.unsqueeze(1))

    # Ensure no negative values
    output[output < 0] = 0 
    return output 


def get_gt_nav_map(num_angles, nav_dict, scan_ids, waypoint_ids):
    ''' 
    Get ground truth navigation maps, including target map, obstacle map, and weight map
    Args:
        num_angles: Number of angles (usually number of image divisions)
        nav_dict: Dictionary containing navigation information
        scan_ids: List of scan IDs
        waypoint_ids: List of waypoint IDs
    Returns:
        target: Target map, 1 indicates ground truth keypoints, 2 indicates ignored indices
        obstacle: Obstacle map
        weight: Weight map, 0 indicates ignore, 1 indicates waypoint/far from waypoint/obstacle, (0,1) indicates other open space
        source_pos: List of source positions
        target_pos: List of target positions
    '''
    bs = len(scan_ids)  # Batch size
    # Initialize tensors
    target = torch.zeros(bs, num_angles, 12)
    obstacle = torch.zeros(bs, num_angles, 12)
    weight = torch.zeros(bs, num_angles, 12)
    source_pos = []
    target_pos = []

    # Fill data for each sample
    for i in range(bs):
        target[i] = torch.tensor(nav_dict[scan_ids[i]][waypoint_ids[i]]['target'])
        obstacle[i] = torch.tensor(nav_dict[scan_ids[i]][waypoint_ids[i]]['obstacle'])
        weight[i] = torch.tensor(nav_dict[scan_ids[i]][waypoint_ids[i]]['weight'])
        source_pos.append(nav_dict[scan_ids[i]][waypoint_ids[i]]['source_pos'])
        target_pos.append(nav_dict[scan_ids[i]][waypoint_ids[i]]['target_pos'])

    return target, obstacle, weight, source_pos, target_pos


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=10):
    """
    Call in a loop to create terminal progress bar
    Args:
        iteration   - Required  : Current iteration (Int)
        total       - Required  : Total iterations (Int)
        prefix      - Optional  : Prefix string (Str)
        suffix      - Optional  : Suffix string (Str)
        decimals    - Optional  : Positive number of decimals in percent complete (Int)
        bar_length  - Optional  : Character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def save_checkpoint(epoch, net, net_optimizer, path):
    ''' 
    Save model checkpoint
    Args:
        epoch: Current training epoch
        net: Network model
        net_optimizer: Network optimizer
        path: Save path
    '''
    states = {}
    def create_state(name, model, optimizer):
        states[name] = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    all_tuple = [("predictor", net, net_optimizer)]
    for param in all_tuple:
        create_state(*param)
    torch.save(states, path)


def load_checkpoint(net, net_optimizer, path):
    ''' 
    Load model parameters (but not training state)
    Args:
        net: Network model
        net_optimizer: Network optimizer
        path: Checkpoint path
    Returns:
        epoch: Loaded epoch
        net: Network model with loaded parameters
        net_optimizer: Network optimizer with loaded parameters
    '''
    states = torch.load(path)
    def recover_state(name, model, optimizer):
        state = model.state_dict()
        model_keys = set(state.keys())
        load_keys = set(states[name]['state_dict'].keys())
        if model_keys != load_keys:
            print("NOTICE: DIFFERENT KEYS FOUND")
        state.update(states[name]['state_dict'])
        model.load_state_dict(state)
        optimizer.load_state_dict(states[name]['optimizer'])
    all_tuple = [("predictor", net, net_optimizer)]
    for param in all_tuple:
        recover_state(*param)
    return states['predictor']['epoch'], all_tuple[0][1], all_tuple[0][2]


def get_attention_mask(num_imgs=24, neighbor=2):
    """
    Generate Transformer attention mask to control each position can only attend to itself and neighboring positions
    Args:
        num_imgs: Number of images, default 24 (usually corresponding to 360-degree panoramic division)
        neighbor: Number of neighbors allowed to attend to on each side, default 2
    Returns:
        Attention mask tensor of shape [1,1,num_imgs,num_imgs], 1 indicates allowed attention, 0 indicates forbidden attention
    """
    assert neighbor <= 5  # Ensure number of neighbors doesn't exceed 5

    # Initialize all-zero mask matrix
    mask = np.zeros((num_imgs,num_imgs))
    
    # Create template row representing attention pattern for a single position
    t = np.zeros(num_imgs)
    t[:neighbor+1] = np.ones(neighbor+1)  # Set self and right neighbor positions to 1
    if neighbor != 0:
        t[-neighbor:] = np.ones(neighbor)  # Set left neighbor positions to 1
    
    # Fill each row of mask matrix in a loop
    for ri in range(num_imgs):
        mask[ri] = t  # Fill current template into row ri
        t = np.roll(t, 1)  # Circular right shift template for next row

    # Return tensor reshaped to Transformer attention mask format
    return torch.from_numpy(mask).reshape(1,1,num_imgs,num_imgs).long()


def load_gt_navigability(path):
    ''' 
    Load ground truth waypoint navigability data
    Args:
        path: Data file path prefix
    Returns:
        all_scans_nav_map: Dictionary containing all scan navigation maps
    '''
    all_scans_nav_map = {}
    gt_dir = glob.glob('%s*'%(path))
    for gt_dir_i in gt_dir:
        with open(gt_dir_i, 'r') as f:
            nav_map = json.load(f)
        for scan_id, values in nav_map.items():
            all_scans_nav_map[scan_id] = values
    return all_scans_nav_map
