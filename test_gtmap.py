import numpy as np
import habitat_sim
from habitat.sims import make_sim
from gen_training_data.utils import Simulator, horizontal_distance, get_obstacle_distanceIndex12, get_obstacle_info
import utils
import math
import yaml
import habitat
import magnum as mn
import json
def check_targets_in_habitat(sim, nodes, nav_dict, raw_graph_data, angles=120, distance_resolution=0.25):
    """
    Check if non-zero values in the target map for each node are directly reachable in a straight line in Habitat.
    
    Args:
        sim: Habitat simulator instance.
        nodes: Dictionary containing target maps for each node within a single scan.
        angles: Number of angle divisions (default: 120).
        distance_resolution: Resolution for distance indexing (default: 0.25).
    
    Returns:
        Dictionary with results for each node.
    """
    scan_results = {}
    
    base_height = sim.get_agent(0).state.position[1]
    for node_id, node_data in nodes.items():
        print(f"Node: {node_id}")
        target_map = np.array(node_data['target'])
        obstacle_map = np.array(node_data['obstacle'])
        #source_pos = node_data['source_pos']
        # Ensure source_pos has three coordinates
        #source_pos = np.array([source_pos[0], base_height, source_pos[1]])
        source_pos = np.array(raw_graph_data['nodes'][node_id])
        gau_peak = 10
        # Find indices of non-zero values in the target map
        non_zero_indices = np.argwhere(target_map > 0)
        peak_indices = np.argwhere(target_map == gau_peak)
        # Check if these indices are directly reachable
        valid_targets = True
        invalid_indices = []
        
        for angle_idx, dist_idx in peak_indices:
            # Calculate the heading and distance
            heading = angle_idx * math.pi / (angles / 2)
            # distance = (dist_idx + 1) * distance_resolution
            
            # Calculate the target position
            # target_x = source_pos[0] + distance * math.cos(heading)
            # target_y = source_pos[1]  # Use the same y-coordinate as source
            # target_z = source_pos[2] + distance * math.sin(heading)
            # target_pos = [target_x, target_y, target_z]
            theta = -(heading - np.pi)/2
            rotation = np.quaternion(np.cos(theta),0,np.sin(theta),0)
            distance, index = get_obstacle_info(source_pos, heading, sim)
            print(f"Nav dict:{nav_dict[node_id][str(angle_idx)]['obstacle_distance']}, {nav_dict[node_id][str(angle_idx)]['obstacle_index']}")
            print(f"Obstacle map: {obstacle_map[angle_idx]}")
            print(f"Actual: {distance}, {index}")
            # Step the agent forward and check for collisions
            sim.set_agent_state(source_pos,rotation)
            for _ in range(dist_idx + 1):
                sim.step_without_obs(1)
                if sim.previous_step_collided:
                    valid_targets = False
                    print(f"Invalid location:{angle_idx}, {dist_idx}")
                    invalid_indices.append((angle_idx, dist_idx))
                    break
            
        # Store the result for this node
        scan_results[node_id] = {
            "valid_targets": valid_targets,
            "peak_count": len(peak_indices),
            "invalid_indices": invalid_indices
        }
    
    return scan_results

def create_simulator(config_path, scene_path, scene):
    # Create a Habitat configuration object
    config = habitat.get_config(config_path)
    config.defrost()  # Allow modifications to the config

    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
    config.SIMULATOR.TYPE = 'Sim-v1'
    config.SIMULATOR.SCENE = scene_path.format(scan=scene)
    # config.freeze()
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
    
    return sim

if __name__ == "__main__":
    # Example usage
    config_path = 'gen_training_data/config.yaml'
    scene_path = './data/scene_datasets/mp3d/{scan}/{scan}.glb'

    nav_dict = {}
    for split in ['train']:
        path = f'./gen_training_data/nav_dicts/navigability_dict_{split}.json'
        with open(path) as f:
            data = json.load(f)
            nav_dict.update(data) 

    # Assuming `habitat_sim` is your simulator instance
    nav_dict_path = './training_data/120_train_mp3d_waypoint_twm0.2_obstacle_first_withpos.json'

    navigability_dict = utils.load_gt_navigability(nav_dict_path)

    SPLIT = 'train'
    RAW_GRAPH_PATH= './data/adapted_mp3d_connectivity_graphs/%s.json' 

    with open(RAW_GRAPH_PATH%SPLIT, 'r') as f:
        raw_graph_data = json.load(f)
    
    results = {}
    cnt = 0
    for scene, nodes in navigability_dict.items():
        sim = create_simulator(config_path, scene_path, scene)
        results[scene] = check_targets_in_habitat(sim, navigability_dict[scene], nav_dict[scene], raw_graph_data[scene])
        sim.close()  # Close the simulator after use

        # Print results for the current scan
        print(f"Results for Scan: {scene}")
        for node_id, node_results in results[scene].items():
            print(f"Results for Node: {node_id}")
            if node_results["valid_targets"] == True:
                print("All valid target!")
            else:
                invalid_count = len(node_results["invalid_indices"])
                print(f"  Node {node_id} has invalid targets:")
                print(f"    Peak target count: {node_results['peak_count']}")
                print(f"    Invalid target count: {invalid_count}")

        cnt += 1
        if cnt >= 3:
            break
