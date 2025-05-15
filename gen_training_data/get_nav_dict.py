import json
import numpy as np
import utils
import habitat
from habitat.sims import make_sim
from utils import Simulator

# 配置文件和路径设置
config_path = 'gen_training_data/config.yaml'  # Habitat配置文件路径
scene_path = '/home/vlnce/vln-ce/data/scene_datasets/mp3d/{scan}/{scan}.glb'  # 3D场景模型路径模板
RAW_GRAPH_PATH= '/home/vlnce/habitat_connectivity_graph/%s.json'  # 连接图数据路径模板
NUMBER = 120  # 角度划分数量，将360度划分为120个角度

SPLIT = 'val_unseen'  # 数据集划分（未见验证集）

# 加载连接图数据
with open(RAW_GRAPH_PATH%SPLIT, 'r') as f:
    raw_graph_data = json.load(f)

nav_dict = {}  # 导航字典，存储所有场景的导航信息
total_invalids = 0  # 无效点计数
total = 0  # 总点数计数

# 遍历每个场景
for scene, data in raw_graph_data.items():
    ''' 构建连接性字典 '''
    connect_dict = {}  # 节点连接关系字典
    for edge_id, edge_info in data['edges'].items():
        node_a = edge_info['nodes'][0]  # 边的第一个节点
        node_b = edge_info['nodes'][1]  # 边的第二个节点

        # 为每个节点添加其相邻节点信息
        if node_a not in connect_dict:
            connect_dict[node_a] = [node_b]
        else:
            connect_dict[node_a].append(node_b)
        if node_b not in connect_dict:
            connect_dict[node_b] = [node_a]
        else:
            connect_dict[node_b].append(node_a)


    '''创建Habitat模拟器实例用于障碍物检查'''
    config = habitat.get_config(config_path)  # 加载配置
    config.defrost()  # 解冻配置以进行修改
    # config.TASK.POSSIBLE_ACTIONS = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'FORWARD_BY_DIS']
    # config.SIMULATOR.AGENT_0.SENSORS = []
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # 设置前进步长
    config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False  # 禁止滑动
    config.SIMULATOR.TYPE = 'Sim-v1'  # 设置模拟器类型
    config.SIMULATOR.SCENE = scene_path.format(scan=scene)  # 设置场景路径
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)  # 创建模拟器实例

    ''' 处理每个节点，生成标准数据格式 '''
    navigability_dict = {}  # 可导航性字典，存储当前场景的导航信息
    total = len(connect_dict)  # 当前场景的节点总数
    for i, pair in enumerate(connect_dict.items()):
        node_a, neighbors = pair  # 当前节点及其邻居节点
        # 初始化当前节点的可导航性字典，包含NUMBER个方向的信息
        navigability_dict[node_a] = utils.init_single_node_dict(number=NUMBER)
        # 提取节点的x,z坐标（水平面坐标）
        node_a_pos = np.array(data['nodes'][node_a])[[0,2]]
    
        # 获取节点在Habitat中的完整位置（x,y,z）
        habitat_pos = np.array(data['nodes'][node_a])
        # 检查每个方向的障碍物信息
        for id, info in navigability_dict[node_a].items():
            # 获取特定方向的障碍物距离和索引
            obstacle_distance, obstacle_index = utils.get_obstacle_info(habitat_pos,info['heading'],sim)
            info['obstacle_distance'] = obstacle_distance  # 记录障碍物距离
            info['obstacle_index'] = obstacle_index  # 记录障碍物索引
    
        # 处理每个邻居节点，建立导航关系
        for node_b in neighbors:
            # 获取邻居节点的水平面坐标
            node_b_pos = np.array(data['nodes'][node_b])[[0,2]]
    
            # 计算从当前节点到邻居节点的向量
            edge_vec = (node_b_pos - node_a_pos)
            # 将向量转换为角度索引和距离索引
            angle, angleIndex, distance, distanceIndex = utils.edge_vec_to_indexes(edge_vec,number=NUMBER)
    
            # 在对应角度索引的方向上标记存在路径点
            navigability_dict[node_a][str(angleIndex)]['has_waypoint'] = True
            # 添加路径点详细信息
            navigability_dict[node_a][str(angleIndex)]['waypoint'].append(
                {
                    'node_id': node_b,  # 邻居节点ID
                    'position': node_b_pos.tolist(),  # 邻居节点位置
                    'angle': angle,  # 角度（弧度）
                    'angleIndex': angleIndex,  # 角度索引
                    'distance': distance,  # 距离（米）
                    'distanceIndex': distanceIndex,  # 距离索引
                })
        utils.print_progress(i+1,total)  # 打印进度
    
    # 将当前场景的导航信息添加到总导航字典
    nav_dict[scene] = navigability_dict
    sim.close()  # 关闭模拟器

# 保存导航字典到文件
output_path = './gen_training_data/nav_dicts/navigability_dict_%s.json'%SPLIT
with open(output_path, 'w') as fo:
    json.dump(nav_dict, fo, ensure_ascii=False, indent=4)
