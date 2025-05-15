import json
import numpy as np
import utils
import habitat
import os
import pickle
from habitat.sims import make_sim


# 配置文件和路径设置
config_path = './gen_training_data/config.yaml'  # Habitat配置文件路径
scene_path = '/home/vlnce/vln-ce/data/scene_datasets/mp3d/{scan}/{scan}.glb'  # 3D场景模型路径模板
image_path = './training_data/rgbd_fov90/'  # 保存RGB-D图像的基础路径
save_path = os.path.join(image_path,'{split}/{scan}/{scan}_{node}_mp3d_imgs.pkl')  # 保存图像的具体路径模板
RAW_GRAPH_PATH= '/home/vlnce/habitat_connectivity_graph/%s.json'  # 连接图数据路径模板
NUMBER = 12  # 每个节点采集的图像数量（对应不同视角）

SPLIT = 'train'  # 数据集划分（训练集）

# 加载连接图数据
with open(RAW_GRAPH_PATH%SPLIT, 'r') as f:
    raw_graph_data = json.load(f)

nav_dict = {}  # 导航字典
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
    config.TASK.SENSORS = []  # 清空传感器配置
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # 设置前进步长
    config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False  # 禁止滑动
    config.SIMULATOR.SCENE = scene_path.format(scan=scene)  # 设置场景路径
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)  # 创建模拟器实例

    '''保存图像数据'''
    # 确保保存路径存在
    if not os.path.exists(image_path+'{split}/{scan}'.format(split=SPLIT,scan=scene)):
        os.makedirs(image_path+'{split}/{scan}'.format(split=SPLIT,scan=scene))
    navigability_dict = {}  # 可导航性字典
    
    i = 0  # 节点计数器
    # 遍历每个节点及其邻居
    for node_a, neighbors in connect_dict.items():
        # 初始化当前节点的可导航性字典
        navigability_dict[node_a] = utils.init_single_node_dict(number=NUMBER)
        rgbs = []  # RGB图像列表
        depths = []  # 深度图像列表
        node_a_pos = np.array(data['nodes'][node_a])[[0, 2]]  # 提取节点的x,z坐标

        # 获取节点在Habitat中的位置
        habitat_pos = np.array(data['nodes'][node_a])
        # 为每个视角采集图像
        for info in navigability_dict[node_a].values():
            position, heading = habitat_pos, info['heading']  # 位置和朝向
            theta = -(heading - np.pi) / 2  # 计算旋转角度
            # 创建四元数表示旋转
            rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
            # 在指定位置和朝向获取观察结果
            obs = sim.get_observations_at(position, rotation)
            rgbs.append(obs['rgb'])  # 添加RGB图像
            depths.append(obs['depth'])  # 添加深度图像
        
        # 保存当前节点的所有图像数据
        with open(save_path.format(split=SPLIT, scan=scene, node=node_a), 'wb') as f:
            pickle.dump({'rgb': np.array(rgbs),
                         'depth': np.array(depths, dtype=np.float16)}, f)
        utils.print_progress(i+1,total)  # 打印进度
        i+=1

    sim.close()  # 关闭模拟器
