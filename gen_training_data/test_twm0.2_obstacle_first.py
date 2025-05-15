import json
import math
import numpy as np
import copy
import torch
import os
import utils
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

# 全局参数设置
ANGLES = 120  # 角度划分数量，将360度划分为120个角度
DISTANCES = 12  # 距离划分数量，将距离空间划分为12个区间
RAW_GRAPH_PATH = '/home/vlnce/habitat_connectivity_graph/%s.json'  # 连接图数据路径模板

RADIUS = 3.25  # 最大前进距离为2米对应的半径值

# 打印运行信息
print('Running TRM-0.2 !!!!!!!!!!')

print('\nProcessing navigability and connectivity to GT maps')
print('Using %s ANGLES and %s DISTANCES'%(ANGLES, DISTANCES))
print('Maximum radius for each waypoint: %s'%(RADIUS))
print('\nConstraining each angle sector has at most one GT waypoint')
print('For all sectors with edges greater than %s, create a virtual waypoint at %s'%(RADIUS, 2.50))
print('\nThis script will return the target map, the obstacle map and the weigh map for each environment')

np.random.seed(7)  # 设置随机种子，确保结果可重复

# 处理不同数据集划分
splits = ['train', 'val_unseen']  # 训练集和未见验证集
for split in splits:
    print('\nProcessing %s split data'%(split))

    # 加载原始连接图数据
    with open(RAW_GRAPH_PATH%split, 'r') as f:
        data = json.load(f)
    # 加载导航字典（如果存在）
    if os.path.exists('./gen_training_data/nav_dicts/navigability_dict_%s.json'%split):
        with open('./gen_training_data/nav_dicts/navigability_dict_%s.json'%split) as f:
            nav_dict = json.load(f)
    
    # 初始化数据结构
    raw_nav_dict = {}
    nodes = {}  # 节点字典
    edges = {}  # 边字典
    obstacles = {}  # 障碍物字典
    
    # 从原始数据中提取节点、边和障碍物信息
    for k, v in data.items():
        nodes[k] = data[k]['nodes']
        edges[k] = data[k]['edges']
        obstacles[k] = nav_dict[k]
    raw_nav_dict['nodes'], raw_nav_dict['edges'], raw_nav_dict['obstacles'] = nodes, edges, obstacles
    data_scans = {
        'nodes': raw_nav_dict['nodes'],
        'edges': raw_nav_dict['edges'],
    }
    obstacle_dict_scans = raw_nav_dict['obstacles']
    scans = list(raw_nav_dict['nodes'].keys())  # 获取所有场景ID列表

    # 初始化统计变量和结果字典
    overall_nav_dict = {}  # 最终的导航字典
    del_nodes = 0  # 删除的节点计数
    count_nodes = 0  # 总节点计数
    target_count = 0  # 目标计数（由于使用高斯，不计数）
    openspace_count = 0  # 开放空间计数
    obstacle_count = 0  # 障碍物计数
    rawedges_count = 0  # 原始边计数
    postedges_count = 0  # 处理后边计数

    # 处理每个场景
    for scan in scans:
        ''' 获取障碍物字典 '''
        obstacle_dict = obstacle_dict_scans[scan]

        ''' 构建连接性字典 '''
        connect_dict = {}  # 节点连接关系字典
        for edge_id, edge_info in data_scans['edges'][scan].items():
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

        ''' 处理每个节点，生成标准数据格式 '''
        navigability_dict = {}  # 可导航性字典
        groundtruth_dict = {}  # 地面真值字典
        count_nodes_i = 0  # 当前场景节点计数
        del_nodes_i = 0  # 当前场景删除节点计数
        
        # 处理每个节点及其邻居
        for node_a, neighbors in connect_dict.items():
            count_nodes += 1
            count_nodes_i += 1
            # 初始化节点的导航字典和地面真值字典
            navigability_dict[node_a] = utils.init_node_nav_dict(ANGLES)
            groundtruth_dict[node_a] = utils.init_node_gt_dict(ANGLES)

            # 获取节点的水平面坐标
            node_a_pos = np.array(data_scans['nodes'][scan][node_a])[[0,2]]
            groundtruth_dict[node_a]['source_pos'] = node_a_pos.tolist()  # 记录源位置

            # 处理每个邻居节点
            for node_b in neighbors:
                # 获取邻居节点的水平面坐标
                node_b_pos = np.array(data_scans['nodes'][scan][node_b])[[0,2]]

                # 计算从当前节点到邻居节点的向量
                edge_vec = (node_b_pos - node_a_pos)
                # 将向量转换为角度索引和距离索引
                angle, angleIndex, \
                distance, \
                distanceIndex = utils.edge_vec_to_indexes(
                    edge_vec, ANGLES)

                # 移除太远或太近的视点
                if distanceIndex == -1:
                    continue
                # 在同一扇区保留更远的关键点
                if navigability_dict[node_a][str(angleIndex)]['has_waypoint']:
                    if distance < navigability_dict[node_a][str(angleIndex)]['waypoint']['distance']:
                        continue

                # 记录路径点信息
                # if distance <= RADIUS:
                navigability_dict[node_a][str(angleIndex)]['has_waypoint'] = True
                navigability_dict[node_a][str(angleIndex)]['waypoint'] = {
                        'node_id': node_b,  # 邻居节点ID
                        'position': node_b_pos,  # 邻居节点位置
                        'angle': angle,  # 角度（弧度）
                        'angleIndex': angleIndex,  # 角度索引
                        'distance': distance,  # 距离（米）
                        'distanceIndex': distanceIndex,  # 距离索引
                    }
                ''' 设置目标图 '''
                groundtruth_dict[node_a]['target'][angleIndex, distanceIndex] = 1
                groundtruth_dict[node_a]['target_pos'].append(node_b_pos.tolist())

            # 记录原始目标数量
            raw_target_count = groundtruth_dict[node_a]['target'].sum()

            # 如果没有目标，删除该节点
            if raw_target_count == 0:
                del(groundtruth_dict[node_a])
                del_nodes += 1
                del_nodes_i += 1                
                continue

            ''' 创建高斯目标图 '''
            gau_peak = 10  # 高斯峰值
            gau_sig_angle = 1.0  # 角度方向的高斯标准差
            gau_sig_dist = 2.0  # 距离方向的高斯标准差
            groundtruth_dict[node_a]['target'] = groundtruth_dict[node_a]['target'].astype(np.float32)

            # 在距离维度上填充零，以便应用高斯滤波
            gau_temp_in = np.concatenate(
                (
                    np.zeros((ANGLES,10)),  # 前填充
                    groundtruth_dict[node_a]['target'],  # 原始目标图
                    np.zeros((ANGLES,10)),  # 后填充
                ), axis=1)

            # 应用高斯滤波，创建平滑的目标分布
            gau_target = gaussian_filter(
                gau_temp_in,
                sigma=(1,2),  # 高斯滤波的标准差
                mode='wrap',  # 使用环绕模式处理边界
            )
            # 裁剪回原始大小
            gau_target = gau_target[:, 10:10+DISTANCES]

            # 归一化并缩放高斯目标
            gau_target_maxnorm = gau_target / gau_target.max()
            groundtruth_dict[node_a]['target'] = gau_peak * gau_target_maxnorm

            # 处理每个角度的障碍物信息
            for k in range(ANGLES):
                # 获取该角度的障碍物距离
                k_dist = obstacle_dict[node_a][str(k)]['obstacle_distance']
                if k_dist is None:
                    k_dist = 100  # 如果没有障碍物，设置一个很大的距离
                navigability_dict[node_a][str(k)]['obstacle_distance'] = k_dist

                # 将障碍物距离转换为距离索引
                k_dindex = utils.get_obstacle_distanceIndex12(k_dist)
                navigability_dict[node_a][str(k)]['obstacle_index'] = k_dindex

                ''' 处理障碍物 '''
                if k_dindex != -1:
                    # 将障碍物前的区域设置为可通行（0）
                    groundtruth_dict[node_a]['obstacle'][k][:k_dindex] = np.zeros(k_dindex)
                else:
                    # 如果没有障碍物，整个方向都是可通行的
                    groundtruth_dict[node_a]['obstacle'][k] = np.zeros(12)


            ''' ********** 非常重要 ********** '''
            ''' 调整目标图 '''
            ''' 障碍物优先 - 障碍物会覆盖目标点 '''

            # 保存原始目标图的副本
            rawt = copy.deepcopy(groundtruth_dict[node_a]['target'])

            # 将目标图与障碍物图相乘，障碍物处的目标被设为0
            groundtruth_dict[node_a]['target'] = \
                groundtruth_dict[node_a]['target'] * (groundtruth_dict[node_a]['obstacle'] == 0)

            # 置信度阈值处理 - 如果最大值小于阈值，删除该节点
            if groundtruth_dict[node_a]['target'].max() < 0.75*gau_peak:
                del(groundtruth_dict[node_a])
                del_nodes += 1
                del_nodes_i += 1
                continue

            # 保存处理后的目标图副本
            postt = copy.deepcopy(groundtruth_dict[node_a]['target'])
            # 统计原始边和处理后边的数量
            rawedges_count += (rawt==gau_peak).sum()
            postedges_count += (postt==gau_peak).sum()

            ''' ********** 非常重要 ********** '''

            # 统计开放空间和障碍物的数量
            openspace_count += (groundtruth_dict[node_a]['obstacle'] == 0).sum()
            obstacle_count += (groundtruth_dict[node_a]['obstacle'] == 1).sum()

            # 将numpy数组转换为列表，以便JSON序列化
            groundtruth_dict[node_a]['target'] = groundtruth_dict[node_a]['target'].tolist()
            groundtruth_dict[node_a]['weight'] = groundtruth_dict[node_a]['weight'].tolist()
            groundtruth_dict[node_a]['obstacle'] = groundtruth_dict[node_a]['obstacle'].tolist()

        # 将当前场景的地面真值字典添加到总导航字典
        overall_nav_dict[scan] = groundtruth_dict

    # 打印处理结果统计信息
    print('Obstacle comes before target !!!')
    print('Number of deleted nodes: %s/%s'%(del_nodes, count_nodes))
    print('Ratio of obstacle behind target: %s/%s'%(postedges_count,rawedges_count))

    print('Ratio of openspace %.5f'%(openspace_count/(openspace_count+obstacle_count)))
    print('Ratio of obstacle %.5f'%(obstacle_count/(openspace_count+obstacle_count)))

    # 保存处理后的导航字典
    with open('./training_data/%s_%s_mp3d_waypoint_twm0.2_obstacle_first_withpos.json'%(ANGLES, split), 'w') as f:
        json.dump(overall_nav_dict, f)
    print('Done')

# import pdb; pdb.set_trace()
