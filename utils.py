import torch
import numpy as np
import sys
import glob
import json

def neighborhoods(mu, x_range, y_range, sigma, circular_x=True, gaussian=False):
    """ 
    生成以mu为中心的掩码，给定x和y范围，原点在输出的中心
    参数:
        mu: 张量 (N, 2)，中心点坐标
        x_range: x轴范围
        y_range: y轴范围
        sigma: 掩码的大小或高斯分布的标准差
        circular_x: 是否在x方向上循环连接（如全景图像）
        gaussian: 是否使用高斯分布而不是二值掩码
    返回:
        张量 (N, y_range, x_range)，生成的掩码
    """
    # 提取中心点的x和y坐标，并扩展维度以便后续广播
    x_mu = mu[:,0].unsqueeze(1).unsqueeze(1)
    y_mu = mu[:,1].unsqueeze(1).unsqueeze(1)

    # 生成以mu为中心的坐标网格
    x = torch.arange(start=0,end=x_range, device=mu.device, dtype=mu.dtype).unsqueeze(0).unsqueeze(0)
    y = torch.arange(start=0,end=y_range, device=mu.device, dtype=mu.dtype).unsqueeze(1).unsqueeze(0)

    # 计算每个点到中心的距离
    y_diff = y - y_mu
    x_diff = x - x_mu
    if circular_x:
        # 如果x轴是循环的（如全景图），取两侧距离的最小值
        x_diff = torch.min(torch.abs(x_diff), torch.abs(x_diff + x_range))
    
    # 越靠近中心越接近1
    if gaussian:
        # 生成高斯分布掩码
        output = torch.exp(-0.5 * ((x_diff/sigma[0])**2 + (y_diff/sigma[1])**2 ))
    else:
        # 生成二值掩码（矩形区域）
        output = torch.logical_and(
            torch.abs(x_diff) <= sigma[0], torch.abs(y_diff) <= sigma[1]
        ).type(mu.dtype)

    return output


def nms(pred, max_predictions=10, sigma=(1.0,1.0), gaussian=False):
    ''' 
    非极大值抑制函数，用于查找预测图中的局部最大值
    参数:
        pred: 输入预测图，形状为 (batch_size, 1, height, width)
        max_predictions: 每个样本要保留的最大预测点数
        sigma: 抑制区域的大小
        gaussian: 是否使用高斯抑制而不是矩形抑制
    返回:
        经过非极大值抑制的预测图
    '''

    shape = pred.shape

    # 初始化输出和抑制用的预测图副本
    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0],-1))  # (BATCH_SIZE, height*width)
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0],-1))  # (BATCH_SIZE, height*width)

    # 迭代寻找最大预测点
    for i in range(max_predictions):
        # 找到当前预测图中的全局最大值
        flat_supp_pred = supp_pred.reshape((shape[0],-1))
        val, ix = torch.max(flat_supp_pred, dim=1)
        indices = torch.arange(0,shape[0])
        # 将最大值保存到输出图中
        flat_output[indices,ix] = flat_pred[indices,ix]

        # 计算抑制区域
        # 将一维索引转换为二维坐标
        y = ix / shape[-1]
        x = ix % shape[-1]
        mu = torch.stack([x,y], dim=1).float() # 将x,y坐标堆叠成(batch_size, 2)形状的张量

        # 生成以最大值为中心的抑制掩码
        g = neighborhoods(mu, shape[-1], shape[-2], sigma, gaussian=gaussian)

        # 抑制靠近中心极大值的区域
        supp_pred *= (1-g.unsqueeze(1))

    # 确保没有负值
    output[output < 0] = 0 
    return output 


def get_gt_nav_map(num_angles, nav_dict, scan_ids, waypoint_ids):
    ''' 
    获取地面真值导航图，包括目标图、障碍物图和权重图
    参数:
        num_angles: 角度数量（通常是图像划分的数量）
        nav_dict: 包含导航信息的字典
        scan_ids: 扫描ID列表
        waypoint_ids: 路径点ID列表
    返回:
        target: 目标图，1表示地面真值关键点，2表示忽略的索引
        obstacle: 障碍物图
        weight: 权重图，0表示忽略，1表示路径点/远离路径点/障碍物，(0,1)表示其他开放空间
        source_pos: 源位置列表
        target_pos: 目标位置列表
    '''
    bs = len(scan_ids)  # 批次大小
    # 初始化张量
    target = torch.zeros(bs, num_angles, 12)
    obstacle = torch.zeros(bs, num_angles, 12)
    weight = torch.zeros(bs, num_angles, 12)
    source_pos = []
    target_pos = []

    # 为每个样本填充数据
    for i in range(bs):
        target[i] = torch.tensor(nav_dict[scan_ids[i]][waypoint_ids[i]]['target'])
        obstacle[i] = torch.tensor(nav_dict[scan_ids[i]][waypoint_ids[i]]['obstacle'])
        weight[i] = torch.tensor(nav_dict[scan_ids[i]][waypoint_ids[i]]['weight'])
        source_pos.append(nav_dict[scan_ids[i]][waypoint_ids[i]]['source_pos'])
        target_pos.append(nav_dict[scan_ids[i]][waypoint_ids[i]]['target_pos'])

    return target, obstacle, weight, source_pos, target_pos


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=10):
    """
    在循环中调用以创建终端进度条
    参数:
        iteration   - 必需  : 当前迭代次数 (Int)
        total       - 必需  : 总迭代次数 (Int)
        prefix      - 可选  : 前缀字符串 (Str)
        suffix      - 可选  : 后缀字符串 (Str)
        decimals    - 可选  : 百分比完成度中的正小数位数 (Int)
        bar_length  - 可选  : 进度条的字符长度 (Int)
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
    保存模型检查点
    参数:
        epoch: 当前训练轮次
        net: 网络模型
        net_optimizer: 网络优化器
        path: 保存路径
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
    加载模型参数（但不加载训练状态）
    参数:
        net: 网络模型
        net_optimizer: 网络优化器
        path: 检查点路径
    返回:
        epoch: 加载的轮次
        net: 加载参数后的网络模型
        net_optimizer: 加载参数后的网络优化器
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
    生成Transformer注意力掩码，控制每个位置只能关注自己和邻近的位置
    参数:
        num_imgs: 图像数量，默认24（通常对应360度全景的划分）
        neighbor: 每侧允许关注的邻居数量，默认2
    返回:
        形状为[1,1,num_imgs,num_imgs]的注意力掩码张量，1表示允许关注，0表示禁止关注
    """
    assert neighbor <= 5  # 确保邻居数不超过5

    # 初始化全零掩码矩阵
    mask = np.zeros((num_imgs,num_imgs))
    
    # 创建模板行，表示单个位置的注意力模式
    t = np.zeros(num_imgs)
    t[:neighbor+1] = np.ones(neighbor+1)  # 自身和右侧neighbor个位置设为1
    if neighbor != 0:
        t[-neighbor:] = np.ones(neighbor)  # 左侧neighbor个位置设为1
    
    # 循环填充掩码矩阵的每一行
    for ri in range(num_imgs):
        mask[ri] = t  # 将当前模板填入第ri行
        t = np.roll(t, 1)  # 循环右移模板，准备下一行

    # 返回重塑为Transformer注意力掩码格式的张量
    return torch.from_numpy(mask).reshape(1,1,num_imgs,num_imgs).long()


def load_gt_navigability(path):
    ''' 
    加载路径点地面真值导航性数据
    参数:
        path: 数据文件路径前缀
    返回:
        all_scans_nav_map: 包含所有扫描导航图的字典
    '''
    all_scans_nav_map = {}
    gt_dir = glob.glob('%s*'%(path))
    for gt_dir_i in gt_dir:
        with open(gt_dir_i, 'r') as f:
            nav_map = json.load(f)
        for scan_id, values in nav_map.items():
            all_scans_nav_map[scan_id] = values
    return all_scans_nav_map
