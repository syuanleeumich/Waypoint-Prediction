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
    设置随机种子和创建实验目录
    参数:
        args: 命令行参数
    """
    torch.manual_seed(0)  # 设置PyTorch随机种子
    random.seed(0)  # 设置Python随机种子
    exp_log_path = './checkpoints/%s/'%(args.EXP_ID)  # 实验日志路径
    os.makedirs(exp_log_path, exist_ok=True)  # 创建实验目录
    exp_log_path = './checkpoints/%s/snap/'%(args.EXP_ID)  # 模型快照路径
    os.makedirs(exp_log_path, exist_ok=True)  # 创建模型快照目录

class Param():
    """参数类，处理命令行参数"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train waypoint predictor')

        # 实验设置
        self.parser.add_argument('--EXP_ID', type=str, default='test_0')  # 实验ID
        self.parser.add_argument('--TRAINEVAL', type=str, default='train', help='trian or eval mode')  # 训练或评估模式
        self.parser.add_argument('--VIS', type=int, default=0, help='visualize predicted hearmaps')  # 是否可视化预测热图
        # self.parser.add_argument('--LOAD_EPOCH', type=int, default=None, help='specific an epoch to load for eval')

        # 模型结构参数
        self.parser.add_argument('--ANGLES', type=int, default=24)  # 角度划分数量
        self.parser.add_argument('--NUM_IMGS', type=int, default=24)  # 图像数量
        self.parser.add_argument('--NUM_CLASSES', type=int, default=12)  # 类别数量
        self.parser.add_argument('--MAX_NUM_CANDIDATES', type=int, default=5)  # 最大候选点数量

        self.parser.add_argument('--PREDICTOR_NET', type=str, default='TRM', help='TRM only')  # 预测网络类型

        # 训练参数
        self.parser.add_argument('--EPOCH', type=int, default=10)  # 训练轮数
        self.parser.add_argument('--BATCH_SIZE', type=int, default=2)  # 批次大小
        self.parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)  # 学习率
        self.parser.add_argument('--WEIGHT', type=int, default=0, help='weight the target map')  # 是否加权目标图

        # Transformer模型参数
        self.parser.add_argument('--TRM_LAYER', default=2, type=int, help='number of TRM hidden layers')  # TRM隐藏层数
        self.parser.add_argument('--TRM_NEIGHBOR', default=2, type=int, help='number of attention mask neighbor')  # 注意力掩码邻居数
        self.parser.add_argument('--HEATMAP_OFFSET', default=2, type=int, help='an offset determined by image FoV and number of images')  # 热图偏移量
        self.parser.add_argument('--HIDDEN_DIM', default=768, type=int)  # 隐藏层维度

        self.args = self.parser.parse_args()

def predict_waypoints(args):
    """
    路径点预测主函数
    参数:
        args: 命令行参数
    """
    print('\nArguments', args)
    log_dir = './checkpoints/%s/tensorboard/'%(args.EXP_ID)  # TensorBoard日志目录
    writer = SummaryWriter(log_dir=log_dir)  # 创建TensorBoard写入器

    ''' 初始化网络模型 '''
    # 初始化RGB编码器，使用预训练权重，不进行微调
    rgb_encoder = RGBEncoder(resnet_pretrain=True, trainable=False).to(device)
    # 初始化深度编码器，使用预训练权重，不进行微调
    depth_encoder = DepthEncoder(resnet_pretrain=True, trainable=False).to(device)
    if args.PREDICTOR_NET == 'TRM':
        print('\nUsing TRM predictor')
        print('HIDDEN_DIM default to 768')
        args.HIDDEN_DIM = 768
        # 初始化基于Transformer的二元分布预测器
        predictor = BinaryDistPredictor_TRM(args=args,
            hidden_dim=args.HIDDEN_DIM, n_classes=args.NUM_CLASSES).to(device)

    ''' 加载可导航性数据（地面真值路径点、障碍物和权重） '''
    navigability_dict = utils.load_gt_navigability(
        './training_data/%s_*_mp3d_waypoint_twm0.2_obstacle_first_withpos.json'%(args.ANGLES))

    ''' 为RGB和深度图像创建数据加载器 '''
    train_img_dir = './gen_training_data/rgbd_fov90/train/*/*.pkl'  # 训练图像目录
    traindataloader = RGBDepthPano(args, train_img_dir, navigability_dict)  # 训练数据加载器
    eval_img_dir = './gen_training_data/rgbd_fov90/val_unseen/*/*.pkl'  # 评估图像目录
    evaldataloader = RGBDepthPano(args, eval_img_dir, navigability_dict)  # 评估数据加载器
    if args.TRAINEVAL == 'train':
        trainloader = torch.utils.data.DataLoader(traindataloader, 
        batch_size=args.BATCH_SIZE, shuffle=True, num_workers=4)  # 训练数据批次加载器
    evalloader = torch.utils.data.DataLoader(evaldataloader, 
        batch_size=args.BATCH_SIZE, shuffle=False, num_workers=4)  # 评估数据批次加载器

    ''' 定义损失函数和优化器 '''
    criterion_bcel = torch.nn.BCEWithLogitsLoss(reduction='none')  # 二元交叉熵损失
    criterion_mse = torch.nn.MSELoss(reduction='none')  # 均方误差损失

    params = list(predictor.parameters())  # 获取预测器参数
    optimizer = torch.optim.AdamW(params, lr=args.LEARNING_RATE)  # 使用AdamW优化器

    ''' 训练循环 '''
    if args.TRAINEVAL == 'train':
        print('\nTraining starts')
        # 记录最佳验证结果
        best_val_1 = {"avg_wayscore": 0.0, "log_string": '', "update":False}  # 基于平均路径点得分的最佳结果
        best_val_2 = {"avg_pred_distance": 10.0, "log_string": '', "update":False}  # 基于平均预测距离的最佳结果

        for epoch in range(args.EPOCH):  # 遍历每个训练轮次
            sum_loss = 0.0  # 累计损失

            # 设置编码器为评估模式（不更新参数）
            rgb_encoder.eval()
            depth_encoder.eval()
            # 设置预测器为训练模式
            predictor.train()

            # 遍历训练数据
            for i, data in enumerate(trainloader):
                scan_ids = data['scan_id']  # 场景ID
                waypoint_ids = data['waypoint_id']  # 路径点ID
                rgb_imgs = data['rgb'].to(device)  # RGB图像
                depth_imgs = data['depth'].to(device)  # 深度图像

                ''' 检查图像方向（已注释代码） '''
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

                ''' 处理观察数据 '''
                rgb_feats = rgb_encoder(rgb_imgs)        # 提取RGB特征 (BATCH_SIZE*ANGLES, 2048)
                depth_feats = depth_encoder(depth_imgs)  # 提取深度特征 (BATCH_SIZE*ANGLES, 128, 4, 4)

                ''' 获取学习目标 '''
                # 获取地面真值导航图（目标、障碍物、权重）
                target, obstacle, weight, _, _ = utils.get_gt_nav_map(
                    args.ANGLES, navigability_dict, scan_ids, waypoint_ids)
                target = target.to(device)
                obstacle = obstacle.to(device)
                weight = weight.to(device)

                # 使用TRM预测器进行预测
                if args.PREDICTOR_NET == 'TRM':
                    vis_logits = TRM_predict('train', args,
                        predictor, rgb_feats, depth_feats)

                    # 计算损失
                    loss_vis = criterion_mse(vis_logits, target)
                    if args.WEIGHT:
                        loss_vis = loss_vis * weight  # 如果启用加权，应用权重
                    total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES  # 计算总损失

                # 反向传播和优化
                optimizer.zero_grad()  # 清除梯度
                total_loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                sum_loss += total_loss.item()  # 累加损失

                # 打印训练进度
                print_progress(i+1, len(trainloader), prefix='Epoch: %d/%d'%((epoch+1),args.EPOCH))
            
            # 记录训练损失到TensorBoard
            writer.add_scalar("Train/Loss", sum_loss/(i+1), epoch)
            print('Train Loss: %.5f' % (sum_loss/(i+1)))  # 打印平均训练损失

            ''' 评估阶段 '''
            # print('Evaluation ...')
            sum_loss = 0.0  # 评估损失
            # 存储预测结果
            predictions = {'sample_id': [], 
                'source_pos': [], 'target_pos': [],
                'probs': [], 'logits': [],
                'target': [], 'obstacle': [], 'sample_loss': []}

            # 设置所有网络为评估模式
            rgb_encoder.eval()
            depth_encoder.eval()
            predictor.eval()

            # 遍历评估数据
            for i, data in enumerate(evalloader):
                scan_ids = data['scan_id']
                waypoint_ids = data['waypoint_id']
                sample_id = data['sample_id']
                rgb_imgs = data['rgb'].to(device)
                depth_imgs = data['depth'].to(device)

                # 获取地面真值导航图
                target, obstacle, weight, \
                source_pos, target_pos = utils.get_gt_nav_map(
                    args.ANGLES, navigability_dict, scan_ids, waypoint_ids)
                target = target.to(device)
                obstacle = obstacle.to(device)
                weight = weight.to(device)

                ''' 处理观察数据 '''
                rgb_feats = rgb_encoder(rgb_imgs)        # 提取RGB特征
                depth_feats = depth_encoder(depth_imgs)  # 提取深度特征

                # 使用TRM预测器进行预测
                if args.PREDICTOR_NET == 'TRM':
                    vis_probs, vis_logits = TRM_predict('eval', args,
                        predictor, rgb_feats, depth_feats)
                    overall_probs = vis_probs  # 总体概率
                    overall_logits = vis_logits  # 总体逻辑值
                    
                    # 计算损失
                    loss_vis = criterion_mse(vis_logits, target)
                    if args.WEIGHT:
                        loss_vis = loss_vis * weight  # 如果启用加权，应用权重
                    sample_loss = loss_vis.sum(-1).sum(-1) / args.ANGLES  # 样本损失
                    total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES  # 总损失

                # 累加损失并存储预测结果
                sum_loss += total_loss.item()
                predictions['sample_id'].append(sample_id)
                predictions['source_pos'].append(source_pos)
                predictions['target_pos'].append(target_pos)
                predictions['probs'].append(overall_probs.tolist())
                predictions['logits'].append((overall_logits.tolist()))
                predictions['target'].append(target.tolist())
                predictions['obstacle'].append(obstacle.tolist())
                predictions['sample_loss'].append(target.tolist())

            # 打印评估损失
            print('Eval Loss: %.5f' % (sum_loss/(i+1)))
            # 评估预测结果
            results = waypoint_eval(args, predictions)
            
            # 记录评估指标到TensorBoard
            writer.add_scalar("Evaluation/Loss", sum_loss/(i+1), epoch)
            writer.add_scalar("Evaluation/p_waypoint_openspace", results['p_waypoint_openspace'], epoch)
            writer.add_scalar("Evaluation/p_waypoint_obstacle", results['p_waypoint_obstacle'], epoch)
            writer.add_scalar("Evaluation/avg_wayscore", results['avg_wayscore'], epoch)
            writer.add_scalar("Evaluation/avg_pred_distance", results['avg_pred_distance'], epoch)
            
            # 构建日志字符串
            log_string = 'Epoch %s '%(epoch)
            for key, value in results.items():
                if key != 'candidates': 
                    log_string += '{} {:.5f} | '.format(str(key), value)
            print(log_string)  # 打印评估结果

            # 保存检查点 - 基于平均路径点得分
            if results['avg_wayscore'] > best_val_1['avg_wayscore']:
                checkpoint_save_path = './checkpoints/%s/snap/check_val_best_avg_wayscore'%(args.EXP_ID)
                utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_save_path)
                print('New best avg_wayscore result found, checkpoint saved to %s'%(checkpoint_save_path))
                best_val_1['avg_wayscore'] = results['avg_wayscore']
                best_val_1['log_string'] = log_string
            
            # 保存最新检查点
            checkpoint_reg_save_path = './checkpoints/%s/snap/check_latest'%(args.EXP_ID)
            utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_reg_save_path)
            print('Best avg_wayscore result til now: ', best_val_1['log_string'])

            # 保存检查点 - 基于平均预测距离
            if results['avg_pred_distance'] < best_val_2['avg_pred_distance']:
                checkpoint_save_path = './checkpoints/%s/snap/check_val_best_avg_pred_distance'%(args.EXP_ID)
                utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_save_path)
                print('New best avg_pred_distance result found, checkpoint saved to %s'%(checkpoint_save_path))
                best_val_2['avg_pred_distance'] = results['avg_pred_distance']
                best_val_2['log_string'] = log_string
            
            # 再次保存最新检查点（冗余操作）
            checkpoint_reg_save_path = './checkpoints/%s/snap/check_latest'%(args.EXP_ID)
            utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_reg_save_path)
            print('Best avg_pred_distance result til now: ', best_val_2['log_string'])

    elif args.TRAINEVAL == 'eval':
        ''' 评估模式 - 推理（带有一点专家混合） '''
        print('\nEvaluation mode, please doublecheck EXP_ID and LOAD_EPOCH')
        # 加载最佳检查点
        checkpoint_load_path = './checkpoints/%s/snap/check_val_best_avg_wayscore'%(args.EXP_ID)
        epoch, predictor, optimizer = utils.load_checkpoint(
                        predictor, optimizer, checkpoint_load_path)

        sum_loss = 0.0
        # 存储预测结果
        predictions = {'sample_id': [], 
            'source_pos': [], 'target_pos': [],
            'probs': [], 'logits': [],
            'target': [], 'obstacle': [], 'sample_loss': []}

        # 设置所有网络为评估模式
        rgb_encoder.eval()
        depth_encoder.eval()
        predictor.eval()

        # 遍历评估数据
        for i, data in enumerate(evalloader):
            # 如果启用可视化并且已处理5个样本，则跳出循环
            if args.VIS and i == 5:
                break

            scan_ids = data['scan_id']
            waypoint_ids = data['waypoint_id']
            sample_id = data['sample_id']
            rgb_imgs = data['rgb'].to(device)
            depth_imgs = data['depth'].to(device)

            # 获取地面真值导航图
            target, obstacle, weight, \
            source_pos, target_pos = utils.get_gt_nav_map(
                args.ANGLES, navigability_dict, scan_ids, waypoint_ids)
            target = target.to(device)
            obstacle = obstacle.to(device)
            weight = weight.to(device)

            ''' 处理观察数据 '''
            rgb_feats = rgb_encoder(rgb_imgs)        # 提取RGB特征
            depth_feats = depth_encoder(depth_imgs)  # 提取深度特征

            ''' 预测路径点概率 '''
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

            # 累加损失并存储预测结果
            sum_loss += total_loss.item()
            predictions['sample_id'].append(sample_id)
            predictions['source_pos'].append(source_pos)
            predictions['target_pos'].append(target_pos)
            predictions['probs'].append(overall_probs.tolist())
            predictions['logits'].append(overall_logits.tolist())
            predictions['target'].append(target.tolist())
            predictions['obstacle'].append(obstacle.tolist())
            predictions['sample_loss'].append(target.tolist())

        # 打印评估损失
        print('Eval Loss: %.5f' % (sum_loss/(i+1)))
        # 评估预测结果
        results = waypoint_eval(args, predictions)
        
        # 构建日志字符串
        log_string = 'Epoch %s '%(epoch)
        for key, value in results.items():
            if key != 'candidates':
                log_string += '{} {:.5f} | '.format(str(key), value)
        print(log_string)
        print('Evaluation Done')

    else:
        RunningModeError  # 运行模式错误

if __name__ == "__main__":
    param = Param()  # 创建参数对象
    args = param.args  # 获取命令行参数
    setup(args)  # 设置实验环境

    # 如果启用可视化，确保是评估模式
    if args.VIS:
        assert args.TRAINEVAL == 'eval'

    predict_waypoints(args)  # 执行路径点预测
