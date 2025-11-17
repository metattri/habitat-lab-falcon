"""
Falcon Habitat 风格推理流程
完整复现原始项目的推理逻辑，包括所有Habitat组件
"""

import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
from gym import spaces
import sys
import os

from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import (
    batch_obs,
    get_action_space_info,
    inference_mode,
)


class FalconInferenceEngine:
    """
    Falcon 推理引擎
    复现完整的 Habitat 推理流程，包括:
    - 策略加载
    - 观测预处理
    - 动作采样
    - RNN状态管理
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cpu',
        use_multi_agent: bool = True,
    ):
        """
        初始化推理引擎
        
        Args:
            checkpoint_path: 检查点路径
            device: 运行设备 ('cpu' or 'cuda')
            use_multi_agent: 是否使用多智能体格式
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.use_multi_agent = use_multi_agent
        
        # 加载检查点
        print(f"[FalconInferenceEngine] Loading checkpoint: {checkpoint_path}")
        self.ckpt_dict = self._load_checkpoint()
        
        # 创建观测和动作空间
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
        # 创建策略网络
        self.policy = self._create_policy()
        self._load_policy_weights()
        
        # 设置为评估模式
        self.policy.eval()
        
        # 获取动作空间信息
        self.action_shape, self.discrete_actions = get_action_space_info(
            self.policy.policy_action_space
        )
        
        print(f"[FalconInferenceEngine] Initialization complete")
        print(f"  Device: {self.device}")
        print(f"  Action space: {self.action_space}")
        print(f"  Action shape: {self.action_shape}")
        print(f"  Discrete actions: {self.discrete_actions}")
    
    def _load_checkpoint(self) -> Dict:
        """加载检查点"""
        ckpt = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # 处理多智能体格式
        if self.use_multi_agent and 0 in ckpt:
            print(f"[FalconInferenceEngine] Detected multi-agent checkpoint")
            return {
                'state_dict': ckpt[0]['state_dict'],
                'config': ckpt.get('config', None),
                'extra_state': ckpt.get('extra_state', {}),
            }
        else:
            return ckpt
    
    def _create_observation_space(self) -> spaces.Dict:
        """创建观测空间（使用策略网络期望的键名）"""
        return spaces.Dict({
            'articulated_agent_jaw_depth': spaces.Box(
                low=0,
                high=1,
                shape=(256, 256, 1),
                dtype=np.float32,
            ),
            'pointgoal_with_gps_compass': spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
        })
    
    def _create_action_space(self) -> spaces.Discrete:
        """创建动作空间"""
        return spaces.Discrete(4)  # STOP, FORWARD, LEFT, RIGHT
    
    def _create_policy(self) -> PointNavResNetPolicy:
        """创建策略网络（直接构造，避免复杂的配置依赖）"""
        # 直接构造策略，而不是使用 from_config
        # 
        # 注意：两种不同的归一化！
        # 1. Habitat 传感器归一化（normalize_depth=True）：
        #    - 在传感器层面进行：depth = (depth - min_depth) / (max_depth - min_depth)
        #    - 配置：min_depth=0.0, max_depth=10.0
        #    - 输出：[0, 1] 范围的深度图
        #    - 这个已经在环境中完成，推理时会自动应用
        # 
        # 2. ResNet 网络归一化（normalize_visual_inputs）：
        #    - 在网络层面进行：x = (x - running_mean) / sqrt(running_var)
        #    - 使用 RunningMeanAndVar 在线统计均值和方差
        #    - 训练时根据 "rgb" in observation_space 自动决定
        #    - Falcon 只有 depth，所以 normalize_visual_inputs=False
        # 
        # 检查点验证：没有 running_mean_and_var 的权重，确认训练时未启用网络归一化
        policy = PointNavResNetPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            hidden_size=512,
            num_recurrent_layers=2,
            rnn_type="LSTM",
            backbone="resnet50",
            normalize_visual_inputs=False,  # 网络层归一化：与训练保持一致
            force_blind_policy=False,
            policy_config=None,
            aux_loss_config=None,
            fuse_keys=None,
        )
        policy.to(self.device)
        return policy
    
    def _create_default_config(self):
        """创建默认配置"""
        from omegaconf import DictConfig, OmegaConf
        
        # 模拟完整的 Habitat 配置
        config = OmegaConf.create({
            'habitat_baselines': {
                'rl': {
                    'ppo': {
                        'hidden_size': 512,
                    },
                    'ddppo': {
                        'backbone': 'resnet50',
                        'rnn_type': 'LSTM',
                        'num_recurrent_layers': 2,
                        'normalize_visual_inputs': True,
                        'force_blind_policy': False,
                        'pretrained_encoder': False,
                        'pretrained': False,
                    },
                    'auxiliary_losses': {},
                },
                'eval': {
                    'extra_sim_sensors': {},
                }
            }
        })
        return config
    
    def _load_policy_weights(self):
        """加载策略权重"""
        state_dict = self.ckpt_dict['state_dict']
        
        # 直接加载，不做任何键名转换
        # PointNavResNetPolicy 的结构完全匹配检查点：
        #   - net.visual_encoder.xxx
        #   - net.visual_fc.xxx
        #   - net.state_encoder.rnn.xxx  (RNNStateEncoder 包装的 LSTM)
        #   - net.prev_action_embedding.xxx
        #   - net.tgt_embeding.xxx
        #   - action_distribution.linear.xxx  (CategoricalNet 的 linear 子模块)
        #   - critic.fc.xxx  (CriticHead 的 fc 子模块)
        
        # 加载到策略
        missing_keys, unexpected_keys = self.policy.load_state_dict(
            state_dict,
            strict=False
        )
        
        print(f"[FalconInferenceEngine] Loaded policy weights")
        if missing_keys:
            # 过滤掉不重要的缺失键
            important_missing = [k for k in missing_keys if 'running_mean_and_var' not in k]
            if important_missing:
                print(f"  重要的缺失键 ({len(important_missing)}): {important_missing[:5]}")
        if unexpected_keys:
            important_unexpected = [k for k in unexpected_keys if 'running_mean_and_var' not in k]
            if important_unexpected:
                print(f"  重要的意外键 ({len(important_unexpected)}): {important_unexpected[:5]}")
    
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        重置推理状态（新 episode 开始）
        
        Returns:
            recurrent_hidden_states: (num_layers, batch, hidden_size)
            prev_actions: (batch, action_dim)
            not_done_masks: (batch, 1)
        """
        batch_size = 1
        
        # 初始化 RNN 隐藏状态
        recurrent_hidden_states = torch.zeros(
            (batch_size, *self.policy.hidden_state_shape),
            device=self.device,
        )
        
        # 初始化前一个动作
        prev_actions = torch.zeros(
            batch_size,
            *self.action_shape,
            device=self.device,
            dtype=torch.long if self.discrete_actions else torch.float,
        )
        
        # 初始化 not_done_masks
        # 注意：第一步应该是 False（告诉LSTM这是新episode的开始）
        # 但从第二步开始应该设置为 True 来保持RNN状态的连续性
        not_done_masks = torch.zeros(
            batch_size,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        
        return recurrent_hidden_states, prev_actions, not_done_masks
    
    def prepare_batch(
        self,
        depth: np.ndarray,
        pointgoal: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """
        准备观测批次（使用策略网络期望的键名）
        
        Args:
            depth: (H, W, C) 深度图
            pointgoal: (2,) 目标点
        
        Returns:
            batch: 字典格式的观测批次
        """
        # 创建观测字典（使用策略网络期望的键名）
        observations = [{
            'articulated_agent_jaw_depth': depth,
            'pointgoal_with_gps_compass': pointgoal,
        }]
        
        # 使用 Habitat 的 batch_obs 函数
        batch = batch_obs(observations, device=self.device)
        
        return batch
    
    @torch.no_grad()
    def act(
        self,
        batch: Dict[str, torch.Tensor],
        recurrent_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        not_done_masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Dict]:
        """
        执行动作选择（完整的 Habitat 流程）
        
        Args:
            batch: 观测批次
            recurrent_hidden_states: RNN 隐藏状态
            prev_actions: 前一个动作
            not_done_masks: episode 是否结束的掩码
            deterministic: 是否使用确定性策略
        
        Returns:
            action: 选择的动作索引
            recurrent_hidden_states: 新的 RNN 隐藏状态
            prev_actions: 新的前一个动作
            action_data: 额外的动作数据（值函数、分布等）
        """
        with inference_mode():
            # 调用策略的 act 方法（与 FALCONEvaluator 一致）
            action_data = self.policy.act(
                batch,
                recurrent_hidden_states,
                prev_actions,
                not_done_masks,
                deterministic=deterministic,
            )
            
            # 更新 RNN 隐藏状态
            if action_data.should_inserts is None:
                recurrent_hidden_states = action_data.rnn_hidden_states
                prev_actions.copy_(action_data.actions)
            else:
                # 多智能体情况
                self.policy.update_hidden_state(
                    recurrent_hidden_states, prev_actions, action_data
                )
            
            # 提取动作
            if self.discrete_actions:
                action = action_data.env_actions.cpu().item()
            else:
                action = action_data.env_actions.cpu().numpy()
        
        # 准备返回数据
        extra_data = {
            'value': action_data.values.cpu().item() if action_data.values is not None else None,
            'action_log_probs': action_data.action_log_probs.cpu().item() if action_data.action_log_probs is not None else None,
        }
        
        return action, recurrent_hidden_states, prev_actions, extra_data
    
    def step(
        self,
        depth: np.ndarray,
        pointgoal: np.ndarray,
        recurrent_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        not_done_masks: torch.Tensor,
        deterministic: bool = False,  # 改为 False，与原始推理一致
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Dict]:
        """
        单步推理（简化接口）
        
        Args:
            depth: (H, W, C) 深度图
            pointgoal: (2,) 目标点 [rho, phi]
            recurrent_hidden_states: RNN 隐藏状态
            prev_actions: 前一个动作
            not_done_masks: episode 状态掩码
            deterministic: 是否使用确定性策略
        
        Returns:
            action: 动作索引
            recurrent_hidden_states: 新的 RNN 隐藏状态
            prev_actions: 新的前一个动作
            extra_data: 额外数据（值函数等）
        """
        # 准备批次
        batch = self.prepare_batch(depth, pointgoal)
        
        # 执行动作选择
        return self.act(
            batch,
            recurrent_hidden_states,
            prev_actions,
            not_done_masks,
            deterministic=deterministic,
        )

def demo_usage():
    # 创建 image 文件夹
    image_dir = "image"
    os.makedirs(image_dir, exist_ok=True)
    """演示如何使用 FalconInferenceEngine"""
    import warnings
    warnings.filterwarnings('ignore')
    
    print("="*80)
    print("Falcon Habitat 风格推理演示")
    print("="*80)
    
    # 1. 创建推理引擎
    engine = FalconInferenceEngine(
        checkpoint_path='pretrained_model/falcon_noaux_25.pth',
        device='cpu',
        use_multi_agent=True,
    )
    
    # 2. 重置状态
    print("\n" + "="*80)
    print("重置状态")
    print("="*80)
    rnn_states, prev_actions, not_done_masks = engine.reset()
    print(f"RNN states shape: {rnn_states.shape}")
    print(f"Prev actions shape: {prev_actions.shape}")
    print(f"Not done masks shape: {not_done_masks.shape}")
    
    # 3. 模拟一个 episode
    print("\n" + "="*80)
    print("模拟导航 episode")
    print("="*80)
    
    action_names = ['STOP', 'FORWARD', 'LEFT', 'RIGHT']
    
    # 模拟轨迹
    goals = [
        (3.0, 0),
        (9.0, 0.4),
        (8.0, 0.3),
        (7.0, 0.2),
        (6.0, 0.1),
        (5.0, 0.0),
        (4.0, -0.1),
        (3.0, 0.0),
        (2.0, 0.0),
        (1.0, 0.0),
    ]
  
    for step, (rho, phi) in enumerate(goals):
        # 创建观测
        depth = np.random.randn(256, 256, 1).astype(np.float32)*0.5 + 0.5  # 模拟深度图
        pointgoal = np.array([rho, phi], dtype=np.float32)
        
 
        # 将深度图归一化到 0-255 范围并转换为 uint8 格式
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        image_path = os.path.join(image_dir, f"{step}.png")
        cv2.imwrite(image_path, depth_normalized)
        print(f"Saved depth image to {image_path}")
    
        # 执行单步推理
        action, rnn_states, prev_actions, extra_data = engine.step(
            depth=depth,
            pointgoal=pointgoal,
            recurrent_hidden_states=rnn_states,
            prev_actions=prev_actions,
            not_done_masks=not_done_masks,
            deterministic=False,  # 使用随机策略（与原始推理一致）
        )
        
        # 打印结果
        print(f"Step {step}: goal=({rho:.1f}, {phi:.1f}), "
              f"action={action_names[action]}, "
              f"value={extra_data['value']:.4f}")
        
        # 模拟环境响应
        done = (action == 0)  # STOP 动作表示结束
        not_done_masks.fill_(not done)
        
        if done:
            print(f"\nEpisode ended at step {step}")
            break
    
    print("\n" + "="*80)
    print("演示完成")
    print("="*80)


if __name__ == "__main__":
    demo_usage()
