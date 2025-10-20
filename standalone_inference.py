#!/usr/bin/env python3
"""
Standalone Falcon Model Inference Script

This script provides a minimal, independent inference interface for the Falcon navigation model
without requiring the full habitat-baselines infrastructure.

Usage:
    python standalone_inference.py --model-path falcon_noaux_25.pth

The script loads the pre-trained Falcon model and provides a simple inference interface.
"""

import argparse
import math
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import gymnasium as gym
    spaces = gym.spaces
except ImportError:
    import gym
    spaces = gym.spaces


# ====== Minimal ResNet Implementation ======

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, baseplanes=32, ngroups=32, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        self._norm_layer = norm_layer

        self.inplanes = baseplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)  # Assuming depth input (1 channel)
        self.bn1 = norm_layer(ngroups, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, baseplanes, layers[0], norm_layer=lambda x: norm_layer(ngroups, x))
        self.layer2 = self._make_layer(block, baseplanes * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], norm_layer=lambda x: norm_layer(ngroups, x))
        self.layer3 = self._make_layer(block, baseplanes * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], norm_layer=lambda x: norm_layer(ngroups, x))
        self.layer4 = self._make_layer(block, baseplanes * 8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], norm_layer=lambda x: norm_layer(ngroups, x))

        self.final_channels = baseplanes * 8 * block.expansion
        self.final_spatial_compress = 1.0 / 32.0  # Due to the stride reductions: 7x7 conv stride 2, maxpool stride 2, then 3 layers with stride 2 = 2^5 = 32

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, norm_layer=None):
        if norm_layer is None:
            norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50(input_channels, baseplanes=32, ngroups=32):
    model = ResNet(Bottleneck, [3, 4, 6, 3], baseplanes=baseplanes, ngroups=ngroups)
    # Modify the first conv layer for the correct input channels
    model.conv1 = nn.Conv2d(input_channels, baseplanes, kernel_size=7, stride=2, padding=3, bias=False)
    return model


# ====== RNN State Encoder ======

class RNNStateEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type="LSTM", num_layers=1):
        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type
        self._hidden_size = hidden_size

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers

    def forward(self, x, hidden_states, masks):
        if self._rnn_type == "LSTM":
            # LSTM expects (h_n, c_n) tuple
            if isinstance(hidden_states, torch.Tensor):
                # Convert single tensor to tuple for LSTM
                h_n = hidden_states
                c_n = torch.zeros_like(h_n)
                hidden_states = (h_n, c_n)
            x, hidden_states = self.rnn(x, hidden_states)
            # Return only h_n for consistency
            return x, hidden_states[0]
        else:  # GRU
            x, hidden_states = self.rnn(x, hidden_states)
            return x, hidden_states


# ====== ResNet Encoder ======

class ResNetEncoder(nn.Module):
    def __init__(self, observation_space, baseplanes=32, ngroups=32, make_backbone=None, normalize_visual_inputs=False):
        super().__init__()
        
        # Determine visual observation keys (assuming depth input)
        self.visual_keys = [k for k, v in observation_space.spaces.items() if len(v.shape) > 1]
        
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )
        
        if not self.is_blind:
            # The actual input size to the backbone after preprocessing
            # Input images are 256x256, but ResNet processes them at full resolution
            # The actual compression comes from the ResNet layers
            # With stride 2 in conv1, maxpool stride 2, and layer2,3,4 each having stride 2
            # Total reduction is 2 * 2 * 2 * 2 * 2 = 32, so 256/32 = 8
            actual_spatial_h = 8
            actual_spatial_w = 8
            
            self.backbone = make_backbone(self._n_input_channels, baseplanes, ngroups)
            
            after_compression_flat_size = 2048
            num_compression_channels = int(round(after_compression_flat_size / (actual_spatial_h * actual_spatial_w)))
            
            self.compression = nn.Sequential(
                nn.Conv2d(self.backbone.final_channels, num_compression_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )
            
            # Calculate actual output size - this is what gets flattened
            actual_output_size = num_compression_channels * actual_spatial_h * actual_spatial_w
            self.output_shape = (actual_output_size,)

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def forward(self, observations):
        if self.is_blind:
            return None
            
        # Concatenate visual inputs
        visual_feats = []
        for k in self.visual_keys:
            obs = observations[k]
            if obs.dtype == torch.uint8:
                obs = obs.float() / 255.0
            visual_feats.append(obs)
        
        x = torch.cat(visual_feats, dim=-1)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        x = self.backbone(x)
        x = self.compression(x)
        
        return x


# ====== Main Falcon Model ======

class FalconNet(nn.Module):
    """Standalone Falcon Navigation Network"""
    
    def __init__(self, observation_space, action_space, hidden_size=512, num_recurrent_layers=2, 
                 rnn_type="LSTM", backbone="resnet50", resnet_baseplanes=32, normalize_visual_inputs=False):
        super().__init__()
        
        self._hidden_size = hidden_size
        
        # Visual encoder
        if backbone == "resnet50":
            make_backbone = resnet50
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=make_backbone,
            normalize_visual_inputs=normalize_visual_inputs,
        )
        
        # Visual feature projection
        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.visual_encoder.output_shape[0], hidden_size),
                nn.ReLU(True),
            )
        
        # Goal encoding (pointgoal_with_gps_compass)
        rnn_input_size = 0
        if not self.visual_encoder.is_blind:
            rnn_input_size += hidden_size
            
        # Add pointgoal encoding
        self.pointgoal_embedding = nn.Linear(2, 32)  # GPS compass has 2D input
        rnn_input_size += 32
        
        # RNN state encoder
        self.state_encoder = RNNStateEncoder(
            rnn_input_size, hidden_size, rnn_type, num_recurrent_layers
        )
        
        # Action head
        self.action_head = nn.Linear(hidden_size, action_space.n)
        self.value_head = nn.Linear(hidden_size, 1)

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        
        # Process visual features
        if not self.is_blind:
            visual_feats = self.visual_encoder(observations)
            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)
        
        # Process pointgoal
        if "pointgoal_with_gps_compass" in observations:
            pointgoal_embedding = self.pointgoal_embedding(observations["pointgoal_with_gps_compass"])
            x.append(pointgoal_embedding)
        
        # Concatenate all features
        x = torch.cat(x, dim=-1)
        x = x.unsqueeze(1)  # Add time dimension for RNN
        
        # RNN forward pass
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        x = x.squeeze(1)  # Remove time dimension
        
        # Action and value heads
        actions = self.action_head(x)
        values = self.value_head(x)
        
        return actions, values, rnn_hidden_states


class FalconPolicy(nn.Module):
    """Complete Falcon Policy with action distribution"""
    
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__()
        self.net = FalconNet(observation_space, action_space, **kwargs)
        self.action_distribution = torch.distributions.Categorical

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        actions, values, rnn_hidden_states = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return values, actions, rnn_hidden_states

    def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False):
        values, actions_logits, rnn_hidden_states = self.forward(
            observations, rnn_hidden_states, prev_actions, masks
        )
        
        distribution = self.action_distribution(logits=actions_logits)
        
        if deterministic:
            actions = distribution.mode
        else:
            actions = distribution.sample()
        
        return {
            "actions": actions,
            "values": values,
            "action_log_probs": distribution.log_prob(actions),
            "rnn_hidden_states": rnn_hidden_states,
            "distribution": distribution,
        }

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        values, _, _ = self.forward(observations, rnn_hidden_states, prev_actions, masks)
        return values

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, actions):
        values, actions_logits, rnn_hidden_states = self.forward(
            observations, rnn_hidden_states, prev_actions, masks
        )
        
        distribution = self.action_distribution(logits=actions_logits)
        action_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, action_log_probs, entropy, rnn_hidden_states


# ====== Standalone Inference Class ======

class FalconInference:
    """Standalone Falcon Model Inference Interface"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize Falcon model for inference
        
        Args:
            model_path: Path to the falcon_noaux_25.pth checkpoint
            device: Device to run inference on ("auto", "cpu", "cuda")
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Define observation and action spaces based on Falcon configuration
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32),
            "pointgoal_with_gps_compass": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
        })
        
        self.action_space = spaces.Discrete(4)  # [STOP, FORWARD, LEFT, RIGHT]
        
        # Initialize model
        self.model = FalconPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            hidden_size=512,
            num_recurrent_layers=2,
            rnn_type="LSTM",
            backbone="resnet50",
            resnet_baseplanes=32,
            normalize_visual_inputs=False,
        )
        
        # Load checkpoint
        self._load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize hidden states
        self.reset()

    def _load_checkpoint(self, model_path: str):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Extract actor_critic weights
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if "actor_critic" in k:
                    # Remove "actor_critic." prefix
                    new_key = k.replace("actor_critic.", "")
                    state_dict[new_key] = v
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded Falcon model from {model_path}")

    def reset(self):
        """Reset hidden states (call at episode start)"""
        self.rnn_hidden_states = torch.zeros(
            self.model.net.num_recurrent_layers, 1, self.model.net.recurrent_hidden_size,
            device=self.device, dtype=torch.float32
        )
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

    def predict(self, depth_image: np.ndarray, pointgoal: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action given observations
        
        Args:
            depth_image: Depth image of shape (H, W) or (H, W, 1), values in [0, 1]
            pointgoal: GPS compass coordinates of shape (2,) - [distance, angle]
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Integer action [0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT]
        """
        # Prepare observations
        if depth_image.ndim == 2:
            depth_image = depth_image[..., None]  # Add channel dimension
        
        # Ensure correct shape and type
        if depth_image.shape != (256, 256, 1):
            # You might want to resize here if needed
            print(f"Warning: Expected depth image shape (256, 256, 1), got {depth_image.shape}")
        
        observations = {
            "depth": torch.from_numpy(depth_image).float().unsqueeze(0).to(self.device),
            "pointgoal_with_gps_compass": torch.from_numpy(pointgoal).float().unsqueeze(0).to(self.device),
        }
        
        with torch.no_grad():
            action_data = self.model.act(
                observations,
                self.rnn_hidden_states,
                self.prev_actions,
                self.masks,
                deterministic=deterministic
            )
            
            # Update states
            self.rnn_hidden_states = action_data["rnn_hidden_states"]
            self.prev_actions = action_data["actions"].unsqueeze(1)
            # Keep masks as True (not done)
        
        return action_data["actions"].item()

    def get_action_name(self, action: int) -> str:
        """Convert action integer to human-readable name"""
        action_names = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        return action_names[action] if 0 <= action < len(action_names) else "UNKNOWN"


# ====== Main Script ======

def main():
    parser = argparse.ArgumentParser(description="Falcon Standalone Inference")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to falcon_noaux_25.pth checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for inference")
    
    args = parser.parse_args()
    
    try:
        # Initialize Falcon inference
        falcon = FalconInference(args.model_path, args.device)
        print(f"Falcon model loaded successfully on {falcon.device}")
        print(f"Model architecture: {falcon.model}")
        
        # Example usage
        print("\n=== Example Usage ===")
        
        # Create dummy inputs
        depth_image = np.random.rand(256, 256, 1).astype(np.float32)
        pointgoal = np.array([5.0, 0.5], dtype=np.float32)  # 5 meters away, 0.5 radians
        
        print(f"Input depth image shape: {depth_image.shape}")
        print(f"Input pointgoal: {pointgoal}")
        
        # Reset episode
        falcon.reset()
        
        # Predict action
        action = falcon.predict(depth_image, pointgoal, deterministic=True)
        action_name = falcon.get_action_name(action)
        
        print(f"Predicted action: {action} ({action_name})")
        
        print("\n=== Integration Example ===")
        print("To use this in your code:")
        print("1. falcon = FalconInference('falcon_noaux_25.pth')")
        print("2. falcon.reset()  # At episode start")
        print("3. action = falcon.predict(depth_image, pointgoal)")
        print("4. # Execute action in your environment")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()