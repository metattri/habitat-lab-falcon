# Falcon Standalone Inference

This directory contains a standalone inference script for the Falcon navigation model that doesn't require the full habitat-baselines infrastructure.

## Overview

The Falcon model is a deep reinforcement learning policy for social navigation that uses:
- **Visual Input**: Depth images (256x256x1)
- **Goal Input**: GPS compass coordinates (distance, angle)
- **Architecture**: ResNet50 visual encoder + LSTM + action/value heads
- **Actions**: 4 discrete actions [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]

## Files

- `standalone_inference.py` - Complete standalone inference implementation
- `README.md` - This documentation

## Requirements

```bash
pip install torch torchvision numpy gymnasium
```

## Usage

### Quick Start

```python
from standalone_inference import FalconInference
import numpy as np

# Initialize the model
falcon = FalconInference('falcon_noaux_25.pth')

# Reset at episode start
falcon.reset()

# Predict actions
depth_image = np.random.rand(256, 256, 1).astype(np.float32)  # Your depth image
pointgoal = np.array([5.0, 0.5], dtype=np.float32)  # [distance, angle_in_radians]

action = falcon.predict(depth_image, pointgoal, deterministic=True)
action_name = falcon.get_action_name(action)
print(f"Action: {action} ({action_name})")
```

### Command Line Testing

```bash
python standalone_inference.py --model-path falcon_noaux_25.pth --device auto
```

## Model Architecture

The standalone implementation includes:

1. **ResNet50 Visual Encoder**: Processes 256x256 depth images
2. **Goal Embedding**: Encodes GPS compass coordinates 
3. **LSTM**: 2-layer LSTM with 512 hidden units for temporal processing
4. **Action Head**: Outputs logits for 4 discrete actions
5. **Value Head**: Estimates state value (for training, not used in inference)

## Input Specifications

- **Depth Image**: 
  - Shape: (256, 256, 1) or (256, 256)
  - Type: numpy.ndarray with dtype float32
  - Range: [0, 1] (normalized depth values)

- **Point Goal**:
  - Shape: (2,)
  - Type: numpy.ndarray with dtype float32
  - Format: [distance_to_goal, angle_to_goal_in_radians]

## Action Space

- **0**: STOP - Stop moving
- **1**: MOVE_FORWARD - Move forward
- **2**: TURN_LEFT - Turn left  
- **3**: TURN_RIGHT - Turn right

## Key Features

- **No habitat-baselines dependency**: Completely self-contained
- **GPU/CPU support**: Automatic device selection or manual specification
- **State management**: Automatic hidden state tracking across time steps
- **Simple interface**: Just call `predict()` with your observations

## Technical Details

### Model Parameters
- Total parameters: ~12.4M
- Visual encoder: ResNet50 with 32 base planes
- Hidden size: 512
- RNN layers: 2 (LSTM)
- Input compression: 2048 visual features + 32 goal features = 544 RNN input

### Memory Requirements
- Model size: ~50MB
- Forward pass: ~200MB GPU memory for single inference

## Integration Example

```python
# Episode loop
falcon = FalconInference('falcon_noaux_25.pth')

for episode in range(num_episodes):
    falcon.reset()  # Reset hidden states
    obs = env.reset()
    
    while not done:
        # Extract required inputs
        depth = obs['depth']  # Shape: (256, 256, 1)
        pointgoal = obs['pointgoal_with_gps_compass']  # Shape: (2,)
        
        # Predict action
        action = falcon.predict(depth, pointgoal)
        
        # Execute in environment
        obs, reward, done, info = env.step(action)
```

## Differences from Original

This standalone version:
- Removes habitat-baselines dependencies
- Simplifies observation processing
- Provides a clean inference-only interface
- Includes minimal ResNet implementation
- Handles LSTM state management automatically

For training or full habitat integration, use the original habitat-baselines framework.