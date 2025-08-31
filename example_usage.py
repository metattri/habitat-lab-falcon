#!/usr/bin/env python3
"""
Example usage of the Falcon standalone inference script.

This demonstrates how to use the extracted Falcon model for navigation inference
without the habitat-baselines infrastructure.
"""

import numpy as np
from standalone_inference import FalconInference

def example_usage():
    """
    Example showing how to use Falcon for navigation inference
    """
    
    print("=== Falcon Standalone Inference Example ===")
    
    # Step 1: Initialize the model
    # Replace with your actual path to falcon_noaux_25.pth
    model_path = "falcon_noaux_25.pth"  # You need to download this
    
    try:
        falcon = FalconInference(model_path, device="auto")
        print(f"✓ Falcon model loaded successfully on {falcon.device}")
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        print("Please download falcon_noaux_25.pth from the provided link")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Step 2: Simulate an episode
    print("\n=== Simulating Navigation Episode ===")
    
    # Reset at episode start
    falcon.reset()
    
    # Simulate navigation steps
    num_steps = 10
    
    for step in range(num_steps):
        # Step 3: Generate example observations
        # In real usage, these would come from your environment/sensors
        
        # Depth image: 256x256 normalized depth values [0,1]
        # Here we create a simple synthetic depth image
        depth_image = create_synthetic_depth_image()
        
        # Point goal: [distance_to_goal, angle_to_goal_in_radians]
        # Distance decreases over time, angle varies
        distance = max(1.0, 10.0 - step * 0.5)  # Getting closer to goal
        angle = np.sin(step * 0.3) * 0.5  # Some angular variation
        pointgoal = np.array([distance, angle], dtype=np.float32)
        
        # Step 4: Predict action
        action = falcon.predict(depth_image, pointgoal, deterministic=True)
        action_name = falcon.get_action_name(action)
        
        # Display results
        print(f"Step {step + 1:2d}: Goal=({distance:.1f}m, {angle:.2f}rad) → Action: {action} ({action_name})")
        
        # In real usage, you would:
        # obs, reward, done, info = env.step(action)
        # if done:
        #     break
    
    print("\n=== Episode Completed ===")


def create_synthetic_depth_image():
    """
    Create a synthetic depth image for demonstration.
    In real usage, this would come from your depth sensor.
    """
    # Create a synthetic depth image with some structure
    depth = np.ones((256, 256, 1), dtype=np.float32) * 0.8  # Base distance
    
    # Add some obstacles (closer objects)
    x, y = np.meshgrid(np.arange(256), np.arange(256))
    
    # Circular obstacle
    obstacle1 = ((x - 128)**2 + (y - 100)**2) < 20**2
    depth[obstacle1] = 0.3
    
    # Wall on the left
    depth[:, :50] = 0.2
    
    # Add some noise
    depth += np.random.normal(0, 0.05, depth.shape)
    depth = np.clip(depth, 0, 1)
    
    return depth


def integration_example():
    """
    Example showing how to integrate with your own environment
    """
    print("\n=== Integration Pattern ===")
    print("""
    # Your environment integration pattern:
    
    from standalone_inference import FalconInference
    
    # Initialize model once
    falcon = FalconInference('falcon_noaux_25.pth')
    
    # Episode loop
    for episode in range(num_episodes):
        falcon.reset()  # Reset hidden states for new episode
        obs = your_env.reset()
        done = False
        
        while not done:
            # Extract observations in the required format
            depth = obs['depth']  # Shape: (256, 256, 1), dtype: float32, range: [0,1]
            pointgoal = obs['pointgoal']  # Shape: (2,), [distance, angle_in_radians]
            
            # Get action prediction
            action = falcon.predict(depth, pointgoal)
            
            # Execute action in your environment
            obs, reward, done, info = your_env.step(action)
            
            # Optional: Convert action to your environment's action space
            # your_action = convert_falcon_action(action)
    """)


def convert_falcon_action_example():
    """
    Example of how to convert Falcon actions to your environment's action space
    """
    print("\n=== Action Conversion Example ===")
    
    def convert_falcon_to_custom_actions(falcon_action):
        """
        Convert Falcon's discrete actions to your environment's action format
        """
        action_mapping = {
            0: "stop",           # STOP
            1: "move_forward",   # MOVE_FORWARD  
            2: "turn_left",      # TURN_LEFT
            3: "turn_right"      # TURN_RIGHT
        }
        return action_mapping.get(falcon_action, "stop")
    
    def convert_falcon_to_continuous_actions(falcon_action):
        """
        Convert Falcon's discrete actions to continuous control
        """
        # Example: convert to [linear_velocity, angular_velocity]
        action_mapping = {
            0: [0.0, 0.0],      # STOP
            1: [0.25, 0.0],     # MOVE_FORWARD
            2: [0.0, 0.1],      # TURN_LEFT (positive angular velocity)
            3: [0.0, -0.1],     # TURN_RIGHT (negative angular velocity)
        }
        return np.array(action_mapping.get(falcon_action, [0.0, 0.0]))
    
    # Demo
    for action in range(4):
        action_name = FalconInference.get_action_name(None, action)
        custom_action = convert_falcon_to_custom_actions(action)
        continuous_action = convert_falcon_to_continuous_actions(action)
        print(f"Falcon {action} ({action_name}) → '{custom_action}' or {continuous_action}")


if __name__ == "__main__":
    example_usage()
    integration_example()
    convert_falcon_action_example()
    
    print("\n=== Next Steps ===")
    print("1. Download falcon_noaux_25.pth from the provided link")
    print("2. Adapt the observation preprocessing to your environment")
    print("3. Integrate the action predictions into your control loop")
    print("4. Test with real depth sensor data")