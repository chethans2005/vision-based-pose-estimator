"""
Real-time inference module for live video streams.
Supports both single-agent and multi-agent pose estimation.
"""
import cv2
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Union
import threading
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class RealTimePoseEstimator:
    """
    Real-time pose estimation from video streams.
    """
    
    def __init__(self, model_path: str, model_type: str = 'simple', 
                 device: str = 'auto', max_agents: int = 5):
        """
        Initialize the real-time pose estimator.
        
        Args:
            model_path: Path to the trained model checkpoint
            model_type: Type of model ('simple', 'advanced', 'multi_agent')
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            max_agents: Maximum number of agents for multi-agent models
        """
        self.model_type = model_type
        self.max_agents = max_agents
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize tracking variables
        self.pose_history = []
        self.presence_history = []
        self.timestamps = []
        
        # Threading for real-time processing
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.running = False
        self.processing_thread = None
        
    def _load_model(self, model_path: str):
        """Load the trained model."""
        from pose_train.model import SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if self.model_type == 'simple':
            model = SimplePoseNet()
        elif self.model_type == 'advanced':
            model = AdvancedPoseNet(use_attention=True)
        elif self.model_type == 'multi_agent':
            model = MultiAgentPoseNet(max_agents=self.max_agents, use_attention=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model.load_state_dict(checkpoint['model_state'])
        return model.to(self.device)
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (240, 320)) -> torch.Tensor:
        """
        Preprocess a video frame for inference.
        
        Args:
            frame: Input frame (BGR)
            target_size: Target size (height, width)
            
        Returns:
            Preprocessed tensor
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize
        frame = cv2.resize(frame, (target_size[1], target_size[0]))
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return frame_tensor.to(self.device)
    
    def predict_pose(self, frame: np.ndarray) -> Union[Dict, Tuple[Dict, Dict]]:
        """
        Predict pose from a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            For single-agent: Dictionary with pose information
            For multi-agent: Tuple of (poses, presence) dictionaries
        """
        with torch.no_grad():
            # Preprocess frame
            frame_tensor = self.preprocess_frame(frame)
            
            if self.model_type == 'multi_agent':
                # Multi-agent prediction
                agent_poses, agent_presence = self.model(frame_tensor)
                
                poses_dict = {}
                presence_dict = {}
                
                for i in range(self.max_agents):
                    pose = agent_poses[0, i].cpu().numpy()  # [7]
                    presence = agent_presence[0, i].cpu().numpy()
                    
                    poses_dict[i] = {
                        'translation': pose[:3],
                        'rotation': pose[3:7],
                        'quaternion': pose[3:7]
                    }
                    presence_dict[i] = float(presence)
                
                return poses_dict, presence_dict
            else:
                # Single-agent prediction
                pose = self.model(frame_tensor).cpu().numpy()[0]  # [7]
                
                return {
                    'translation': pose[:3],
                    'rotation': pose[3:7],
                    'quaternion': pose[3:7]
                }
    
    def _processing_loop(self):
        """Main processing loop for real-time inference."""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame, timestamp = self.frame_queue.get(timeout=0.1)
                    
                    # Predict pose
                    if self.model_type == 'multi_agent':
                        poses, presence = self.predict_pose(frame)
                        result = {'poses': poses, 'presence': presence, 'timestamp': timestamp}
                    else:
                        pose = self.predict_pose(frame)
                        result = {'pose': pose, 'timestamp': timestamp}
                    
                    # Add to result queue
                    if not self.result_queue.full():
                        self.result_queue.put(result)
                    
                    # Update history
                    self._update_history(result)
                    
            except:
                continue
    
    def _update_history(self, result: Dict):
        """Update pose and presence history."""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        if self.model_type == 'multi_agent':
            self.pose_history.append(result['poses'])
            self.presence_history.append(result['presence'])
        else:
            self.pose_history.append(result['pose'])
        
        # Keep only last 1000 frames
        if len(self.pose_history) > 1000:
            self.pose_history.pop(0)
            self.presence_history.pop(0) if self.model_type == 'multi_agent' else None
            self.timestamps.pop(0)
    
    def start_realtime_processing(self, video_source: Union[int, str] = 0):
        """
        Start real-time processing from video source.
        
        Args:
            video_source: Video source (0 for webcam, path for video file)
        """
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        # Open video capture
        self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        print("Real-time processing started. Press 'q' to quit, 's' to save trajectory.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Add frame to processing queue
            if not self.frame_queue.full():
                self.frame_queue.put((frame, time.time()))
            
            # Display frame with pose overlay
            display_frame = self._overlay_poses(frame)
            cv2.imshow('Real-time Pose Estimation', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_trajectory()
        
        self.stop_realtime_processing()
    
    def _overlay_poses(self, frame: np.ndarray) -> np.ndarray:
        """Overlay pose information on frame."""
        display_frame = frame.copy()
        
        # Get latest result
        if not self.result_queue.empty():
            result = self.result_queue.get()
            
            if self.model_type == 'multi_agent':
                poses = result['poses']
                presence = result['presence']
                
                # Draw poses for each agent
                for agent_id, pose_info in poses.items():
                    if presence[agent_id] > 0.5:  # Only draw if agent is present
                        trans = pose_info['translation']
                        text = f"Agent {agent_id}: ({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f})"
                        cv2.putText(display_frame, text, (10, 30 + agent_id * 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                pose_info = result['pose']
                trans = pose_info['translation']
                text = f"Pose: ({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f})"
                cv2.putText(display_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return display_frame
    
    def _save_trajectory(self):
        """Save current trajectory to file."""
        if not self.pose_history:
            print("No trajectory data to save.")
            return
        
        timestamp = int(time.time())
        filename = f"trajectory_{timestamp}.csv"
        
        with open(filename, 'w') as f:
            f.write("timestamp,tx,ty,tz,qw,qx,qy,qz\n")
            
            for i, pose in enumerate(self.pose_history):
                if self.model_type == 'multi_agent':
                    # Save first agent's pose
                    if 0 in pose:
                        p = pose[0]
                        f.write(f"{self.timestamps[i]},{p['translation'][0]},{p['translation'][1]},{p['translation'][2]},"
                               f"{p['quaternion'][0]},{p['quaternion'][1]},{p['quaternion'][2]},{p['quaternion'][3]}\n")
                else:
                    f.write(f"{self.timestamps[i]},{pose['translation'][0]},{pose['translation'][1]},{pose['translation'][2]},"
                           f"{pose['quaternion'][0]},{pose['quaternion'][1]},{pose['quaternion'][2]},{pose['quaternion'][3]}\n")
        
        print(f"Trajectory saved to {filename}")
    
    def stop_realtime_processing(self):
        """Stop real-time processing."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
    
    def get_latest_pose(self) -> Optional[Dict]:
        """Get the latest pose prediction."""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
    
    def get_trajectory_data(self) -> Tuple[List, List]:
        """Get trajectory data for visualization."""
        return self.pose_history, self.timestamps


class RealTimeVisualizer:
    """
    Real-time visualization for pose estimation results.
    """
    
    def __init__(self, estimator: RealTimePoseEstimator):
        self.estimator = estimator
        self.fig = None
        self.ax = None
        self.animation = None
        
    def start_live_plot(self, update_interval: int = 100):
        """
        Start live plotting of pose trajectories.
        
        Args:
            update_interval: Update interval in milliseconds
        """
        self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))
        self.ax = self.ax.flatten()
        
        # Initialize plots
        self.ax[0].set_title('XY Trajectory')
        self.ax[0].set_xlabel('X (m)')
        self.ax[0].set_ylabel('Y (m)')
        self.ax[0].grid(True)
        
        self.ax[1].set_title('XZ Trajectory')
        self.ax[1].set_xlabel('X (m)')
        self.ax[1].set_ylabel('Z (m)')
        self.ax[1].grid(True)
        
        self.ax[2].set_title('Translation Error Over Time')
        self.ax[2].set_xlabel('Time (s)')
        self.ax[2].set_ylabel('Error (m)')
        self.ax[2].grid(True)
        
        self.ax[3].set_title('Rotation Error Over Time')
        self.ax[3].set_xlabel('Time (s)')
        self.ax[3].set_ylabel('Error (deg)')
        self.ax[3].grid(True)
        
        # Start animation
        self.animation = FuncAnimation(self.fig, self._update_plot, 
                                     interval=update_interval, blit=False)
        plt.show()
    
    def _update_plot(self, frame):
        """Update the live plot."""
        poses, timestamps = self.estimator.get_trajectory_data()
        
        if not poses:
            return
        
        # Clear previous plots
        for ax in self.ax:
            ax.clear()
        
        if self.estimator.model_type == 'multi_agent':
            # Multi-agent visualization
            colors = plt.cm.tab10(np.linspace(0, 1, self.estimator.max_agents))
            
            for agent_id in range(self.estimator.max_agents):
                x_coords = []
                y_coords = []
                z_coords = []
                
                for pose in poses:
                    if agent_id in pose:
                        trans = pose[agent_id]['translation']
                        x_coords.append(trans[0])
                        y_coords.append(trans[1])
                        z_coords.append(trans[2])
                
                if x_coords:
                    self.ax[0].plot(x_coords, y_coords, color=colors[agent_id], 
                                   label=f'Agent {agent_id}', linewidth=2)
                    self.ax[1].plot(x_coords, z_coords, color=colors[agent_id], 
                                   label=f'Agent {agent_id}', linewidth=2)
        else:
            # Single-agent visualization
            x_coords = [pose['translation'][0] for pose in poses]
            y_coords = [pose['translation'][1] for pose in poses]
            z_coords = [pose['translation'][2] for pose in poses]
            
            self.ax[0].plot(x_coords, y_coords, 'b-', linewidth=2, label='Trajectory')
            self.ax[1].plot(x_coords, z_coords, 'b-', linewidth=2, label='Trajectory')
        
        # Set plot properties
        self.ax[0].set_title('XY Trajectory')
        self.ax[0].set_xlabel('X (m)')
        self.ax[0].set_ylabel('Y (m)')
        self.ax[0].grid(True)
        self.ax[0].legend()
        self.ax[0].axis('equal')
        
        self.ax[1].set_title('XZ Trajectory')
        self.ax[1].set_xlabel('X (m)')
        self.ax[1].set_ylabel('Z (m)')
        self.ax[1].grid(True)
        self.ax[1].legend()
        self.ax[1].axis('equal')
        
        # Error plots (placeholder - would need ground truth for real errors)
        self.ax[2].plot(timestamps, [0] * len(timestamps), 'r-', linewidth=1)
        self.ax[2].set_title('Translation Error Over Time')
        self.ax[2].set_xlabel('Time (s)')
        self.ax[2].set_ylabel('Error (m)')
        self.ax[2].grid(True)
        
        self.ax[3].plot(timestamps, [0] * len(timestamps), 'r-', linewidth=1)
        self.ax[3].set_title('Rotation Error Over Time')
        self.ax[3].set_xlabel('Time (s)')
        self.ax[3].set_ylabel('Error (deg)')
        self.ax[3].grid(True)
    
    def stop_live_plot(self):
        """Stop the live plot."""
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)


def create_realtime_demo(model_path: str, model_type: str = 'simple', 
                        video_source: Union[int, str] = 0):
    """
    Create a real-time pose estimation demo.
    
    Args:
        model_path: Path to trained model
        model_type: Type of model to use
        video_source: Video source for real-time processing
    """
    # Create estimator
    estimator = RealTimePoseEstimator(model_path, model_type)
    
    # Create visualizer
    visualizer = RealTimeVisualizer(estimator)
    
    # Start live plot in separate thread
    plot_thread = threading.Thread(target=visualizer.start_live_plot)
    plot_thread.daemon = True
    plot_thread.start()
    
    try:
        # Start real-time processing
        estimator.start_realtime_processing(video_source)
    except KeyboardInterrupt:
        print("Demo stopped by user.")
    finally:
        estimator.stop_realtime_processing()
        visualizer.stop_live_plot()


if __name__ == "__main__":
    # Example usage
    model_path = "work_compare_aug/best_ckpt.pt"  # Update with your model path
    create_realtime_demo(model_path, model_type='advanced', video_source=0)
