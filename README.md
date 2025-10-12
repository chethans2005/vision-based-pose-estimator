# Vision-Based Multi-Agent Pose Estimator

A comprehensive computer vision system for estimating precise poses of multiple autonomous agents (drones) using image and video data. This project demonstrates the integration of computer vision and machine learning to support formation flying and navigation in multi-agent systems.

## 🚀 Features

### Core Capabilities
- **Single-Agent Pose Estimation**: Precise 6DOF pose estimation from monocular images
- **Multi-Agent Pose Estimation**: Simultaneous pose estimation for multiple agents
- **Formation Flying Support**: Specialized metrics and visualization for formation patterns
- **Real-Time Inference**: Live video stream processing with real-time visualization
- **Advanced Computer Vision**: Attention mechanisms, specialized pose heads, and data augmentation

### Model Architectures
- **SimplePoseNet**: Lightweight CNN for basic pose estimation
- **AdvancedPoseNet**: Enhanced model with attention mechanisms and specialized pose regression
- **MultiAgentPoseNet**: Multi-agent model with agent detection and formation analysis

### Evaluation & Visualization
- **Comprehensive Metrics**: ATE, RPE, formation metrics, and agent detection accuracy
- **Advanced Visualizations**: 2D/3D trajectories, error analysis, formation patterns
- **Real-Time Monitoring**: Live trajectory plotting and error tracking

## 📁 Project Structure

```
pose_train/
├── model.py                 # Model architectures (Simple, Advanced, Multi-Agent)
├── dataset.py              # Dataset loader with augmentation support
├── train.py                # Basic training script
├── train_advanced.py       # Advanced training with comprehensive features
├── evaluation.py           # Evaluation metrics and reporting
├── visualization.py        # Visualization tools and plotting
├── realtime_inference.py   # Real-time inference and live visualization
├── eval.py                 # Basic evaluation utilities
├── utils.py                # Utility functions
├── keyframe_extract.py     # Keyframe extraction utilities
└── feature_extract.py      # Feature extraction utilities

examples/
└── complete_usage_example.py  # Comprehensive usage demonstration

notebooks/
├── train_and_eval.ipynb    # Basic training and evaluation notebook
└── complete_demo.ipynb     # Complete feature demonstration notebook
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- OpenCV
- NumPy, Pandas, Matplotlib
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Basic Training
```bash
# Train a simple model
python -m pose_train.train --data-root . --cams cam0 --epochs 20 --work-dir work

# Train with data augmentation
python -m pose_train.train --data-root . --cams cam0 --epochs 20 --augment --work-dir work_aug
```

### 2. Advanced Training
```bash
# Train advanced model with attention
python pose_train/train_advanced.py \
    --data-root . \
    --cams cam0 cam1 \
    --model-type advanced \
    --backbone resnet18 \
    --pretrained \
    --use-attention \
    --augment \
    --epochs 50 \
    --batch-size 16 \
    --work-dir work_advanced

# Train multi-agent model
python pose_train/train_advanced.py \
    --data-root . \
    --cams cam0 \
    --model-type multi_agent \
    --backbone resnet18 \
    --pretrained \
    --use-attention \
    --max-agents 5 \
    --epochs 50 \
    --work-dir work_multi_agent
```

### 3. Complete Usage Example
```bash
# Run comprehensive demonstration
python examples/complete_usage_example.py
```

### 4. Real-Time Inference
```bash
# Run real-time inference with webcam
python pose_train/realtime_inference.py

# Run with specific model
python -c "
from pose_train.realtime_inference import create_realtime_demo
create_realtime_demo('work_advanced/best_model.pt', 'advanced', 0)
"
```

## 📊 Usage Examples

### Single-Agent Pose Estimation

```python
import torch
from pose_train.model import AdvancedPoseNet
from pose_train.dataset import MAVPoseDataset
from pose_train.visualization import plot_trajectory_2d, plot_error_analysis

# Load dataset
dataset = MAVPoseDataset('.', cams=['cam0'], augment=True)

# Create model
model = AdvancedPoseNet(backbone='resnet18', pretrained=True, use_attention=True)
model.load_state_dict(torch.load('work_advanced/best_model.pt')['model_state_dict'])
model.eval()

# Predict poses
with torch.no_grad():
    img, gt_pose = dataset[0]
    pred_pose = model(img.unsqueeze(0))
    
    print(f"Ground truth: {gt_pose.numpy()}")
    print(f"Predicted: {pred_pose.numpy()}")

# Visualize results
plot_trajectory_2d(gt_poses, pred_poses, title="Pose Estimation Results")
plot_error_analysis(gt_poses, pred_poses, title="Error Analysis")
```

### Multi-Agent Pose Estimation

```python
from pose_train.model import MultiAgentPoseNet
from pose_train.visualization import plot_multi_agent_trajectory, plot_formation_analysis

# Create multi-agent model
model = MultiAgentPoseNet(
    backbone='resnet18', 
    pretrained=True, 
    max_agents=5, 
    use_attention=True
)

# Predict poses for multiple agents
with torch.no_grad():
    img, gt_pose = dataset[0]
    agent_poses, agent_presence = model(img.unsqueeze(0))
    
    print(f"Agent poses shape: {agent_poses.shape}")
    print(f"Agent presence: {agent_presence}")

# Visualize multi-agent trajectories
agent_poses_dict = {i: agent_poses[0, i].numpy() for i in range(5)}
agent_presence_dict = {i: agent_presence[0, i].numpy() for i in range(5)}

plot_multi_agent_trajectory(agent_poses_dict, agent_presence_dict)
plot_formation_analysis(agent_poses_dict, agent_presence_dict)
```

## 🔧 Advanced Configuration

### Model Configuration

```python
# Advanced model with custom configuration
model = AdvancedPoseNet(
    in_channels=1,           # Number of input channels
    backbone='resnet50',     # Backbone architecture
    pretrained=True,         # Use pretrained weights
    use_attention=True       # Enable attention mechanisms
)

# Multi-agent model
model = MultiAgentPoseNet(
    in_channels=1,
    backbone='resnet18',
    pretrained=True,
    max_agents=10,           # Maximum number of agents
    use_attention=True
)
```

### Training Configuration

```python
# Advanced training with custom parameters
python pose_train/train_advanced.py \
    --data-root /path/to/dataset \
    --cams cam0 cam1 cam2 \
    --model-type advanced \
    --backbone resnet50 \
    --pretrained \
    --use-attention \
    --augment \
    --use-imagenet-norm \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --grad-clip 1.0 \
    --early-stopping 15 \
    --work-dir work_custom
```

## 📈 Performance Metrics

### Single-Agent Metrics
- **Translation RMSE**: Root mean square error for position
- **Rotation RMSE**: Root mean square error for orientation
- **ATE (Absolute Trajectory Error)**: Overall trajectory accuracy
- **RPE (Relative Pose Error)**: Relative pose accuracy

### Multi-Agent Metrics
- **Agent Detection**: Precision, recall, F1-score for agent presence
- **Formation Center Error**: Accuracy of formation center estimation
- **Formation Shape Preservation**: Consistency of inter-agent distances
- **Formation Stability**: Variance in formation patterns

### Example Results
```
Translation RMSE: 0.198 m
Rotation RMSE: 3.37°
Detection Precision: 0.95
Detection Recall: 0.92
Formation Center RMSE: 0.15 m
Formation Shape RMSE: 0.08 m
```

## 🎯 Real-Time Applications

### Live Video Processing
```python
from pose_train.realtime_inference import RealTimePoseEstimator, RealTimeVisualizer

# Create estimator
estimator = RealTimePoseEstimator(
    model_path='work_advanced/best_model.pt',
    model_type='advanced',
    device='cuda'
)

# Start real-time processing
estimator.start_realtime_processing(video_source=0)  # 0 for webcam
```

### Formation Flying Demo
```python
# Multi-agent real-time processing
estimator = RealTimePoseEstimator(
    model_path='work_multi_agent/best_model.pt',
    model_type='multi_agent',
    max_agents=5
)

# Start with live visualization
visualizer = RealTimeVisualizer(estimator)
visualizer.start_live_plot()

estimator.start_realtime_processing(video_source=0)
```

## 🔬 Research Applications

### Keyframe Extraction
```bash
# Extract keyframes using interval method
python pose_train/keyframe_extract.py \
    --img-dir cam0/data \
    --out-dir cam0_keyframes \
    --mode interval \
    --interval 10

# Extract keyframes using ORB-based scene change detection
python pose_train/keyframe_extract.py \
    --img-dir cam0/data \
    --out-dir cam0_keyframes \
    --mode orb_change \
    --threshold 30
```

### Feature Extraction
```bash
# Extract ORB features
python pose_train/feature_extract.py \
    --img-dir cam0_keyframes \
    --out-dir cam0_features \
    --max-features 1000
```

### Model Comparison
```bash
# Compare different models
python pose_train/compare_models.py \
    --data-root . \
    --cams cam0 \
    --epochs 5 \
    --work-dir work_compare \
    --augment \
    --use-imagenet-norm
```

## 📚 Documentation

- **[Advanced README](README_ADVANCED.md)**: Comprehensive documentation with detailed API reference
- **[Complete Usage Example](examples/complete_usage_example.py)**: Full demonstration script
- **[Jupyter Notebooks](notebooks/)**: Interactive tutorials and examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- OpenCV team for computer vision utilities
- The computer vision and robotics research community
- Contributors and users of this project

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers
- Check the documentation and examples

---

**Vision-Based Multi-Agent Pose Estimator** - Enabling precise formation flying through advanced computer vision and machine learning.