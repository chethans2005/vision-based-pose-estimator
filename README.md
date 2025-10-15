# Vision-Based Multi-Agent Pose Estimator# Vision-Based Multi-Agent Pose Estimator



A PyTorch-based deep learning system for estimating 6-DoF poses of autonomous agents (drones) from monocular camera images. Trained on the EuRoC MAV dataset, this project provides end-to-end training, evaluation, and interactive visualization tools for single-agent and multi-agent pose estimation.A comprehensive computer vision system for estimating precise poses of multiple autonomous agents (drones) using image and video data. This project demonstrates the integration of computer vision and machine learning to support formation flying and navigation in multi-agent systems.



## 🚀 Features## 🚀 Features



### Core Capabilities### Core Capabilities

- **Single-Agent Pose Estimation**: Precise 6-DoF pose (translation + quaternion rotation) from grayscale images- **Single-Agent Pose Estimation**: Precise 6DOF pose estimation from monocular images

- **Multi-Agent Pose Estimation**: Simultaneous pose estimation for multiple agents with presence detection- **Multi-Agent Pose Estimation**: Simultaneous pose estimation for multiple agents

- **Advanced Architectures**: ResNet backbones with attention mechanisms (CBAM) and specialized pose heads- **Formation Flying Support**: Specialized metrics and visualization for formation patterns

- **Interactive UI**: Gradio-based web interface for model inference and training metrics visualization- **Real-Time Inference**: Live video stream processing with real-time visualization

- **Geodesic Quaternion Loss**: Proper rotation error metric for orientation estimation- **Advanced Computer Vision**: Attention mechanisms, specialized pose heads, and data augmentation



### Model Architectures### Model Architectures

- **SimplePoseNet**: Lightweight CNN or ResNet18 baseline- **SimplePoseNet**: Lightweight CNN for basic pose estimation

- **AdvancedPoseNet**: ResNet backbone + CBAM attention + specialized pose head- **AdvancedPoseNet**: Enhanced model with attention mechanisms and specialized pose regression

- **MultiAgentPoseNet**: Shared feature extractor + multi-head pose prediction + agent presence detector- **MultiAgentPoseNet**: Multi-agent model with agent detection and formation analysis



### Training & Evaluation### Evaluation & Visualization

- **Automated Training Pipeline**: CLI-based training with checkpointing and CSV metrics logging- **Comprehensive Metrics**: ATE, RPE, formation metrics, and agent detection accuracy

- **Metrics Visualization**: Aggregated loss plots across all models in Gradio UI- **Advanced Visualizations**: 2D/3D trajectories, error analysis, formation patterns

- **EuRoC MAV Dataset Support**: Timestamp-aligned image-pose pairing from ground truth CSV- **Real-Time Monitoring**: Live trajectory plotting and error tracking



## 📁 Project Structure## 📁 Project Structure



``````

pose_train/pose_train/

├── model.py           # Model architectures (SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet)├── model.py                 # Model architectures (Simple, Advanced, Multi-Agent)

├── dataset.py         # EuRoC MAV dataset loader with timestamp alignment├── dataset.py              # Dataset loader with augmentation support

├── train.py           # Main training script with geodesic loss├── train.py                # Basic training script

├── gradio_app.py      # Interactive UI for inference and metrics visualization├── train_advanced.py       # Advanced training with comprehensive features

├── compare_models.py  # Model comparison utility (optional CLI tool)├── evaluation.py           # Evaluation metrics and reporting

└── prepare_metrics.py # Quick metrics caching for UI demos (optional)├── visualization.py        # Visualization tools and plotting

├── realtime_inference.py   # Real-time inference and live visualization

notebooks/├── eval.py                 # Basic evaluation utilities

└── train_and_eval.ipynb  # Training and evaluation notebook├── utils.py                # Utility functions

├── keyframe_extract.py     # Keyframe extraction utilities

work/└── feature_extract.py      # Feature extraction utilities

├── run_simple_full/      # SimplePoseNet training outputs

├── run_advanced_full/    # AdvancedPoseNet training outputsexamples/

└── run_multi_full/       # MultiAgentPoseNet training outputs└── complete_usage_example.py  # Comprehensive usage demonstration

    ├── results.csv       # Per-epoch train/val losses

    ├── best_ckpt.pt      # Best model checkpointnotebooks/

    └── ckpt_epoch_*.pt   # Per-epoch checkpoints├── train_and_eval.ipynb    # Basic training and evaluation notebook

```└── complete_demo.ipynb     # Complete feature demonstration notebook

```

## 🛠️ Installation

## 🛠️ Installation

### Prerequisites

- Python 3.8+### Prerequisites

- PyTorch 1.12+- Python 3.8+

- CUDA (optional, for GPU acceleration)- PyTorch 1.12+

- OpenCV

### Setup- NumPy, Pandas, Matplotlib

```bash- CUDA (optional, for GPU acceleration)

# Create virtual environment

python -m venv .venv### Setup

.\.venv\Scripts\Activate.ps1  # Windows PowerShell```bash

# source .venv/bin/activate    # Linux/Mac# Create virtual environment

python -m venv venv

# Install dependenciessource venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

```# Install dependencies

pip install -r requirements.txt

### Dependencies```

```

torch>=1.12.0## 🚀 Quick Start

torchvision

opencv-python### 1. Basic Training

numpy```bash

pandas# Train a simple model

matplotlibpython -m pose_train.train --data-root . --cams cam0 --epochs 20 --work-dir work

tqdm

pyyaml# Train with data augmentation

gradiopython -m pose_train.train --data-root . --cams cam0 --epochs 20 --augment --work-dir work_aug

Pillow```

```

### 2. Advanced Training

## 🚀 Quick Start```bash

# Train advanced model with attention

### 1. Training Modelspython pose_train/train_advanced.py \

    --data-root . \

#### Simple Model (25 epochs, batch 16)    --cams cam0 cam1 \

```powershell    --model-type advanced \

python -m pose_train.train `    --backbone resnet18 \

    --data-root . `    --pretrained \

    --model-type simple `    --use-attention \

    --backbone small `    --augment \

    --epochs 25 `    --epochs 50 \

    --batch-size 16 `    --batch-size 16 \

    --lr 1e-4 `    --work-dir work_advanced

    --work-dir work/run_simple_full

```# Train multi-agent model

python pose_train/train_advanced.py \

#### Advanced Model (40 epochs, batch 8)    --data-root . \

```powershell    --cams cam0 \

python -m pose_train.train `    --model-type multi_agent \

    --data-root . `    --backbone resnet18 \

    --model-type advanced `    --pretrained \

    --backbone resnet18 `    --use-attention \

    --epochs 40 `    --max-agents 5 \

    --batch-size 8 `    --epochs 50 \

    --lr 5e-5 `    --work-dir work_multi_agent

    --work-dir work/run_advanced_full```

```

### 3. Complete Usage Example

#### Multi-Agent Model (60 epochs, batch 4)```bash

```powershell# Run comprehensive demonstration

python -m pose_train.train `python examples/complete_usage_example.py

    --data-root . ````

    --model-type multi_agent `

    --backbone resnet18 `### 4. Real-Time Inference

    --epochs 60 ````bash

    --batch-size 4 `# Run real-time inference with webcam

    --lr 1e-5 `python pose_train/realtime_inference.py

    --max-agents 5 `

    --work-dir work/run_multi_full# Run with specific model

```python -c "

from pose_train.realtime_inference import create_realtime_demo

### 2. Launch Interactive UIcreate_realtime_demo('work_advanced/best_model.pt', 'advanced', 0)

"

```powershell```

# Start Gradio web interface

python -m pose_train.gradio_app## 📊 Usage Examples

```

### Single-Agent Pose Estimation

The UI provides:

- **Inference Tab**: Upload images and get pose predictions with visualization```python

- **Metrics Tab**: Aggregated train/val loss plots across all training runsimport torch

from pose_train.model import AdvancedPoseNet

Access at: `http://127.0.0.1:7860`from pose_train.dataset import MAVPoseDataset

from pose_train.visualization import plot_trajectory_2d, plot_error_analysis

### 3. Training Arguments Reference

# Load dataset

```dataset = MAVPoseDataset('.', cams=['cam0'], augment=True)

--data-root PATH         # Dataset root directory (default: current dir)

--cams CAM [CAM ...]     # Camera folders to use (default: cam0)# Create model

--model-type TYPE        # Model: simple, advanced, multi_agentmodel = AdvancedPoseNet(backbone='resnet18', pretrained=True, use_attention=True)

--backbone ARCH          # Backbone: small, resnet18, resnet50model.load_state_dict(torch.load('work_advanced/best_model.pt')['model_state_dict'])

--pretrained             # Use ImageNet pretrained weightsmodel.eval()

--use-imagenet-norm      # Use ImageNet normalization (RGB mode)

--epochs N               # Number of training epochs# Predict poses

--batch-size N           # Batch sizewith torch.no_grad():

--lr FLOAT               # Learning rate    img, gt_pose = dataset[0]

--max-agents N           # Max agents (multi_agent only, default: 3)    pred_pose = model(img.unsqueeze(0))

--work-dir PATH          # Output directory for checkpoints and logs    

```    print(f"Ground truth: {gt_pose.numpy()}")

    print(f"Predicted: {pred_pose.numpy()}")

## 📊 Usage Examples

# Visualize results

### Single-Agent Inferenceplot_trajectory_2d(gt_poses, pred_poses, title="Pose Estimation Results")

plot_error_analysis(gt_poses, pred_poses, title="Error Analysis")

```python```

import torch

from PIL import Image### Multi-Agent Pose Estimation

from pose_train.model import AdvancedPoseNet

from pose_train.dataset import get_transform```python

from pose_train.model import MultiAgentPoseNet

# Load modelfrom pose_train.visualization import plot_multi_agent_trajectory, plot_formation_analysis

model = AdvancedPoseNet(in_channels=1, backbone='resnet18', pretrained=False)

checkpoint = torch.load('work/run_advanced_full/best_ckpt.pt')# Create multi-agent model

model.load_state_dict(checkpoint['model_state_dict'])model = MultiAgentPoseNet(

model.eval()    backbone='resnet18', 

    pretrained=True, 

# Prepare image    max_agents=5, 

transform = get_transform(use_imagenet_norm=False, img_size=(240, 320))    use_attention=True

img = Image.open('cam0/data/1403636579763555584.png').convert('L'))

img_tensor = transform(img).unsqueeze(0)

# Predict poses for multiple agents

# Predict posewith torch.no_grad():

with torch.no_grad():    img, gt_pose = dataset[0]

    pose = model(img_tensor)  # [1, 7] -> [tx, ty, tz, qw, qx, qy, qz]    agent_poses, agent_presence = model(img.unsqueeze(0))

        

print(f"Translation: {pose[0, :3].numpy()}")    print(f"Agent poses shape: {agent_poses.shape}")

print(f"Rotation (quat): {pose[0, 3:].numpy()}")    print(f"Agent presence: {agent_presence}")

```

# Visualize multi-agent trajectories

### Multi-Agent Inferenceagent_poses_dict = {i: agent_poses[0, i].numpy() for i in range(5)}

agent_presence_dict = {i: agent_presence[0, i].numpy() for i in range(5)}

```python

from pose_train.model import MultiAgentPoseNetplot_multi_agent_trajectory(agent_poses_dict, agent_presence_dict)

plot_formation_analysis(agent_poses_dict, agent_presence_dict)

# Load multi-agent model```

model = MultiAgentPoseNet(

    in_channels=1, ## 🔧 Advanced Configuration

    backbone='resnet18', 

    pretrained=False, ### Model Configuration

    max_agents=5

)```python

checkpoint = torch.load('work/run_multi_full/best_ckpt.pt')# Advanced model with custom configuration

model.load_state_dict(checkpoint['model_state_dict'])model = AdvancedPoseNet(

model.eval()    in_channels=1,           # Number of input channels

    backbone='resnet50',     # Backbone architecture

# Predict    pretrained=True,         # Use pretrained weights

with torch.no_grad():    use_attention=True       # Enable attention mechanisms

    poses, presence = model(img_tensor))

    # poses: [1, max_agents, 7]

    # presence: [1, max_agents] (probabilities)# Multi-agent model

    model = MultiAgentPoseNet(

    for i in range(5):    in_channels=1,

        if presence[0, i] > 0.5:    backbone='resnet18',

            print(f"Agent {i}: {poses[0, i].numpy()}")    pretrained=True,

```    max_agents=10,           # Maximum number of agents

    use_attention=True

### Loading Training Results)

```

```python

import pandas as pd### Training Configuration



# Load training metrics```python

df = pd.read_csv('work/run_advanced_full/results.csv')# Advanced training with custom parameters

print(df.head())python pose_train/train_advanced.py \

    --data-root /path/to/dataset \

# epoch,train_loss,val_loss    --cams cam0 cam1 cam2 \

# 1,2.709286,0.444366    --model-type advanced \

# 2,0.686694,0.265551    --backbone resnet50 \

# ...    --pretrained \

```    --use-attention \

    --augment \

## 📈 Model Performance    --use-imagenet-norm \

    --epochs 100 \

### Training Results (EuRoC MH_01_easy)    --batch-size 32 \

    --lr 1e-4 \

| Model | Epochs | Final Train Loss | Final Val Loss | Training Time |    --weight-decay 1e-4 \

|-------|--------|------------------|----------------|---------------|    --optimizer adamw \

| SimplePoseNet | 25 | 0.6305 | 0.5931 | ~5 min/epoch |    --scheduler cosine \

| AdvancedPoseNet | 40 | 0.0242 | 0.0071 | ~4 min/epoch |    --grad-clip 1.0 \

| MultiAgentPoseNet | 60 | TBD | TBD | ~6 min/epoch |    --early-stopping 15 \

    --work-dir work_custom

*Training on CPU. GPU training is significantly faster.*```



### Loss Function## 📈 Performance Metrics

- **Translation**: Mean Squared Error (MSE) on `[tx, ty, tz]`

- **Rotation**: Geodesic distance between quaternions (proper rotation metric)### Single-Agent Metrics

- **Combined**: `loss = trans_loss + rot_loss`- **Translation RMSE**: Root mean square error for position

- **Rotation RMSE**: Root mean square error for orientation

## 🎯 Gradio UI Guide- **ATE (Absolute Trajectory Error)**: Overall trajectory accuracy

- **RPE (Relative Pose Error)**: Relative pose accuracy

### Inference Tab

1. Select model type (simple/advanced/multi_agent)### Multi-Agent Metrics

2. Choose backbone (small/resnet18/resnet50)- **Agent Detection**: Precision, recall, F1-score for agent presence

3. Upload an image from the dataset- **Formation Center Error**: Accuracy of formation center estimation

4. Click "Run" to get pose prediction- **Formation Shape Preservation**: Consistency of inter-agent distances

5. View annotated image with predicted pose overlay- **Formation Stability**: Variance in formation patterns



### Metrics Tab### Example Results

- Displays two plots:```

  - **Train Loss**: All models' training losses over epochsTranslation RMSE: 0.198 m

  - **Val Loss**: All models' validation losses over epochsRotation RMSE: 3.37°

- Each line represents a different model (color-coded)Detection Precision: 0.95

- Click "Refresh plots" after new training runs completeDetection Recall: 0.92

Formation Center RMSE: 0.15 m

## 🔧 Advanced UsageFormation Shape RMSE: 0.08 m

```

### Custom Dataset

## 🎯 Real-Time Applications

```python

from pose_train.dataset import MAVPoseDataset### Live Video Processing

```python

# Create dataset with custom camerasfrom pose_train.realtime_inference import RealTimePoseEstimator, RealTimeVisualizer

dataset = MAVPoseDataset(

    data_root='/path/to/dataset',# Create estimator

    cams=['cam0', 'cam1'],  # Multi-camera inputestimator = RealTimePoseEstimator(

    gt_file='state_groundtruth_estimate0/data.csv',    model_path='work_advanced/best_model.pt',

    use_imagenet_norm=False,    model_type='advanced',

    img_size=(240, 320)    device='cuda'

))



# Access samples# Start real-time processing

img, pose = dataset[0]estimator.start_realtime_processing(video_source=0)  # 0 for webcam

print(f"Image shape: {img.shape}")  # [C, H, W]```

print(f"Pose: {pose}")  # [7] -> [tx, ty, tz, qw, qx, qy, qz]

```### Formation Flying Demo

```python

### Resume Training# Multi-agent real-time processing

estimator = RealTimePoseEstimator(

```python    model_path='work_multi_agent/best_model.pt',

# train.py automatically saves checkpoints    model_type='multi_agent',

# To resume, load the checkpoint manually and continue training    max_agents=5

checkpoint = torch.load('work/run_advanced_full/ckpt_epoch_20.pt'))

model.load_state_dict(checkpoint['model_state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])# Start with live visualization

start_epoch = checkpoint['epoch'] + 1visualizer = RealTimeVisualizer(estimator)

```visualizer.start_live_plot()



### Export Model for Deploymentestimator.start_realtime_processing(video_source=0)

```

```python

# Save model in inference-ready format## 🔬 Research Applications

model.eval()

scripted_model = torch.jit.script(model)### Keyframe Extraction

scripted_model.save('model_deployment.pt')```bash

# Extract keyframes using interval method

# Load for inferencepython pose_train/keyframe_extract.py \

loaded_model = torch.jit.load('model_deployment.pt')    --img-dir cam0/data \

loaded_model.eval()    --out-dir cam0_keyframes \

```    --mode interval \

    --interval 10

## 📚 Dataset Information

# Extract keyframes using ORB-based scene change detection

### EuRoC MAV Datasetpython pose_train/keyframe_extract.py \

- **Source**: ETH Zurich Autonomous Systems Lab    --img-dir cam0/data \

- **Sequence Used**: MH_01_easy (Machine Hall, easy difficulty)    --out-dir cam0_keyframes \

- **Sensors**: Stereo cameras (cam0, cam1), IMU, ground truth (Leica MS50)    --mode orb_change \

- **Ground Truth**: 6-DoF pose at 200Hz with sub-millimeter accuracy    --threshold 30

- **Images**: Grayscale, VGA resolution, 20Hz```



### Data Structure### Feature Extraction

``````bash

mav0/# Extract ORB features

├── cam0/python pose_train/feature_extract.py \

│   ├── data/           # Timestamped images (*.png)    --img-dir cam0_keyframes \

│   ├── data.csv        # Image timestamps    --out-dir cam0_features \

│   └── sensor.yaml     # Camera calibration    --max-features 1000

├── cam1/               # Second camera (stereo pair)```

├── imu0/               # IMU measurements

├── leica0/             # External pose measurements### Model Comparison

└── state_groundtruth_estimate0/```bash

    └── data.csv        # Ground truth poses [timestamp, tx, ty, tz, qw, qx, qy, qz, ...]# Compare different models

```python pose_train/compare_models.py \

    --data-root . \

## 🤝 Contributing    --cams cam0 \

    --epochs 5 \

Contributions are welcome! To contribute:    --work-dir work_compare \

1. Fork the repository    --augment \

2. Create a feature branch (`git checkout -b feature/new-feature`)    --use-imagenet-norm

3. Commit changes (`git commit -m 'Add new feature'`)```

4. Push to branch (`git push origin feature/new-feature`)

5. Open a Pull Request## 📚 Documentation



## 📄 License- **[Advanced README](README_ADVANCED.md)**: Comprehensive documentation with detailed API reference

- **[Complete Usage Example](examples/complete_usage_example.py)**: Full demonstration script

This project is licensed under the MIT License.- **[Jupyter Notebooks](notebooks/)**: Interactive tutorials and examples



## 🙏 Acknowledgments## 🤝 Contributing



- **EuRoC MAV Dataset**: ETH Zurich ASL for the high-quality benchmark dataset1. Fork the repository

- **PyTorch**: For the deep learning framework2. Create a feature branch (`git checkout -b feature/amazing-feature`)

- **Gradio**: For the interactive UI library3. Commit your changes (`git commit -m 'Add amazing feature'`)

- **ResNet & CBAM**: Original authors for architecture designs4. Push to the branch (`git push origin feature/amazing-feature`)

5. Open a Pull Request

## 📞 Support

## 📄 License

For questions or issues:

- Open an issue on GitHubThis project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

- Check the [Gradio UI documentation](https://gradio.app/docs)

- Review example code in `notebooks/`## 🙏 Acknowledgments



---- PyTorch team for the deep learning framework

- OpenCV team for computer vision utilities

**Vision-Based Multi-Agent Pose Estimator** - Deep learning for autonomous agent pose estimation- The computer vision and robotics research community

- Contributors and users of this project

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers
- Check the documentation and examples

---

**Vision-Based Multi-Agent Pose Estimator** - Enabling precise formation flying through advanced computer vision and machine learning.