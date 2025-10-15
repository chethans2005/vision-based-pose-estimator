# Vision-Based Multi-Agent Pose Estimator# Vision-Based Multi-Agent Pose Estimator# Vision-Based Multi-Agent Pose Estimator



A PyTorch-based deep learning system for estimating 6-DoF poses of autonomous agents (drones) from monocular camera images. Trained on the EuRoC MAV dataset, this project provides end-to-end training, evaluation, and interactive visualization tools for single-agent and multi-agent pose estimation.



## 🚀 FeaturesA PyTorch-based deep learning system for estimating 6-DoF poses of autonomous agents (drones) from monocular camera images. Trained on the EuRoC MAV dataset, this project provides end-to-end training, evaluation, and interactive visualization tools for single-agent and multi-agent pose estimation.A comprehensive computer vision system for estimating precise poses of multiple autonomous agents (drones) using image and video data. This project demonstrates the integration of computer vision and machine learning to support formation flying and navigation in multi-agent systems.



### Core Capabilities

- **Single-Agent Pose Estimation**: Precise 6-DoF pose (translation + quaternion rotation) from grayscale images

- **Multi-Agent Pose Estimation**: Simultaneous pose estimation for multiple agents with presence detection## 🚀 Features## 🚀 Features

- **Advanced Architectures**: ResNet backbones with attention mechanisms (CBAM) and specialized pose heads

- **Interactive UI**: Gradio-based web interface for model inference and training metrics visualization

- **Geodesic Quaternion Loss**: Proper rotation error metric for orientation estimation

### Core Capabilities### Core Capabilities

### Model Architectures

- **SimplePoseNet**: Lightweight CNN or ResNet18 baseline- **Single-Agent Pose Estimation**: Precise 6-DoF pose (translation + quaternion rotation) from grayscale images- **Single-Agent Pose Estimation**: Precise 6DOF pose estimation from monocular images

- **AdvancedPoseNet**: ResNet backbone + CBAM attention + specialized pose head

- **MultiAgentPoseNet**: Shared feature extractor + multi-head pose prediction + agent presence detector- **Multi-Agent Pose Estimation**: Simultaneous pose estimation for multiple agents with presence detection- **Multi-Agent Pose Estimation**: Simultaneous pose estimation for multiple agents



### Training & Evaluation- **Advanced Architectures**: ResNet backbones with attention mechanisms (CBAM) and specialized pose heads- **Formation Flying Support**: Specialized metrics and visualization for formation patterns

- **Automated Training Pipeline**: CLI-based training with checkpointing and CSV metrics logging

- **Metrics Visualization**: Aggregated loss plots across all models in Gradio UI- **Interactive UI**: Gradio-based web interface for model inference and training metrics visualization- **Real-Time Inference**: Live video stream processing with real-time visualization

- **Comprehensive Evaluation**: ATE, RPE, Translation RMSE, Rotation Geodesic Error metrics

- **EuRoC MAV Dataset Support**: Timestamp-aligned image-pose pairing from ground truth CSV- **Geodesic Quaternion Loss**: Proper rotation error metric for orientation estimation- **Advanced Computer Vision**: Attention mechanisms, specialized pose heads, and data augmentation



## 📁 Project Structure



```### Model Architectures### Model Architectures

pose_train/

├── model.py                # Model architectures (SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet)- **SimplePoseNet**: Lightweight CNN or ResNet18 baseline- **SimplePoseNet**: Lightweight CNN for basic pose estimation

├── dataset.py              # EuRoC MAV dataset loader with timestamp alignment

├── train.py                # Main training script with geodesic quaternion loss- **AdvancedPoseNet**: ResNet backbone + CBAM attention + specialized pose head- **AdvancedPoseNet**: Enhanced model with attention mechanisms and specialized pose regression

├── gradio_app.py           # Interactive UI for inference and metrics visualization

├── generate_predictions.py # Generate predictions for evaluation- **MultiAgentPoseNet**: Shared feature extractor + multi-head pose prediction + agent presence detector- **MultiAgentPoseNet**: Multi-agent model with agent detection and formation analysis

└── eval.py                 # Evaluation script (ATE, RPE, RMSE metrics)



notebooks/

└── train_and_eval.ipynb    # Training and evaluation notebook### Training & Evaluation### Evaluation & Visualization



work/- **Automated Training Pipeline**: CLI-based training with checkpointing and CSV metrics logging- **Comprehensive Metrics**: ATE, RPE, formation metrics, and agent detection accuracy

├── run_simple_full/        # SimplePoseNet training outputs (25 epochs)

├── run_advanced_full/      # AdvancedPoseNet training outputs (40 epochs)- **Metrics Visualization**: Aggregated loss plots across all models in Gradio UI- **Advanced Visualizations**: 2D/3D trajectories, error analysis, formation patterns

└── run_multi_full/         # MultiAgentPoseNet training outputs (60 epochs)

    ├── results.csv         # Per-epoch train/val losses- **EuRoC MAV Dataset Support**: Timestamp-aligned image-pose pairing from ground truth CSV- **Real-Time Monitoring**: Live trajectory plotting and error tracking

    ├── best_ckpt.pt        # Best model checkpoint

    └── ckpt_epoch_*.pt     # Per-epoch checkpoints

```

## 📁 Project Structure## 📁 Project Structure

## 🛠️ Installation



### Prerequisites

- Python 3.8+``````

- PyTorch 1.12+

- CUDA (optional, for GPU acceleration)pose_train/pose_train/



### Setup├── model.py           # Model architectures (SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet)├── model.py                 # Model architectures (Simple, Advanced, Multi-Agent)

```powershell

# Create virtual environment├── dataset.py         # EuRoC MAV dataset loader with timestamp alignment├── dataset.py              # Dataset loader with augmentation support

python -m venv .venv

.\.venv\Scripts\Activate.ps1  # Windows PowerShell├── train.py           # Main training script with geodesic loss├── train.py                # Basic training script



# Install dependencies├── gradio_app.py      # Interactive UI for inference and metrics visualization├── train_advanced.py       # Advanced training with comprehensive features

pip install -r requirements.txt

```├── compare_models.py  # Model comparison utility (optional CLI tool)├── evaluation.py           # Evaluation metrics and reporting



### Dependencies└── prepare_metrics.py # Quick metrics caching for UI demos (optional)├── visualization.py        # Visualization tools and plotting

```

torch>=1.12.0├── realtime_inference.py   # Real-time inference and live visualization

torchvision

opencv-pythonnotebooks/├── eval.py                 # Basic evaluation utilities

numpy

pandas└── train_and_eval.ipynb  # Training and evaluation notebook├── utils.py                # Utility functions

matplotlib

tqdm├── keyframe_extract.py     # Keyframe extraction utilities

pyyaml

gradiowork/└── feature_extract.py      # Feature extraction utilities

Pillow

```├── run_simple_full/      # SimplePoseNet training outputs



## 🚀 Quick Start├── run_advanced_full/    # AdvancedPoseNet training outputsexamples/



### 1. Training Models└── run_multi_full/       # MultiAgentPoseNet training outputs└── complete_usage_example.py  # Comprehensive usage demonstration



#### Simple Model (25 epochs, batch 16)    ├── results.csv       # Per-epoch train/val losses

```powershell

python -m pose_train.train `    ├── best_ckpt.pt      # Best model checkpointnotebooks/

    --data-root . `

    --model-type simple `    └── ckpt_epoch_*.pt   # Per-epoch checkpoints├── train_and_eval.ipynb    # Basic training and evaluation notebook

    --backbone small `

    --epochs 25 ````└── complete_demo.ipynb     # Complete feature demonstration notebook

    --batch-size 16 `

    --lr 1e-4 ````

    --work-dir work/run_simple_full

```## 🛠️ Installation



#### Advanced Model (40 epochs, batch 8)## 🛠️ Installation

```powershell

python -m pose_train.train `### Prerequisites

    --data-root . `

    --model-type advanced `- Python 3.8+### Prerequisites

    --backbone resnet18 `

    --epochs 40 `- PyTorch 1.12+- Python 3.8+

    --batch-size 8 `

    --lr 5e-5 `- CUDA (optional, for GPU acceleration)- PyTorch 1.12+

    --work-dir work/run_advanced_full

```- OpenCV



#### Multi-Agent Model (60 epochs, batch 4, 5 agents)### Setup- NumPy, Pandas, Matplotlib

```powershell

python -m pose_train.train ````bash- CUDA (optional, for GPU acceleration)

    --data-root . `

    --model-type multi_agent `# Create virtual environment

    --backbone resnet18 `

    --epochs 60 `python -m venv .venv### Setup

    --batch-size 4 `

    --lr 1e-5 `.\.venv\Scripts\Activate.ps1  # Windows PowerShell```bash

    --max-agents 5 `

    --work-dir work/run_multi_full# source .venv/bin/activate    # Linux/Mac# Create virtual environment

```

python -m venv venv

### 2. Launch Interactive UI

# Install dependenciessource venv/bin/activate  # On Windows: venv\Scripts\activate

```powershell

# Start Gradio web interfacepip install -r requirements.txt

python -m pose_train.gradio_app

``````# Install dependencies



The UI provides:pip install -r requirements.txt

- **Inference Tab**: Upload images and get pose predictions with visualization

- **Metrics Tab**: Aggregated train/val loss plots across all training runs### Dependencies```



Access at: `http://127.0.0.1:7860````



### 3. Generate Predictions & Evaluatetorch>=1.12.0## 🚀 Quick Start



```powershelltorchvision

# Generate predictions from trained model

python -m pose_train.generate_predictions `opencv-python### 1. Basic Training

    --checkpoint work/run_advanced_full/best_ckpt.pt `

    --model-type advanced `numpy```bash

    --backbone resnet18 `

    --output-dir work/run_advanced_full/eval_outputspandas# Train a simple model



# Evaluate with metricsmatplotlibpython -m pose_train.train --data-root . --cams cam0 --epochs 20 --work-dir work

python pose_train/eval.py `

    --gt work/run_advanced_full/eval_outputs/ground_truth.csv `tqdm

    --pred work/run_advanced_full/eval_outputs/predicted.csv

```pyyaml# Train with data augmentation



### 4. Training Arguments Referencegradiopython -m pose_train.train --data-root . --cams cam0 --epochs 20 --augment --work-dir work_aug



```Pillow```

--data-root PATH         # Dataset root directory (default: current dir)

--cams CAM [CAM ...]     # Camera folders to use (default: cam0)```

--model-type TYPE        # Model: simple, advanced, multi_agent

--backbone ARCH          # Backbone: small, resnet18, resnet50### 2. Advanced Training

--pretrained             # Use ImageNet pretrained weights

--use-imagenet-norm      # Use ImageNet normalization (RGB mode)## 🚀 Quick Start```bash

--epochs N               # Number of training epochs

--batch-size N           # Batch size# Train advanced model with attention

--lr FLOAT               # Learning rate

--max-agents N           # Max agents (multi_agent only, default: 3)### 1. Training Modelspython pose_train/train_advanced.py \

--work-dir PATH          # Output directory for checkpoints and logs

```    --data-root . \



## 📊 Usage Examples#### Simple Model (25 epochs, batch 16)    --cams cam0 cam1 \



### Single-Agent Inference```powershell    --model-type advanced \



```pythonpython -m pose_train.train `    --backbone resnet18 \

import torch

from PIL import Image    --data-root . `    --pretrained \

from pose_train.model import AdvancedPoseNet

from pose_train.dataset import get_transform    --model-type simple `    --use-attention \



# Load model    --backbone small `    --augment \

model = AdvancedPoseNet(in_channels=1, backbone='resnet18', pretrained=False)

checkpoint = torch.load('work/run_advanced_full/best_ckpt.pt')    --epochs 25 `    --epochs 50 \

model.load_state_dict(checkpoint['model_state'])

model.eval()    --batch-size 16 `    --batch-size 16 \



# Prepare image    --lr 1e-4 `    --work-dir work_advanced

transform = get_transform(use_imagenet_norm=False, img_size=(240, 320))

img = Image.open('cam0/data/1403636579763555584.png').convert('L')    --work-dir work/run_simple_full

img_tensor = transform(img).unsqueeze(0)

```# Train multi-agent model

# Predict pose

with torch.no_grad():python pose_train/train_advanced.py \

    pose = model(img_tensor)  # [1, 7] -> [tx, ty, tz, qw, qx, qy, qz]

    #### Advanced Model (40 epochs, batch 8)    --data-root . \

print(f"Translation: {pose[0, :3].numpy()}")

print(f"Rotation (quat): {pose[0, 3:].numpy()}")```powershell    --cams cam0 \

```

python -m pose_train.train `    --model-type multi_agent \

### Multi-Agent Inference

    --data-root . `    --backbone resnet18 \

```python

from pose_train.model import MultiAgentPoseNet    --model-type advanced `    --pretrained \



# Load multi-agent model    --backbone resnet18 `    --use-attention \

model = MultiAgentPoseNet(

    in_channels=1,     --epochs 40 `    --max-agents 5 \

    backbone='resnet18', 

    pretrained=False,     --batch-size 8 `    --epochs 50 \

    max_agents=5

)    --lr 5e-5 `    --work-dir work_multi_agent

checkpoint = torch.load('work/run_multi_full/best_ckpt.pt')

model.load_state_dict(checkpoint['model_state'])    --work-dir work/run_advanced_full```

model.eval()

```

# Predict

with torch.no_grad():### 3. Complete Usage Example

    poses, presence = model(img_tensor)

    # poses: [1, max_agents, 7]#### Multi-Agent Model (60 epochs, batch 4)```bash

    # presence: [1, max_agents] (probabilities)

    ```powershell# Run comprehensive demonstration

    for i in range(5):

        if presence[0, i] > 0.5:python -m pose_train.train `python examples/complete_usage_example.py

            print(f"Agent {i}: {poses[0, i].numpy()}")

```    --data-root . ````



### Loading Training Results    --model-type multi_agent `



```python    --backbone resnet18 `### 4. Real-Time Inference

import pandas as pd

    --epochs 60 ````bash

# Load training metrics

df = pd.read_csv('work/run_advanced_full/results.csv')    --batch-size 4 `# Run real-time inference with webcam

print(df.head())

    --lr 1e-5 `python pose_train/realtime_inference.py

# epoch,train_loss,val_loss

# 1,2.709286,0.444366    --max-agents 5 `

# 2,0.686694,0.265551

# ...    --work-dir work/run_multi_full# Run with specific model

```

```python -c "

## 📈 Model Performance

from pose_train.realtime_inference import create_realtime_demo

### Training Results (EuRoC MH_01_easy)

### 2. Launch Interactive UIcreate_realtime_demo('work_advanced/best_model.pt', 'advanced', 0)

| Model | Epochs | Final Train Loss | Final Val Loss | Training Time |

|-------|--------|------------------|----------------|---------------|"

| SimplePoseNet | 25 | 0.6305 | 0.5931 | ~5 min/epoch |

| AdvancedPoseNet | 40 | 0.0242 | 0.0071 | ~4 min/epoch |```powershell```

| MultiAgentPoseNet | 60 | 0.0238 | 0.0069 | ~6 min/epoch |

# Start Gradio web interface

*Training on CPU. GPU training is significantly faster.*

python -m pose_train.gradio_app## 📊 Usage Examples

### Evaluation Metrics (Validation Set)

```

| Model | ATE (m) | RPE (m) | Trans RMSE (m) | Rot Error (°) |

|-------|---------|---------|----------------|---------------|### Single-Agent Pose Estimation

| SimplePoseNet | 0.979 | 1.438 | 0.700 | 12.67 |

| **AdvancedPoseNet** | **0.081** | **0.112** | **0.055** | 1.80 |The UI provides:

| MultiAgentPoseNet | 0.087 | 0.120 | 0.058 | **1.69** |

- **Inference Tab**: Upload images and get pose predictions with visualization```python

**Winner: AdvancedPoseNet** - Best overall accuracy with 12x improvement over baseline!

- **Metrics Tab**: Aggregated train/val loss plots across all training runsimport torch

### Loss Function

- **Translation**: Mean Squared Error (MSE) on `[tx, ty, tz]`from pose_train.model import AdvancedPoseNet

- **Rotation**: Geodesic distance between quaternions (proper rotation metric)

- **Combined**: `loss = trans_loss + rot_loss`Access at: `http://127.0.0.1:7860`from pose_train.dataset import MAVPoseDataset



## 🎯 Gradio UI Guidefrom pose_train.visualization import plot_trajectory_2d, plot_error_analysis



### Inference Tab### 3. Training Arguments Reference

1. Select model type (simple/advanced/multi_agent)

2. Choose backbone (small/resnet18/resnet50)# Load dataset

3. Upload an image from the dataset

4. Click "Run" to get pose prediction```dataset = MAVPoseDataset('.', cams=['cam0'], augment=True)

5. View annotated image with predicted pose overlay

--data-root PATH         # Dataset root directory (default: current dir)

### Metrics Tab

- Displays two plots:--cams CAM [CAM ...]     # Camera folders to use (default: cam0)# Create model

  - **Train Loss**: All models' training losses over epochs

  - **Val Loss**: All models' validation losses over epochs--model-type TYPE        # Model: simple, advanced, multi_agentmodel = AdvancedPoseNet(backbone='resnet18', pretrained=True, use_attention=True)

- Each line represents a different model (color-coded)

- Click "Refresh plots" after new training runs complete--backbone ARCH          # Backbone: small, resnet18, resnet50model.load_state_dict(torch.load('work_advanced/best_model.pt')['model_state_dict'])



## 📚 Dataset Information--pretrained             # Use ImageNet pretrained weightsmodel.eval()



### EuRoC MAV Dataset--use-imagenet-norm      # Use ImageNet normalization (RGB mode)

- **Source**: ETH Zurich Autonomous Systems Lab

- **Sequence Used**: MH_01_easy (Machine Hall, easy difficulty)--epochs N               # Number of training epochs# Predict poses

- **Sensors**: Stereo cameras (cam0, cam1), IMU, ground truth (Leica MS50)

- **Ground Truth**: 6-DoF pose at 200Hz with sub-millimeter accuracy--batch-size N           # Batch sizewith torch.no_grad():

- **Images**: Grayscale, 752×480 pixels, 20Hz

- **Total Frames**: ~3,640 images--lr FLOAT               # Learning rate    img, gt_pose = dataset[0]

- **Train/Val Split**: 80/20 (~2,912 train, ~728 validation)

--max-agents N           # Max agents (multi_agent only, default: 3)    pred_pose = model(img.unsqueeze(0))

### Data Structure

```--work-dir PATH          # Output directory for checkpoints and logs    

mav0/

├── cam0/```    print(f"Ground truth: {gt_pose.numpy()}")

│   ├── data/           # Timestamped images (*.png)

│   ├── data.csv        # Image timestamps    print(f"Predicted: {pred_pose.numpy()}")

│   └── sensor.yaml     # Camera calibration

├── cam1/               # Second camera (stereo pair)## 📊 Usage Examples

├── imu0/               # IMU measurements

├── leica0/             # External pose measurements# Visualize results

└── state_groundtruth_estimate0/

    └── data.csv        # Ground truth poses [timestamp, tx, ty, tz, qw, qx, qy, qz, ...]### Single-Agent Inferenceplot_trajectory_2d(gt_poses, pred_poses, title="Pose Estimation Results")

```

plot_error_analysis(gt_poses, pred_poses, title="Error Analysis")

## 🔧 Evaluation Metrics Explained

```python```

### ATE (Absolute Trajectory Error)

- Average Euclidean distance between predicted and ground truth positionsimport torch

- Lower is better

from PIL import Image### Multi-Agent Pose Estimation

### RPE (Relative Pose Error)

- Measures frame-to-frame consistencyfrom pose_train.model import AdvancedPoseNet

- Important for trajectory smoothness

from pose_train.dataset import get_transform```python

### Translation RMSE

- Root Mean Square Error for position (x, y, z)from pose_train.model import MultiAgentPoseNet

- Direct measure of position accuracy

# Load modelfrom pose_train.visualization import plot_multi_agent_trajectory, plot_formation_analysis

### Rotation Geodesic Error

- Angular distance between predicted and true quaternionsmodel = AdvancedPoseNet(in_channels=1, backbone='resnet18', pretrained=False)

- Measured in degrees

checkpoint = torch.load('work/run_advanced_full/best_ckpt.pt')# Create multi-agent model

## 🤝 Contributing

model.load_state_dict(checkpoint['model_state_dict'])model = MultiAgentPoseNet(

Contributions are welcome! To contribute:

1. Fork the repositorymodel.eval()    backbone='resnet18', 

2. Create a feature branch (`git checkout -b feature/new-feature`)

3. Commit changes (`git commit -m 'Add new feature'`)    pretrained=True, 

4. Push to branch (`git push origin feature/new-feature`)

5. Open a Pull Request# Prepare image    max_agents=5, 



## 📄 Licensetransform = get_transform(use_imagenet_norm=False, img_size=(240, 320))    use_attention=True



This project is licensed under the MIT License.img = Image.open('cam0/data/1403636579763555584.png').convert('L'))



## 🙏 Acknowledgmentsimg_tensor = transform(img).unsqueeze(0)



- **EuRoC MAV Dataset**: ETH Zurich ASL for the high-quality benchmark dataset# Predict poses for multiple agents

- **PyTorch**: For the deep learning framework

- **Gradio**: For the interactive UI library# Predict posewith torch.no_grad():

- **ResNet & CBAM**: Original authors for architecture designs

with torch.no_grad():    img, gt_pose = dataset[0]

## 📞 Support

    pose = model(img_tensor)  # [1, 7] -> [tx, ty, tz, qw, qx, qy, qz]    agent_poses, agent_presence = model(img.unsqueeze(0))

For questions or issues:

- Open an issue on GitHub        

- Check the [Gradio UI documentation](https://gradio.app/docs)

- Review example code in `notebooks/`print(f"Translation: {pose[0, :3].numpy()}")    print(f"Agent poses shape: {agent_poses.shape}")

- See `EVALUATION_RESULTS.md` for detailed performance analysis

print(f"Rotation (quat): {pose[0, 3:].numpy()}")    print(f"Agent presence: {agent_presence}")

---

```

**Vision-Based Multi-Agent Pose Estimator** - Deep learning for autonomous agent pose estimation

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