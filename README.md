# Vision-Based Multi-Agent Pose Estimator

A PyTorch-based deep learning system for estimating 6-DoF poses of autonomous agents (drones) from monocular camera images. Trained on the EuRoC MAV dataset, this project provides end-to-end training, evaluation, and interactive visualization tools for single-agent and multi-agent pose estimation.

## 🚀 Features

### Core Capabilities
- **Single-Agent Pose Estimation**: Precise 6-DoF pose (translation + quaternion rotation) from grayscale images
- **Multi-Agent Pose Estimation**: Simultaneous pose estimation for multiple agents with presence detection
- **Advanced Architectures**: ResNet backbones with attention mechanisms (CBAM) and specialized pose heads
- **Interactive UI**: Gradio-based web interface for model inference and training metrics visualization
- **Geodesic Quaternion Loss**: Proper rotation error metric for orientation estimation

### Model Architectures
- **SimplePoseNet**: Lightweight CNN or ResNet18 baseline
- **AdvancedPoseNet**: ResNet backbone + CBAM attention + specialized pose head
- **MultiAgentPoseNet**: Shared feature extractor + multi-head pose prediction + agent presence detector

### Training & Evaluation
- **Automated Training Pipeline**: CLI-based training with checkpointing and CSV metrics logging
- **Metrics Visualization**: Aggregated loss plots across all models in Gradio UI
- **Comprehensive Evaluation**: ATE, RPE, Translation RMSE, Rotation Geodesic Error metrics
- **EuRoC MAV Dataset Support**: Timestamp-aligned image-pose pairing from ground truth CSV

## 📁 Project Structure

```
pose_train/
├── model.py                # Model architectures (SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet)
├── dataset.py              # EuRoC MAV dataset loader with timestamp alignment
├── train.py                # Main training script with geodesic quaternion loss
├── gradio_app.py           # Interactive UI for inference and metrics visualization
├── generate_predictions.py # Generate predictions for evaluation
└── eval.py                 # Evaluation script (ATE, RPE, RMSE metrics)

notebooks/
└── train_and_eval.ipynb    # Training and evaluation notebook

work/
├── run_simple_full/        # SimplePoseNet training outputs (25 epochs)
├── run_advanced_full/      # AdvancedPoseNet training outputs (40 epochs)
└── run_multi_full/         # MultiAgentPoseNet training outputs (60 epochs)
    ├── results.csv         # Per-epoch train/val losses
    ├── best_ckpt.pt        # Best model checkpoint
    └── ckpt_epoch_*.pt     # Per-epoch checkpoints
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Setup
```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=1.12.0
torchvision
opencv-python
numpy
pandas
matplotlib
tqdm
pyyaml
gradio
Pillow
```

## 🚀 Quick Start

### 1. Training Models

#### Simple Model (25 epochs, batch 16)
```powershell
python -m pose_train.train `
    --data-root . `
    --model-type simple `
    --backbone small `
    --epochs 25 `
    --batch-size 16 `
    --lr 1e-4 `
    --work-dir work/run_simple_full
```

#### Advanced Model (40 epochs, batch 8)
```powershell
python -m pose_train.train `
    --data-root . `
    --model-type advanced `
    --backbone resnet18 `
    --epochs 40 `
    --batch-size 8 `
    --lr 5e-5 `
    --work-dir work/run_advanced_full
```

#### Multi-Agent Model (60 epochs, batch 4, 5 agents)
```powershell
python -m pose_train.train `
    --data-root . `
    --model-type multi_agent `
    --backbone resnet18 `
    --epochs 60 `
    --batch-size 4 `
    --lr 1e-5 `
    --max-agents 5 `
    --work-dir work/run_multi_full
```

### 2. Launch Interactive UI

```powershell
# Start Gradio web interface
python -m pose_train.gradio_app
```

The UI provides:
- **Inference Tab**: Upload images and get pose predictions with visualization
- **Metrics Tab**: Aggregated train/val loss plots across all training runs

Access at: `http://127.0.0.1:7860`

### 3. Generate Predictions & Evaluate

```powershell
# Generate predictions from trained model
python -m pose_train.generate_predictions `
    --checkpoint work/run_advanced_full/best_ckpt.pt `
    --model-type advanced `
    --backbone resnet18 `
    --output-dir work/run_advanced_full/eval_outputs

# Evaluate with metrics
python pose_train/eval.py `
    --gt work/run_advanced_full/eval_outputs/ground_truth.csv `
    --pred work/run_advanced_full/eval_outputs/predicted.csv
```

### 4. Training Arguments Reference

```
--data-root PATH         # Dataset root directory (default: current dir)
--cams CAM [CAM ...]     # Camera folders to use (default: cam0)
--model-type TYPE        # Model: simple, advanced, multi_agent
--backbone ARCH          # Backbone: small, resnet18, resnet50
--pretrained             # Use ImageNet pretrained weights
--use-imagenet-norm      # Use ImageNet normalization (RGB mode)
--epochs N               # Number of training epochs
--batch-size N           # Batch size
--lr FLOAT               # Learning rate
--max-agents N           # Max agents (multi_agent only, default: 3)
--work-dir PATH          # Output directory for checkpoints and logs
```

## 📊 Usage Examples

### Single-Agent Inference

```python
import torch
from PIL import Image
from pose_train.model import AdvancedPoseNet
from pose_train.dataset import get_transform

# Load model
model = AdvancedPoseNet(in_channels=1, backbone='resnet18', pretrained=False)
checkpoint = torch.load('work/run_advanced_full/best_ckpt.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = get_transform(use_imagenet_norm=False, img_size=(240, 320))
img = Image.open('cam0/data/1403636579763555584.png').convert('L')
img_tensor = transform(img).unsqueeze(0)

# Predict pose
with torch.no_grad():
    pose = model(img_tensor)  # [1, 7] -> [tx, ty, tz, qw, qx, qy, qz]
    
print(f"Translation: {pose[0, :3].numpy()}")
print(f"Rotation (quat): {pose[0, 3:].numpy()}")
```

### Multi-Agent Inference

```python
from pose_train.model import MultiAgentPoseNet

# Load multi-agent model
model = MultiAgentPoseNet(
    in_channels=1, 
    backbone='resnet18', 
    pretrained=False, 
    max_agents=5
)
checkpoint = torch.load('work/run_multi_full/best_ckpt.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    poses, presence = model(img_tensor)
    # poses: [1, max_agents, 7]
    # presence: [1, max_agents] (probabilities)
    
    for i in range(5):
        if presence[0, i] > 0.5:
            print(f"Agent {i}: {poses[0, i].numpy()}")
```

### Loading Training Results

```python
import pandas as pd

# Load training metrics
df = pd.read_csv('work/run_advanced_full/results.csv')
print(df.head())

# epoch,train_loss,val_loss
# 1,2.709286,0.444366
# 2,0.686694,0.265551
# ...
```

## 📈 Model Performance

### Training Results (EuRoC MH_01_easy)

| Model | Epochs | Final Train Loss | Final Val Loss | Training Time |
|-------|--------|------------------|----------------|---------------|
| SimplePoseNet | 25 | 0.6305 | 0.5931 | ~5 min/epoch |
| AdvancedPoseNet | 40 | 0.0242 | 0.0071 | ~4 min/epoch |
| MultiAgentPoseNet | 60 | 0.0238 | 0.0069 | ~6 min/epoch |

*Training on CPU. GPU training is significantly faster.*

### Evaluation Metrics (Validation Set)

| Model | ATE (m) | RPE (m) | Trans RMSE (m) | Rot Error (°) |
|-------|---------|---------|----------------|---------------|
| SimplePoseNet | 0.979 | 1.438 | 0.700 | 12.67 |
| **AdvancedPoseNet** | **0.081** | **0.112** | **0.055** | 1.80 |
| MultiAgentPoseNet | 0.087 | 0.120 | 0.058 | **1.69** |

**Winner: AdvancedPoseNet** - Best overall accuracy with 12x improvement over baseline!

### Loss Function
- **Translation**: Mean Squared Error (MSE) on `[tx, ty, tz]`
- **Rotation**: Geodesic distance between quaternions (proper rotation metric)
- **Combined**: `loss = trans_loss + rot_loss`

## 🎯 Gradio UI Guide

### Inference Tab
1. Select model type (simple/advanced/multi_agent)
2. Choose backbone (small/resnet18/resnet50)
3. Upload an image from the dataset
4. Click "Run" to get pose prediction
5. View annotated image with predicted pose overlay

### Metrics Tab
- Displays two plots:
  - **Train Loss**: All models' training losses over epochs
  - **Val Loss**: All models' validation losses over epochs
- Each line represents a different model (color-coded)
- Click "Refresh plots" after new training runs complete

## 📚 Dataset Information

### EuRoC MAV Dataset
- **Source**: ETH Zurich Autonomous Systems Lab
- **Sequence Used**: MH_01_easy (Machine Hall, easy difficulty)
- **Sensors**: Stereo cameras (cam0, cam1), IMU, ground truth (Leica MS50)
- **Ground Truth**: 6-DoF pose at 200Hz with sub-millimeter accuracy
- **Images**: Grayscale, 752×480 pixels, 20Hz
- **Total Frames**: ~3,640 images
- **Train/Val Split**: 80/20 (~2,912 train, ~728 validation)

### Data Structure
```
mav0/
├── cam0/
│   ├── data/           # Timestamped images (*.png)
│   ├── data.csv        # Image timestamps
│   └── sensor.yaml     # Camera calibration
├── cam1/               # Second camera (stereo pair)
├── imu0/               # IMU measurements
├── leica0/             # External pose measurements
└── state_groundtruth_estimate0/
    └── data.csv        # Ground truth poses [timestamp, tx, ty, tz, qw, qx, qy, qz, ...]
```

## 🔧 Evaluation Metrics Explained

### ATE (Absolute Trajectory Error)
- Average Euclidean distance between predicted and ground truth positions
- Lower is better

### RPE (Relative Pose Error)
- Measures frame-to-frame consistency
- Important for trajectory smoothness

### Translation RMSE
- Root Mean Square Error for position (x, y, z)
- Direct measure of position accuracy

### Rotation Geodesic Error
- Angular distance between predicted and true quaternions
- Measured in degrees

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **EuRoC MAV Dataset**: ETH Zurich ASL for the high-quality benchmark dataset
- **PyTorch**: For the deep learning framework
- **Gradio**: For the interactive UI library
- **ResNet & CBAM**: Original authors for architecture designs

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the [Gradio UI documentation](https://gradio.app/docs)
- Review example code in `notebooks/`
- See `EVALUATION_RESULTS.md` for detailed performance analysis

---

**Vision-Based Multi-Agent Pose Estimator** - Deep learning for autonomous agent pose estimation
