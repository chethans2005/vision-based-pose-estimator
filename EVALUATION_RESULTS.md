# Model Evaluation Results

## Evaluation Metrics Summary

### SimplePoseNet (25 epochs)
- **ATE (Absolute Trajectory Error)**: 0.9786 m
- **RPE (Relative Pose Error)**: 1.4379 m
- **Translation RMSE**: 0.6997 m
- **Rotation Geodesic Error**: 12.67°

### AdvancedPoseNet (40 epochs)
- **ATE (Absolute Trajectory Error)**: 0.0808 m
- **RPE (Relative Pose Error)**: 0.1124 m
- **Translation RMSE**: 0.0546 m
- **Rotation Geodesic Error**: 1.80°

### MultiAgentPoseNet (60 epochs)
- **ATE (Absolute Trajectory Error)**: 0.0866 m
- **RPE (Relative Pose Error)**: 0.1203 m
- **Translation RMSE**: 0.0576 m
- **Rotation Geodesic Error**: 1.69°

## Comparison

| Metric | SimplePoseNet | AdvancedPoseNet | MultiAgentPoseNet | Best Model |
|--------|---------------|-----------------|-------------------|------------|
| ATE (m) | 0.9786 | **0.0808** | 0.0866 | AdvancedPoseNet |
| RPE (m) | 1.4379 | **0.1124** | 0.1203 | AdvancedPoseNet |
| Translation RMSE (m) | 0.6997 | **0.0546** | 0.0576 | AdvancedPoseNet |
| Rotation Error (°) | 12.67 | 1.80 | **1.69** | MultiAgentPoseNet |

### Improvement Over Baseline (SimplePoseNet)

| Metric | AdvancedPoseNet | MultiAgentPoseNet |
|--------|-----------------|-------------------|
| ATE | **12.1x better** | 11.3x better |
| RPE | **12.8x better** | 12.0x better |
| Translation RMSE | **12.8x better** | 12.2x better |
| Rotation Error | 7.0x better | **7.5x better** |

## Key Findings

1. **Both AdvancedPoseNet and MultiAgentPoseNet vastly outperform SimplePoseNet**
2. **AdvancedPoseNet achieves the best translation accuracy** (RMSE: 0.0546m)
3. **MultiAgentPoseNet achieves the best rotation accuracy** (1.69°)
4. The ResNet18 backbone provides much better feature extraction than the simple CNN
5. Multi-agent model achieves competitive single-agent performance while supporting multiple agents
6. All advanced models achieve sub-10cm translation error and sub-2° rotation error

## Model Selection Recommendations

- **For single-agent pose estimation**: Use **AdvancedPoseNet** (best translation accuracy)
- **For multi-agent scenarios**: Use **MultiAgentPoseNet** (comparable accuracy, supports multiple agents)
- **For resource-constrained environments**: AdvancedPoseNet offers best accuracy-to-complexity ratio

## Conclusion

Both advanced models (AdvancedPoseNet and MultiAgentPoseNet) achieve excellent pose estimation accuracy on the EuRoC MAV dataset:
- **Translation accuracy**: ~5-6 cm RMSE
- **Rotation accuracy**: ~1.7-1.8 degrees
- **Trajectory error**: ~8-9 cm ATE

These results demonstrate that deep learning with ResNet backbones can achieve highly accurate vision-based pose estimation for autonomous agents.

---

*Generated on: October 15, 2025*
*Dataset: EuRoC MAV MH_01_easy*
*Validation set size: 728 samples*
