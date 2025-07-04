# Multiscale feature enhanced gating network for atrial fibrillation detection
Code for Multiscale feature enhanced gating network for atrial fibrillation detection.
---
## 🔑 Key Features
- **Multiscale Feature Extraction**  
  MSConv module captures global rhythm patterns and local waveform features simultaneously through parallel 1×3 to 1×17 kernels
  
- **Adaptive Feature Enhancement**  
  Triple-module architecture eliminates noise while enhancing AF signatures:
  - Soft-threshold residual shrinkage (SRS) with dynamic noise filtering
  - Dilated convolution for extended context awareness
  - SE attention for channel-wise feature recalibration
  
- **Clinical-Grade Robustness**  
  Maintains 86.5% accuracy under extreme noise (0dB SNR) and handles real-world artifacts

## 🚀 Performance Highlights
| Metric        | CinC2017 | CPSC2018 | AFDB   |
|---------------|----------|----------|--------|
| **Accuracy**  | 93.0%    | 90.8%    | 93.8%  |
| **F1 Score**  | 88.3%    | 82.0%    | 93.5%  |
| **Precision** | 89.4%    | 88.1%    | 94.2%  |

**Noise Robustness** (0dB Gaussian):  
86.5% accuracy (vs. 55.6% for ResRNN)

## ⚡️ Inference Speed
| Device         | Latency |
|----------------|---------|
| GPU (RTX A6000)| 2.08ms  |
| CPU (Xeon E5)  | 11.86ms |
| Raspberry Pi 5 | 57.72ms |

## 🚀 Get Started
### Requirements
tensorflow==2.6.0
keras==2.6.0

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@article{wu2025multiscale,
  title={Multiscale feature enhanced gating network for atrial fibrillation detection},
  author={Wu, Xidong and Yan, Mingke and Wang, Renqiao and Xie, Liping},
  journal={Computer Methods and Programs in Biomedicine},
  pages={108606},
  year={2025},
  publisher={Elsevier}
}