# MovieLens Small Dataset Experiment Report

**Experiment Date**: 2025-08-29 17:22:32
**Dataset**: MovieLens Small (100K ratings)

## 📊 Dataset Statistics

- **Users**: 610
- **Movies**: 9,724
- **Total Ratings**: 100,836
- **Training Samples**: 80,668
- **Test Samples**: 20,168

## 🎯 Baseline Results

- **Device**: cuda
- **Model Parameters**: 1,098,406
- **Model Size**: 4.19 MB
- **Data Processing Time**: 4.2s
- **Training Time**: 254.4s
- **Final Train Loss**: 0.5898
- **Final Val Loss**: 0.8019
- **Test RMSE**: 0.8958
- **Inference Time**: 0.08ms

## ⚖️ Model Size Comparison

### TINY
- **Parameters**: 441,044
- **Size**: 1.68 MB
- **Training Time**: 0.3s
- **Inference Time**: 5.16ms

### SMALL
- **Parameters**: 1,098,406
- **Size**: 4.19 MB
- **Training Time**: 0.4s
- **Inference Time**: 1.51ms

### MEDIUM
- **Parameters**: 3,062,090
- **Size**: 11.68 MB
- **Training Time**: 0.7s
- **Inference Time**: 1.67ms

## 📁 Generated Files

- `results.json`: Complete experimental results
- `experiment_results.png`: Training curves and model comparison
- `baseline_checkpoints/`: Model checkpoints
- `baseline_logs/`: Training logs
