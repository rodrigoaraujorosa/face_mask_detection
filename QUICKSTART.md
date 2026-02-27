# Quick Start Guide

Get up and running with Face Mask Detection in 5 minutes!

## 1. Setup (2 minutes)

```bash
# Navigate to project
cd /home/<user>/workspace/face-mask-detection/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Run Training (30-60 minutes)

### Option A: VSCode with Jupyter Extension (Recommended)

1. Open VSCode
2. Install Jupyter extension (if not installed)
3. Open `notebooks/face_mask_detection.ipynb`
4. Click "Select Kernel" → Choose your Python environment
5. Run all cells (Ctrl+Shift+Enter or click "Run All")

### Option B: Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/face_mask_detection.ipynb
# Run all cells
```

### Option C: Jupyter Lab

```bash
# Start Jupyter Lab
jupyter lab

# Navigate to notebooks/face_mask_detection.ipynb
# Run all cells
```

## 3. What Happens During Training

The notebook will automatically:

1. **Download dataset** (~2GB, first run only) ✓
2. **Preprocess images** (resize, normalize, augment) ✓
3. **Split data** (70% train, 15% val, 15% test) ✓
4. **Train 5 models**:
   - Custom CNN from scratch
   - ResNet18 (transfer learning)
   - MobileNetV2 (lightweight)
   - VGG16 (deep architecture)
   - EfficientNet-B0 (state-of-the-art)
5. **Compare all models** with metrics and visualizations ✓
6. **Save best model** to `models/` directory ✓

## 4. Using the Trained Model

### Command-Line Inference

```bash
# Run prediction on an image
python src/inference.py --image path/to/image.jpg

# Save output with visualization
python src/inference.py --image path/to/image.jpg --output result.jpg

# Force CPU usage
python src/inference.py --image path/to/image.jpg --device cpu
```

### Python Script

```python
from src.inference import FaceMaskDetector

# Initialize detector
detector = FaceMaskDetector('models/best_model.pkl')

# Make prediction
result = detector.predict('path/to/image.jpg')

print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

## 5. Expected Results

After training, you should see:

- **Training metrics** for all 5 models
- **Comparison charts** (accuracy, loss, etc.)
- **Confusion matrices** for each model
- **Best model saved** with >95% accuracy

### Example Output:

```
Model Comparison on Test Set
================================================================================
           Model  Accuracy (%)  Precision (%)  Recall (%)  F1-Score (%)
      Custom_CNN         94.50          93.80       95.20         94.50
        ResNet18         97.20          96.90       97.50         97.20
    MobileNetV2          96.10          95.80       96.40         96.10
           VGG16         95.80          95.30       96.30         95.80
  EfficientNet_B0        97.80          97.50       98.10         97.80
================================================================================
```

## 6. Troubleshooting

### CUDA Out of Memory
```python
# In the notebook, reduce batch size
BATCH_SIZE = 16  # or 8
```

### Dataset Download Issues
```bash
# Verify Kaggle credentials
ls ~/.kaggle/kaggle.json

# Manual download alternative
kaggle datasets download -d andrewmvd/face-mask-detection
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Kernel Crashes
- Restart kernel and clear all outputs
- Check available RAM: `free -h`
- Close other applications

## 7. GPU vs CPU

### With GPU (NVIDIA CUDA):
- Training time: ~30-40 minutes
- Recommended batch size: 32

### CPU Only:
- Training time: ~2-3 hours
- Recommended batch size: 16
- Still works perfectly, just slower!

## 8. Next Steps

Once training is complete:

1. **Review results** in the notebook
2. **Test inference** with your own images
3. **Deploy** the model (web API, mobile app, etc.)
4. **Customize** hyperparameters for better results
5. **Add features** (video processing, face detection, etc.)

## 9. Project Structure

```
face-mask-detection/
├── notebooks/
│   └── face_mask_detection.ipynb  ← Start here!
├── models/
│   └── [trained models saved here]
├── src/
│   └── inference.py               ← Use for predictions
├── data/
│   └── [dataset auto-downloaded]
├── README.md                      ← Full documentation
├── requirements.txt               ← Dependencies
└── QUICKSTART.md                  ← This file
```

## 10. Tips for Success

✅ **Do:**
- Read markdown cells in the notebook for explanations
- Monitor GPU/CPU usage during training
- Experiment with hyperparameters
- Save your work frequently

❌ **Don't:**
- Skip cells in the notebook (run sequentially)
- Close the browser/VSCode during training
- Modify the dataset structure manually
- Delete intermediate variables

---

**Need help?** Check the full README.md or open an issue.

**Ready to start?** Open `notebooks/face_mask_detection.ipynb` and run! 🚀
