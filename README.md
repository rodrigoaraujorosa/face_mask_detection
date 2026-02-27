# Face Mask Detection Using Deep Learning

A comprehensive computer vision project that implements and compares 5 different deep learning models for face mask detection. This project uses PyTorch and achieves high accuracy in classifying whether a person is wearing a mask or not.

## 🎯 Project Overview

This project trains and evaluates multiple state-of-the-art deep learning models to detect face masks in images. The models are trained on the Face Mask Detection dataset from Kaggle and compared based on various performance metrics.

## 📊 Models Implemented

1. **Custom CNN** - Built from scratch with 3 convolutional blocks
2. **ResNet18** - Transfer learning with pre-trained weights
3. **MobileNetV2** - Lightweight model optimized for mobile devices
4. **VGG16** - Deep architecture with small filters
5. **EfficientNet-B0** - State-of-the-art efficient architecture

## 🗂️ Project Structure

```
face-mask-detection/
├── data/                    # Dataset directory (auto-downloaded)
├── models/                  # Saved trained models
├── notebooks/               # Jupyter notebooks
│   └── face_mask_detection.ipynb
├── src/                     # Source code modules
├── README.md
├── requirements.txt
└── LICENSE
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)
- VSCode with Jupyter extension (or Jupyter Lab)

### Setup Instructions

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/rodrigo/.openclaw/workspace/face-mask-detection/
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

## 📖 Usage

### Running the Jupyter Notebook

1. **Open VSCode and install the Jupyter extension** (if not already installed)

2. **Open the notebook:**
   - Navigate to `notebooks/face_mask_detection.ipynb`
   - Click "Select Kernel" and choose your Python environment

3. **Run the cells sequentially:**
   - The notebook is structured in sections with markdown documentation
   - First run will download the dataset automatically (may take a few minutes)
   - Training all 5 models may take 30-60 minutes depending on your hardware

### Notebook Sections

1. **Setup & Dependencies** - Import libraries and configure environment
2. **Data Loading** - Download and explore the dataset
3. **Data Preprocessing** - Image augmentation and train/val/test split
4. **Model Definitions** - Define all 5 model architectures
5. **Training** - Train each model with progress tracking
6. **Evaluation** - Compare models with metrics and visualizations
7. **Model Selection** - Choose and save the best performing model

## 📈 Results

The notebook generates comprehensive evaluation metrics for all models:

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

Example output (results may vary):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN | 94.5% | 93.8% | 95.2% | 94.5% |
| ResNet18 | 97.2% | 96.9% | 97.5% | 97.2% |
| MobileNetV2 | 96.1% | 95.8% | 96.4% | 96.1% |
| VGG16 | 95.8% | 95.3% | 96.3% | 95.8% |
| EfficientNet-B0 | 97.8% | 97.5% | 98.1% | 97.8% |

*Note: Actual results will be generated after training in the notebook*

## 🔧 Configuration

### Training Parameters

You can modify these parameters in the notebook:

- **Batch Size**: Default 32 (adjust based on GPU memory)
- **Learning Rate**: Default 0.001
- **Epochs**: Default 20 per model
- **Image Size**: 224x224 pixels
- **Train/Val/Test Split**: 70/15/15

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only (slower training)
- **Recommended**: 16GB RAM, NVIDIA GPU with 4GB+ VRAM
- **Storage**: ~2GB for dataset and models

## 🛠️ Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
- Reduce batch size in the notebook
- Close other GPU-intensive applications

**Issue: kagglehub authentication error**
- Ensure you have Kaggle credentials configured
- Run: `kaggle datasets download -d andrewmvd/face-mask-detection`

**Issue: Import errors**
- Verify all packages are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

**Issue: Notebook kernel crashes**
- Restart the kernel and clear outputs
- Check available RAM with `free -h`

## 📚 Dataset

This project uses the **Face Mask Detection** dataset from Kaggle:
- **Source**: andrewmvd/face-mask-detection
- **Classes**: With Mask, Without Mask
- **Format**: Images with annotations
- **License**: Community Dataset

The dataset is automatically downloaded via `kagglehub` when you run the notebook.

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- Add more model architectures (Vision Transformer, YOLO, etc.)
- Implement real-time video detection
- Create a web API for model inference
- Add data augmentation techniques
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset by Andrew Mvd on Kaggle
- PyTorch and torchvision teams
- Pre-trained models from ImageNet

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Built with ❤️ for the computer vision community**
