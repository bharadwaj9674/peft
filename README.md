# Cross-Modal Attention Network (CMAN)

A unified parameter-efficient framework for cross-modal retrieval across text, audio, and image modalities using transformer-based architectures with Low-Rank Adaptation (LoRA).

## 🎯 Key Features

- **Unified Architecture**: Single framework handles text, audio, and image modalities simultaneously
- **Parameter Efficiency**: 99.43% reduction in trainable parameters (2.5M out of 435M total)
- **Dynamic Processing**: Seamlessly handles single-modal, bi-modal, and tri-modal inputs
- **Balanced Performance**: Consistent retrieval performance across all six cross-modal directions
- **Fast Training**: Complete training in ~8 hours on a single GPU using mixed precision
- **Scalable Design**: Natural extension to additional modalities without architectural changes

## 🏗️ Architecture Overview

CMAN employs a unified fusion transformer architecture that processes multiple modalities through:

- **Modality-Specific Encoders**: RoBERTa-base (text), MIT/ast-finetuned-audioset (audio), google/vit-base-patch16-224 (images)
- **LoRA Adaptation**: Parameter-efficient fine-tuning with rank=8, α=16 applied to attention mechanisms
- **Unified Fusion Transformer**: 6 layers with 8 attention heads capturing cross-modal dependencies
- **Bidirectional Max-Margin Loss**: Hard negative mining for discriminative representation learning
- **512D Embedding Space**: Common representation space for all modalities

## 📋 Requirements

### System Requirements
- Python 3.9+
- CUDA-capable GPU (recommended: RTX A6000 or equivalent)
- 16GB+ RAM
- 50GB+ storage for datasets and models

### Dependencies
```bash
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
torchaudio>=2.0.0
numpy>=1.23.0
omegaconf>=2.3.0
librosa>=0.10.0
pillow>=9.5.0
scikit-learn>=1.2.0
streamlit>=1.24.0
tensorboard>=2.13.0
tqdm>=4.65.0
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/cross-modal-cman.git
cd cross-modal-cman
```

2. **Create virtual environment**
```bash
python -m venv cman_env
source cman_env/bin/activate  # On Windows: cman_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dataset Setup

Dataset Link - https://drive.google.com/file/d/1H_f-uHI8Z0C46dB6BMDQG-SzQq-mtvXh/view?usp=sharing

1. **Download Flickr8k Audio Dataset**
```bash
# Place your Flickr8k Audio dataset in the following structure:
data/flickr8k_audio/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── audio/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata/
    ├── train_metadata.json
    ├── val_metadata.json
    └── test_metadata.json
```

### Training

```bash
python run_train.py --config configs/minimal.yaml
```

**Training Configuration Options:**
- `--config`: Path to configuration file (default: configs/minimal.yaml)
- `--output_dir`: Custom output directory
- `--batch_size`: Override batch size
- `--epochs`: Override number of epochs

### Evaluation

```bash
python run_eval.py --config configs/minimal.yaml --checkpoint outputs/minimal/checkpoints/best.pt
```

**Evaluation Options:**
- `--config`: Configuration file path
- `--checkpoint`: Path to trained model checkpoint
- `--split`: Dataset split to evaluate (train/val/test)
- `--output_dir`: Custom evaluation output directory

### Interactive Demo

```bash
streamlit run run_demo.py -- --config configs/minimal.yaml --checkpoint outputs/minimal/checkpoints/best.pt
```

## 📁 Project Structure

```
cross-modal-cman/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default training configuration
│   └── minimal.yaml           # Minimal configuration for testing
├── src/                       # Source code
│   ├── models/
│   │   ├── encoders.py        # Text, Audio, Image encoders
│   │   ├── peft_modules.py    # LoRA implementation
│   │   └── cman.py            # Main CMAN architecture
│   ├── data/
│   │   └── data.py           # Dataset classes and data loading
│   ├── training/
│   │   └── trainer.py        # Training loop implementation
│   ├── evaluation/
│   │   └── evaluator.py      # Evaluation metrics and procedures
│   └── utils/
│       └── utils.py          # Utility functions
├── scripts/                   # Main execution scripts
│   ├── run_train.py          # Training script
│   ├── run_eval.py           # Evaluation script
│   └── run_demo.py           # Interactive demo
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## ⚙️ Configuration

Key configuration parameters in `configs/minimal.yaml`:

```yaml
# Model Configuration
model:
  embedding_dim: 512           # Common embedding space dimension
  dropout: 0.3

# PEFT Configuration  
peft:
  enabled: true
  method: "lora"
  rank: 8                      # LoRA rank
  alpha: 16                    # LoRA scaling factor
  dropout: 0.2

# Training Configuration
training:
  num_epochs: 70
  learning_rate: 5e-5
  weight_decay: 0.01
  batch_size: 32
  device: "cuda"
```

## 📊 Performance

### Parameter Efficiency

- **Total Parameters**: 435M
- **Trainable Parameters**: 2.5M (0.57%)
- **Parameter Reduction**: 99.43%
- **Training Time**: ~8 hours (single RTX A6000)

## 🔧 Advanced Usage

### Custom Dataset

To use your own tri-modal dataset:

1. **Prepare metadata files** in JSON format:
```json
[
  {
    "id": "sample_001",
    "caption": "A dog playing in the park",
    "audio_filename": "audio_001.wav",
    "image_filename": "image_001.jpg"
  }
]
```

2. **Update configuration**:
```yaml
data:
  root: "path/to/your/dataset"
  # ... other parameters
```

### Model Customization

**Modify fusion transformer layers**:
```yaml
model:
  fusion_layers: 6           # Number of transformer layers
  fusion_heads: 8            # Number of attention heads
  fusion_dim: 2048          # Feed-forward hidden dimension
```

**Adjust LoRA parameters**:
```yaml
peft:
  rank: 16                  # Higher rank = more parameters but potentially better performance
  alpha: 32                 # Scaling factor
  target_modules: ["query", "key", "value", "dense"]
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use smaller model variants

2. **Slow Training**
   - Ensure mixed precision is enabled
   - Check data loading bottlenecks
   - Verify GPU utilization

3. **Poor Performance**
   - Verify dataset alignment
   - Check preprocessing pipeline
   - Adjust learning rate and hyperparameters

### Debug Mode

Enable detailed logging:
```bash
python run_train.py --config configs/minimal.yaml --debug
```


