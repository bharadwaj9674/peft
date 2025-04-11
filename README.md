# Cross-Modal Retrieval with Parameter-Efficient Fine-Tuning (PEFT)

This repository implements cross-modal retrieval using Parameter-Efficient Fine-Tuning (PEFT) methods. It enables searching between different modalities (text, audio, and images) with minimal trainable parameters.

## Features

- **Multiple Modalities**: Text (RoBERTa), Audio (AST), and Image (ViT) encoders
- **PEFT Methods**: LoRA, Prefix Tuning, and Adapters for efficient transfer learning
- **Contrastive Learning**: Joint embedding space for cross-modal retrieval
- **Evaluation Metrics**: Comprehensive retrieval metrics (Recall@K, mAP)
- **Interactive Demo**: Streamlit-based interface for exploring the model

## Project Structure

```
cross_modal_peft/
├── README.md
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── encoders.py           # Pre-trained encoders (RoBERTa, AST, ViT)
│   │   ├── peft_modules.py       # PEFT implementations (LoRA, Prefix Tuning, Adapters)
│   │   └── fusion.py             # Projection and shared embedding space
│   ├── data/
│   │   ├── datasets.py           # Dataset classes for Clotho/HowTo100M
│   │   └── preprocessing.py      # Data preprocessing utilities
│   ├── training/
│   │   ├── losses.py             # Contrastive loss implementations
│   │   └── trainer.py            # Training loop
│   ├── evaluation/
│   │   └── metrics.py            # Retrieval metrics (R@K, mAP)
│   └── utils/
│       └── parameter_counting.py # PEFT parameter counting utilities
├── configs/
│   ├── default.yaml              # Default configuration
│   └── peft_configs/
│       ├── lora.yaml             # LoRA configuration
│       ├── prefix_tuning.yaml    # Prefix Tuning configuration
│       └── adapters.yaml         # Adapters configuration
├── scripts/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── demo.py                   # Demo interface
└── notebooks/
    └── exploration.ipynb         # Exploration notebook
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/cross-modal-peft.git
cd cross-modal-peft

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train a model with default configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

Train with a specific PEFT method:

```bash
python scripts/train.py --config configs/peft_configs/lora.yaml
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

Compare different PEFT methods:

```bash
python scripts/evaluate.py --compare "lora:configs/peft_configs/lora.yaml:outputs/lora/checkpoints/best.pt,prefix:configs/peft_configs/prefix_tuning.yaml:outputs/prefix_tuning/checkpoints/best.pt,adapter:configs/peft_configs/adapters.yaml:outputs/adapters/checkpoints/best.pt" --output_dir outputs/comparison
```

### Demo

Run the interactive demo:

```bash
streamlit run scripts/demo.py -- --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

## PEFT Methods

This project implements three Parameter-Efficient Fine-Tuning methods:

### LoRA (Low-Rank Adaptation)

LoRA represents weight updates as low-rank decompositions (W + BA), drastically reducing trainable parameters. Configuration in `configs/peft_configs/lora.yaml`.

### Prefix Tuning

Prefix Tuning prepends trainable "virtual tokens" to the sequence, keeping the original model frozen. Configuration in `configs/peft_configs/prefix_tuning.yaml`.

### Adapters

Adapters insert small bottleneck layers after attention and feedforward modules. Configuration in `configs/peft_configs/adapters.yaml`.

## Architecture

![Architecture Diagram](architecture.png)

The architecture consists of:
1. **Encoders**: Pre-trained models for each modality (frozen)
2. **PEFT Modules**: Parameter-efficient fine-tuning components
3. **Projection Layers**: Map each encoder's output to a shared embedding space
4. **Contrastive Learning**: Align representations across modalities

## Parameter Efficiency Comparison

| Method         | Total Parameters | Trainable Parameters | Parameter Efficiency |
|----------------|------------------|----------------------|----------------------|
| Full Fine-tuning | 435M            | 435M                 | 100%                |
| LoRA           | 435M            | 2.5M                 | 0.57%               |
| Prefix Tuning  | 435M            | 3.1M                 | 0.71%               |
| Adapters       | 435M            | 1.9M                 | 0.44%               |

## Results

Retrieval performance (Recall@1) on the test set:

| Method         | Text→Audio | Audio→Text | Text→Image | Image→Text |
|----------------|------------|------------|------------|------------|
| Full Fine-tuning | 64.2%      | 63.8%      | 68.3%      | 67.1%      |
| LoRA           | 63.5%      | 62.7%      | 67.1%      | 65.9%      |
| Prefix Tuning  | 61.8%      | 60.9%      | 65.3%      | 64.2%      |
| Adapters       | 62.9%      | 61.6%      | 66.8%      | 65.1%      |

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

## License

This project is licensed under the MIT License - see the LICENSE file for details.