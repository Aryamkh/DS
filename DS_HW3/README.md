# Assignment 3 — Deep Learning: MLP · CNN · RNN · Transformers

## Quick Start
1. Open `HW3_Deep_Learning.ipynb` in Google Colab
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells: **Ctrl+F9**  (total ~60–90 min on T4)

## Framework & Datasets
| Section | Dataset | Task |
|---------|---------|------|
| MLP | Breast Cancer (sklearn) | Binary classification |
| MLP | California Housing (sklearn) | Regression |
| CNN | CIFAR-10 (torchvision) | 10-class image classification |
| RNN / Transformer | IMDB Sentiment (HuggingFace) | Binary sentiment |

## Structure
```
hw3/
├── HW3_Deep_Learning.ipynb   # Full notebook
├── README.md                 # This file
└── hw3_results/              # Auto-created (or Google Drive)
    ├── models/               # Saved .pth checkpoints
    ├── plots/                # All figures as PNG
    └── metrics/              # CSV/JSON metrics
```

## Section Summaries

### Section 1: MLP
- Binary classification (Breast Cancer) + Regression (California Housing)
- Experiments: optimizers, LR, scheduling, batch size, early stopping,
  depth, width, activations, weight init, BatchNorm, L1/L2/Dropout

### Section 2: CNN (CIFAR-10)
- Custom 3-block CNN from scratch
- Architecture experiments: kernel size, filters, depth, pooling
- Data augmentation (standard + strong)
- Transfer learning: ResNet18 feature extraction / partial / full fine-tune
- Feature map visualisations

### Section 3: RNN (IMDB Sentiment)
- Vanilla RNN · LSTM · GRU — all compared
- Experiments: sequence length, hidden size, stacked layers, bidirectionality, dropout

### Section 4: Transformer (IMDB Sentiment)
- PyTorch nn.TransformerEncoder with sinusoidal positional encoding
- Direct comparison vs RNN/LSTM/GRU on identical task
- Optional: DistilBERT fine-tuning cell included (set flag to run)

### Section 5 (Bonus): Industry Research
- Literature review of ML model usage in production
- 5–10 year prediction with domain-specific analysis

## Key Findings
| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| Baseline MLP | ~97% (BC) | Breast Cancer binary clf |
| Regression MLP | ~75% R² | California Housing |
| Baseline CNN | ~72% | CIFAR-10 from scratch |
| ResNet18 Full Fine-tune | ~82% | Transfer learning |
| Vanilla RNN | ~83% | IMDB sentiment |
| LSTM | ~87% | IMDB sentiment |
| GRU | ~87% | IMDB sentiment |
| Transformer (ours) | ~88% | IMDB sentiment |
| DistilBERT (optional) | ~92% | Pretrained 66M params |

## Results Summary

See [report.md](report.md) for the full analysis with plots and metric tables.

| Task | Best Model | Score |
|------|-----------|-------|
| Breast Cancer (clf) | MLP + Adam | 97.67% accuracy |
| California Housing (reg) | MLP | R² = 0.762 |
| CIFAR-10 (image clf) | ResNet18 Full Fine-tune | 90.25% test acc |
| IMDB Sentiment | GRU | 86.39% test acc |
| IMDB Sentiment (optional) | DistilBERT | ~92% test acc |

## Requirements
```
torch >= 2.0 · torchvision >= 0.15 · datasets >= 2.14
transformers >= 4.35 · scikit-learn >= 1.3
matplotlib · seaborn · pandas · numpy · tqdm
```
All installed automatically in Cell 2.
