# 2025-AOP-CNNKAN

## Overview
This project implements a deep learning model for predicting antioxidant proteins from amino acid sequences. The model combines multiple feature extraction methods and a neural network architecture to classify proteins as either antioxidant or non-antioxidant.

## Features
- Multiple feature extraction methods:
  - Binary Profile Features (BPF)
  - Dipeptide Composition (DPC)
  - K-mer Composition
  - Pseudo Amino Acid Composition (PAAC)

### Project Structure
```
source_code/
├── dataset/
│   ├── Anti-protein.txt
│   └── Non-Anti-protein.txt
├── feature_encoder/
│   ├── BPF.py
│   ├── DPC.py
│   ├── K_mer.py
│   └── PAAC.py
├── modules/
│   ├── model.py
│   ├── kan.py
│   ├── cnn.py
├── process_data.py
├── training.py
└── requirements.txt
```

### Requirements
```bash
pip install -r requirements.txt
```

### Usage
Train the model:
```bash
python training.py
```

The model will:
- Process and encode the protein sequences
- Split data into training (70%), validation (10%), and test (20%) sets
- Train the model with early stopping
- Save the best model as 'best_model.pth'
- Display evaluation metrics on the test set
