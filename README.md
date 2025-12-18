# Stock Chart Pattern Classification using CNN

## 📊 Project Overview
Implementation of Convolutional Neural Network (CNN) for classifying stock chart patterns into three categories: Bullish, Bearish, and Sideways.

## 🎯 Objectives
- Build a CNN model for automated chart pattern recognition
- Achieve minimum 75% accuracy on test data
- Create a tool for technical analysis assistance

## 📁 Dataset
- **Source**: Yahoo Finance API
- **Total Images**: 1000
- **Classes**: Bullish (334), Bearish (333), Sideways (333)
- **Split**: Train (800), Validation (100), Test (100)

## 🏗️ Model Architecture
Custom CNN with:
- 4 convolutional blocks with batch normalization
- Dropout regularization
- Global Average Pooling
- Dense layers with ReLU activation
- Output layer with softmax activation

## 📈 Results
### Test Set Performance
- **Accuracy**: 0.9500 (95.00%)
- **Loss**: 0.1987
- **Precision**: 0.9495
- **Recall**: 0.9400

### Error Analysis
- Total errors: 5/100 (5.00%)
- Most common error: sideways→bearish

## 🚀 How to Run
1. Mount Google Drive with dataset in `/content/drive/My Drive/dataset_images/`
2. Run the notebook sequentially (Weeks 1-4)
3. Model will be saved in `/content/drive/My Drive/stock_model_20251213_165451`

## 📂 Repository Structure
```
stock-chart-classification/
├── notebook.ipynb          # Main notebook
├── models/                 # Saved models
├── results/                # Evaluation results
└── README.md               # This file
```

## 👥 Team
- Aryaguna Nugraha Passulleri (231011022) - Data Engineer
- Evaleona Palembangan (231011028) - Model Engineer

## 📅 Timeline
- Week 1 (18-24 Nov): Setup & EDA
- Week 2 (25 Nov-1 Dec): Preprocessing
- Week 3 (2-8 Dec): Model Training
- Week 4 (9-15 Dec): Evaluation

---

*Last Updated: 2025-12-13 19:50:34*