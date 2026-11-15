
# Automatic Song Recognition from Hummed and Whistled Audio

A machine learning project exploring whether a song can be identified from a 10-second recording of a person humming or whistling its melody.

## Project Overview

This project uses the MLEnd Hums and Whistles II dataset to classify audio recordings into eight song categories.

The pipeline combines audio signal processing, handcrafted feature extraction, dimensionality reduction and supervised machine learning.

## Dataset

- Dataset: MLEnd Hums and Whistles II
- Samples: 400
- Song categories: 8
- Samples per category: 50
- Audio duration: 10 seconds

The dataset can be obtained from:
https://github.com/thekmannn/MLEndHW_Sample

## Methodology

1. Audio loading and waveform exploration using Librosa.
2. Feature extraction using MFCCs, chroma, spectral descriptors, zero-crossing rate and tempo.
3. Feature standardisation using StandardScaler.
4. Dimensionality reduction using Principal Component Analysis (PCA).
5. Model training and comparison.
6. Evaluation using validation accuracy, cross-validation, classification reports and confusion matrices.

## Machine Learning Models

- Random Forest
- Gradient Boosting
- Linear Support Vector Machine (SVM)
- RBF Support Vector Machine (SVM)

## Results

| Model | Training Accuracy | Validation Accuracy |
|---|---:|---:|
| Linear SVM | 51.56% | 27.50% |
| RBF SVM | 56.56% | 23.75% |
| Random Forest | 70.31% | 18.75% |
| Gradient Boosting | 79.06% | 18.75% |

PCA reduced the feature space from 87 to 29 components while retaining approximately 85% of the variance.

Linear SVM achieved the highest validation accuracy among the evaluated models.

## Key Learning Outcomes

- Audio signal processing and feature engineering.
- Multi-class classification.
- Feature scaling and dimensionality reduction.
- Model comparison and cross-validation.
- Identifying overfitting and interpreting classification errors.

## Limitations and Future Improvements

The dataset is relatively small, and humming and whistling recordings vary in pitch, tempo, recording quality and individual performance.

Future improvements could explore:
- Mel-spectrogram-based CNN architectures.
- Audio data augmentation.
- Pre-trained audio embeddings such as YAMNet or OpenL3.
- Improved speaker-independent validation.

## Technologies

Python · Google Colab · Librosa · NumPy · pandas · scikit-learn · PCA · SVM · Matplotlib · Seaborn
