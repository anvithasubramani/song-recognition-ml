# Automatic Song Recognition from Hummed and Whistled Audio

A machine learning project that explores automatic song identification from short audio recordings of people humming or whistling familiar melodies.

## Academic Achievement

**Result: Merit**

Completed as part of the MSc Data Science programme at Queen Mary University of London. I achieved a **Merit for this individual mini-project**.

## Project Overview

The objective of this project was to develop a machine learning model capable of predicting the title of a song from a 10-second audio recording of a person humming or whistling its melody.

Rather than relying on the original studio recording, the project investigates whether meaningful audio features can be extracted from human-generated melodies and used to classify them into their corresponding song categories.

The project involved audio preprocessing, feature extraction, dimensionality reduction, machine learning model development and performance evaluation.

## Dataset

**Dataset:** MLEnd Hums and Whistles II

- Dataset subset: 400 audio samples
- Dataset size: Approximately 1.09 GB
- Audio duration: 10 seconds per recording
- Categories: 8 song classes
- Samples per category: 50

**Dataset source:** [MLEnd Hums and Whistles II – GitHub](https://github.com/thekmannn/MLEndHW_Sample)

The dataset is not included in this repository because of its size. Please download the 400-sample dataset from the original source before running the notebook.

## Methodology

The project followed a machine learning workflow consisting of the following stages:

### 1. Audio Data Processing

Audio recordings were processed using Python audio analysis libraries to prepare them for feature extraction and classification.

### 2. Feature Extraction

Using Librosa, audio signals were transformed into numerical features that capture different characteristics of the hummed and whistled melodies.

The extracted features included:

- Mel-Frequency Cepstral Coefficients (MFCCs)
- MFCC delta features
- Chroma features
- Chroma delta features
- Spectral centroid
- Spectral rolloff
- Spectral bandwidth
- Spectral contrast
- Zero-crossing rate
- Tempo

A total of 87 audio features were used in the modelling pipeline.

### 3. Feature Scaling and Dimensionality Reduction

- StandardScaler was applied to standardise the extracted features.
- Principal Component Analysis (PCA) was used to reduce dimensionality.
- The feature space was reduced from 87 to 29 principal components.
- The selected components retained approximately 85.41% of the variance.

### 4. Model Development

Four machine learning classifiers were implemented and evaluated:

- Random Forest Classifier
- Gradient Boosting Classifier
- Linear Support Vector Machine (SVM)
- Radial Basis Function (RBF) Support Vector Machine

### 5. Model Evaluation

The dataset was divided using a stratified 80:20 training and validation split:

- Training samples: 320
- Validation samples: 80

Model performance was examined using training accuracy, validation accuracy and cross-validation results.

## Model Performance

The following results were obtained from the notebook's primary modelling pipeline:

| Model | Training Accuracy | Validation Accuracy |
|---|---:|---:|
| Random Forest | 70.31% | 18.75% |
| Gradient Boosting | 79.06% | 18.75% |
| Linear SVM | 51.56% | 27.50% |
| RBF SVM | 56.56% | 23.75% |

**Best validation performance: Linear SVM – 27.50%.**

The Linear SVM achieved the highest validation accuracy among the four evaluated models. The results also highlighted the difficulty of classifying hummed and whistled audio recordings using a relatively small dataset.

## Key Findings

- Audio feature extraction enabled the conversion of hummed and whistled recordings into numerical representations suitable for machine learning.
- MFCC and chroma features provided useful information for representing audio characteristics.
- PCA reduced the feature space while retaining approximately 85.41% of the original variance.
- Linear SVM achieved the highest validation accuracy among the evaluated classifiers.
- The difference between training and validation performance for the tree-based models highlighted potential overfitting.
- The results demonstrated the challenges associated with recognising melodies from variations in human humming and whistling.

## Limitations and Future Improvements

The validation accuracy indicates that there is considerable scope for improvement.

Potential future improvements include:

- Increasing the size and diversity of the training dataset.
- Exploring additional audio feature extraction techniques.
- Experimenting with different feature selection and dimensionality reduction methods.
- Hyperparameter optimisation and systematic model tuning.
- Exploring deep learning approaches for audio classification.
- Improving generalisation across different humming and whistling styles.

## Technologies and Tools

- Python
- Google Colab
- Jupyter Notebook
- Librosa
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Principal Component Analysis (PCA)
- Support Vector Machines (SVM)
- Random Forest
- Gradient Boosting

## Repository Structure

```text
song-recognition-ml/
│
├── Song_recognition.ipynb
├── song_recognition.py
└── README.md
```

*Note: The Python source file is optional if you are maintaining the project primarily as a Jupyter Notebook.*

## How to Run the Project

1. Clone this repository:

   ```bash
   git clone https://github.com/anvithasubramani/song-recognition-ml.git
   ```

2. Navigate to the project directory:

   ```bash
   cd song-recognition-ml
   ```

3. Download the 400-sample MLEnd Hums and Whistles II dataset from the [original dataset repository](https://github.com/thekmannn/MLEndHW_Sample).

4. Extract the dataset and update the notebook's dataset path to match your local directory.

5. Install the required libraries:

   ```bash
   pip install numpy pandas matplotlib scikit-learn librosa jupyter
   ```

6. Open and execute the notebook:

   ```bash
   jupyter notebook Song_recognition.ipynb
   ```

## Author

**Anvitha Priyaa Vellal Subramani**

MSc Data Science | Queen Mary University of London

- GitHub: [anvithasubramani](https://github.com/anvithasubramani)
- LinkedIn: [Anvitha Priyaa Subramani](https://www.linkedin.com/in/anvitha-subramani-717356305/)

---

*This project reflects my academic work in machine learning, audio processing, feature engineering, dimensionality reduction and predictive modelling.*
