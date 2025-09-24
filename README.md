**Text Detection and Text-Length Prediction from Images**

This project focuses on detecting and extracting text from images and predicting the length of the extracted text using machine learning algorithms. The workflow involves synthetic dataset generation, OCR-based text extraction, feature extraction, and classification.

**Project Overview**
The project aims to automatically detect and extract text from images and predict the length of the text. This is achieved through a pipeline that includes:
Synthetic dataset generation using Faker and Pillow
Text extraction using OCR (Tesseract)
Feature extraction from text and images
Application of multiple machine learning models for text-length prediction
This workflow can be extended to applications such as automated document analysis, CAPTCHA reading, and text-based image indexing.

**Key Features**
Synthetic text-image dataset creation with varied fonts, sizes, and backgrounds
OCR-based text detection and extraction
Feature engineering for text and image attributes
Multiple machine learning models for classification and regression of text length
Evaluation and prediction on new unseen images

**Dataset Generation**
Faker Library: Generates synthetic text data.
Pillow: Creates images with different backgrounds, fonts, and sizes.
Dataset Structure:
_  dataset/
    images/
      img_001.png
      img_002.png
    labels.csv_
labels.csv contains the ground-truth text for each image.
Variations Included:
Different font styles and sizes
Varied background colors and patterns
Randomized text content

**Text Detection and Extraction**
OCR Tool Used: Tesseract OCR
Process:
  Load image
  Apply preprocessing (grayscale, thresholding)
  Detect and extract text using Tesseract
  Store extracted text for feature engineering

**Feature Extraction**
The following features are extracted from the text and image:
Length of the extracted text
Font characteristics (size, style)
Image quality metrics (resolution, noise)
These features are then used as inputs for the classification algorithms.

**Classification Algorithms**
The project implements several models for predicting text length:
Naive Bayes: Probabilistic classifier using Bayes’ theorem to predict text-length categories
Linear Regression: Predicts numerical text length from extracted features
Logistic Regression: Classifies text length into discrete categories such as short, medium, or long
ID3 Decision Tree: Predicts text length based on decision rules from features
Find S & Candidate Algorithms: Generate hypothesis spaces and select the best candidate model

**Prediction**
After training, models predict text length for new unseen images.
  The output includes:
  Predicted text length
  Text length category (short/medium/long)

**Installation**
Clone the repository and install the required dependencies:
  git clone <repository-url>
  cd text-detection-length-prediction
  pip install -r requirements.txt

**Usage**
Generate Dataset:
python dataset_generator.py

Extract Text from Images:
python text_extraction.py

Train Models:
python train_models.py

Predict Text Length:
python predict_text_length.py --image_path "dataset/images/img_001.png"

**Dependencies**
Python 3.x
pillow
faker
pytesseract
numpy
pandas
scikit-learn

**Results**
Performance metrics (accuracy, mean absolute error) for each model
Confusion matrix for classification models
Sample predictions on new images

**Future Work**
Integrate deep learning models for improved text detection
Handle handwritten text using specialized OCR
Expand dataset with real-world images for better generalization
Add multilingual text support

**License**
This project is licensed under the MIT License.
