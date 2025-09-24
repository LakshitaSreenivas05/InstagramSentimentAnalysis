# Instagram Comment Sentiment Analysis using LSTM

This project aims to build a machine learning model that classifies Instagram comments into different sentiment categories: Positive, Neutral, and Negative. This project uses various NLP and deep learning techniques to process and analyze text data, helping understand user sentiment expressed in social media comments.

**Project Overview**

The project focuses on analyzing Instagram comments to determine the sentiment expressed by users. It uses a combination of traditional machine learning models and deep learning approaches to handle sequential text data. The workflow includes data collection, preprocessing, feature extraction, model training, evaluation, and sentiment prediction.

**Key Features**

* Cleaning and preprocessing of Instagram comments

* Tokenization and padding for sequence modeling

* Traditional machine learning models (Naive Bayes, Logistic Regression) using TF-IDF features

* Deep learning model (LSTM) capturing sequential dependencies in text

* Sentiment prediction for new, unseen Instagram comments

**Data Collection & Preprocessing**

* **Data Cleaning**: Comments are lowercased, mentions, hashtags, and URLs removed, and stopwords discarded.

* **Tokenization**: Converts words into numerical representations suitable for model input.

* **Padding**: Ensures uniform sequence length for compatibility with deep learning models.

**Modeling**

* **Naive Bayes**: Probabilistic classifier for text-based sentiment classification.

* **Logistic Regression**: Linear model predicting probabilities of sentiment classes.

* **LSTM (Long Short-Term Memory)**: Deep learning model capturing sequential context and dependencies in comments.

**Evaluation**

* Models are evaluated using accuracy and classification reports.

* The LSTM model provides better handling of sequential dependencies in text compared to traditional models.

**Sentiment Prediction**

* A function is implemented to predict the sentiment of new Instagram comments based on the trained models.

**Algorithms Used**

* **Naive Bayes**: Probabilistic model for text classification.

* **Logistic Regression**: Linear model predicting probabilities of sentiment classes.

* **LSTM**: Deep learning model capturing context and dependencies in text sequences.

**Installation**

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd instagram-sentiment-analysis
pip install -r requirements.txt
```

**Usage**

* **Preprocess Data**:

```bash
python preprocess_data.py
```

* **Train Traditional Models**:

```bash
python train_ml_models.py
```

* **Train LSTM Model**:

```bash
python train_lstm.py
```

* **Predict Sentiment for New Comments**:

```bash
python predict_sentiment.py --comment "Your comment text here"
```

**Dependencies**

* Python 3.x

* numpy

* pandas

* scikit-learn

* tensorflow / keras

* nltk / spaCy

**Results**

* Accuracy and classification reports for each model

* Comparison of traditional ML models vs. LSTM performance

* Sample predictions on new Instagram comments

**Future Work**

* Expand dataset with real Instagram comments for better generalization

* Implement multilingual support for comments in different languages

* Integrate attention mechanisms to improve LSTM performance

* Explore transformer-based models like BERT for sentiment analysis

**License**

This project is licensed under the MIT License.
