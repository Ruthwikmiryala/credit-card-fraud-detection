Credit Card Fraud Detection

I built this project to explore how machine learning can be used to detect fraudulent credit card transactions. The main challenge was working with a highly imbalanced dataset, where fraudulent transactions make up only a very small portion of the total data.

I experimented with different classification algorithms, compared their performance, and selected Random Forest as my final model.

Project Overview

The model takes transaction details as input and predicts whether a transaction is legitimate or potentially fraudulent.

During development, I worked on:

* Exploring and understanding the dataset.
* Cleaning the data and removing duplicate records.
* Handling class imbalance using SMOTE.
* Training and comparing different machine learning models.
* Tuning model parameters using RandomizedSearchCV.
* Finding a suitable classification threshold.
* Analysing feature importance.

Dataset

I used the Credit Card Fraud Detection dataset from Kaggle.

* Total transactions: 284,807
* Input features: 30
* Target column: Class
* 0 – Legitimate transaction
* 1 – Fraudulent transaction

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download creditcard.csv and place it inside the dataset/ directory.

Models Tested:

I experimented with four classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

I also experimented with SMOTE to understand how oversampling affects fraud detection performance.

After comparing the results and tuning Random Forest, I selected it as the final model.

Results

Final model performance on the test dataset:

Metric	    :Score
Precision	:95%
Recall	    :75%
F1-score	:84%
ROC-AUC     :0.9560
PR-AUC      :0.7812



The classification threshold was selected using validation data rather than the test set.

Project Structure

Fraud-Detection-System/
│
├── dataset/
├── models/
│   └── fraud_detection_rf.joblib
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Model_Training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore

Running the Project

1. Clone the repository

git clone https://github.com/Ruthwikmiryala
cd Fraud-Detection-System

2. Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate

3. Install the required libraries

pip install -r requirements.txt

4. Download the dataset

Place creditcard.csv inside the dataset folder.

5. Train the model

python src/train.py

6. Evaluate the model

python src/evaluate.py

7. Test predictions

python src/predict.py

Tools and Libraries

* Python
* Pandas and NumPy
* Matplotlib and Seaborn
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Joblib

Future Improvements

* Build an API to make predictions on new transactions.
* Experiment with other fraud detection techniques.
* Improve fraud detection recall while maintaining good precision.