# Counterfeit-bills-detection

![Texte alternatif](https://github.com/MelvinDerouck/Counterfeit-bills-detection/blob/main/header_Détecteurs-de-faux-billet-desktop.jpg)

## Project Overview:
This project focuses on deploying an algorithm for detecting counterfeit banknotes based on dimensional data (6 metrics) of the bills. The dataset was divided into two parts, one labeled as genuine (True) and the other as counterfeit (False), enabling the assessment of authenticity. Various machine learning techniques were employed for data preprocessing and predictive analysis.

## Methods Used:
 -**Multiple Linear Regression and k-NN (k-Nearest Neighbors)**:
Utilized for imputing missing values in the dataset.

-**Logistic Regression & k-Means with Principal Component Analysis (PCA)**:
Applied for predictive analysis and classification of results.

## Results:
The logistic regression model was chosen because its performance is slightly superior to that of the K-means model (see F1 scores and accuracy scores). A Python script was created to perform the necessary preprocessing for predicting the authenticity of banknotes from a source file.

## Conclusion:
This project demonstrates a comprehensive approach to banknote authenticity detection, integrating various machine learning techniques for preprocessing and predictive analysis. The logistic regression contributes to a robust and accurate model for distinguishing between genuine and counterfeit banknotes.

