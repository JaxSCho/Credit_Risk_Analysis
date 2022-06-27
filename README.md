# Credit Risk Analysis
## Project Overview - Loan Prediction Risk Analysis

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans and therefore, we'll employ different techniques to train and evaluate models with unbalanced clases. Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, this report will evaluate the machine learning model performance of various resampling algorithms and ensemble classifiers and whether they should be used to predict credit risk.

## Results

This section describes the balanced accuracy score and the precision and recall scores of the following six machine learning models:
- Naive Random Oversampling
- Synthetic Minority Oversampling Technique (SMOTE Oversampling)
- Cluster Centeroids: Undersampling
- SMOTEENN: Combination (Over and Under) Sampling
- Balanced Random Forest Classifier
- Easy Ensemble AdaBoost Classifier

### Naive Random Oversampling

Using random oversampling algorithm, our model generated the following results:

- **Balanced accuracy score:** 0.6403
- **Precision scores:** low precision for high-risk (0.01) but high precision for low-risk (1.00)
- **Recall scores:** similar recall for high-risk (0.66) and low-risk (0.62)

![image](https://user-images.githubusercontent.com/99936542/175833083-c9b672e1-4c83-489a-92a8-e9b3cee2a112.png)

<b>Fig.1 - Native Random Oversampling Model Results </b> 

### Synthetic Minority Oversampling Technique (SMOTE Oversampling)

Using SMOTE resampling algorithm, our model generated the following results:

- **Balanced accuracy score:** 0.6515
- **Precision scores:** low precision for high-risk (0.01) but high precision for low-risk (1.00)
- **Recall scores:** similar recall for high-risk (0.61) and low-risk (0.69)

![image](https://user-images.githubusercontent.com/99936542/175833240-d17ad697-427d-465e-b446-13241559eb55.png)

<b>Fig.2 - SMOTE Oversampling Model Results </b> 

### Cluster Centeroids: Undersampling

Using Cluster Centeroids resampling algorithm, our model generated the following results:

- **Balanced accuracy score:** 0.5447
- **Precision scores:** low precision for high-risk (0.01) but high precision for low-risk (1.00)
- **Recall scores:** better recall for high-risk than low-risk (0.69 vs 0.40)

![image](https://user-images.githubusercontent.com/99936542/175833258-2fcb434b-9a8e-448e-8b5c-aa7587b31643.png)

<b>Fig.3 - Cluster Centeroids Model Results </b> 

### SMOTEENN (Synthetic Minority Oversampling Technique + Edited Nearest Neighbors): Combination (Over and Under) Sampling

Using SMOTEEN resampling algorithm, our model generated the following results:

- **Balanced accuracy score:** 0.6551
- **Precision scores:** low precision for high-risk (0.01) but high precision for low-risk (1.00)
- **Recall scores:** better recall for high-risk than low-risk (0.75 vs 0.56)

![image](https://user-images.githubusercontent.com/99936542/175833282-fb4fa407-1303-48fa-a34e-fbb06e047b05.png)

<b>Fig.4 - SMOTEENN Model Results </b> 

### Balanced Random Forest Classifier

Using Balanced Random Forest Classifier, our model generated the following:

- **Balanced accuracy score:** 0.7888
- **Precision scores:** low precision for high-risk (0.03) but high precision for low-risk (1.00)
- **Recall scores:** better recall for low-risk than high-risk (0.87 vs 0.70)

![image](https://user-images.githubusercontent.com/99936542/175833307-c96f15d5-6781-4f9f-a2dd-aba29587c79e.png)

<b>Fig.5 - Balanced Random Forest Classifier Model Results </b> 

### Easy Ensemble AdaBoost Classifier

Using Easy Ensemble AdaBoost Classifier, our model generated the following:

- **Balanced accuracy score:** 0.9316
- **Precision scores:** low precision for high-risk (0.09) but high precision for low-risk (1.00)
- **Recall scores:** high recall for both high-risk (0.92) and low-risk (0.94)

![image](https://user-images.githubusercontent.com/99936542/175833321-71f7a5ae-16f2-4588-a83d-4b660a6ef79c.png)

<b>Fig.6 - Easy Ensemble AdaBoost Model Results </b> 

## Summary

A summary of results all six credit risk models is provided in Figure 7 below. Due to the inherent class imbalance (i.e., 68,470 low-risk vs 347 high-risk credit card credit records in the dataset), we can assume the models will be much better at predicting good loans (i.e., low credit risk) than risky loans (i.e., high credit risk). Overall, all six models were quite poor at detecting true risky loans (i.e., low precision (<0.10) for high-risk applicants) and detected true good loans 100% of the time (i.e., precision score for low-risk was 1.00 for all models). 

![image](https://user-images.githubusercontent.com/99936542/175849602-db5d7a61-3db2-4c7a-9280-127a09c821fa.png)

<b>Fig.7 - Summary Results of All Credit Risk Models </b> 

Based on the above results, I would recommend the Easy Ensemble AdaBoost Classifier (EEC) model to use for determining credit risk. While all six models have poor precision results for high-risk loan applicants, the EEC model was able to predict 93% of the high-risk loan applicants accurately (i.e., balanced accuracy score), which is the highest balanced accuracy score of all six models. Furthermore, the EEC model is able to detect potentially risky loans 92% of the time (i.e., recall score for high-risk applicants). Although the EEC model would have a high number of false positives (i.e., false high-risk applicants), it is more important to detect potentially risky loans (i.e., high recall score for high-risk applicants); false positives can be ruled out by checking their credit. 
 
## Resources
- Data Source: LoanStats_2019Q1.csv
- Software: Python via Jupyter Notebook and Google Collab