# Report Structure

## Introduction
- Describe the motivation
- Summary of the work done

## Related Work
- Describe the related work, add references

## Method Overview
- Problem Formulation
- Overall Pipeline

### Match Prediction
- Feature Extraction
- Feature Selection - PCA, F/W Feature Selection
- Algorithms: Logistic Regression, SVM, Decision Trees, Random Forest
- Semi-supervised learning approach
- Evaluation Metrics

### Groupings
TBA after midsem

## Implementation Details
- Describe the data (add EDA figures)
- Discussion

### Match Prediction
- Data selection (removing the data of other tournaments)
- Hyperparameter tuning
- Train test split
- Experimental setup

### Groupings
TBA after midsem

## Experiments
### Match Prediction
1) Results
   1) Table with precision, recall, accuracy, f1-score, AUC
   2) Figure containing confusion matrix
   3) Figure containing ROC curve
   4) Figure containing learning curve
   5) For logistic regression - results as a function of C
   6) For decision tree - decision tree
   7) For SVM - results as a function of C, results as a function of gamma, results with linear and rbf kernel
   5) Discussion
2) Impact of Ensemble Learning - refer figure from before section and describe overfitting scenarios of ensemble learning approach
3) Impact of Semi-supervised Learning (pick one ensemble and one simple model to compare)
   1) Table with precision, recall, accuracy, f1-score, AUC
   2) Figure containing confusion matrix
   3) Discussion
4) Impact of Feature Selection - pick two models for this approach and all metrics in section 1
   1) without anything
   2) with PCA
   3) with feature selection
   4) Discussion

### Groupings
TBA after midsem

## Tournament Prediction for 2022 World Cup
1) Describe tournament structure
2) Describe how we are going to predict the winner
3) Show tabulated results
4) Show visualization of results for tournament outcomes with different methods

## Future Work
- describe post-midsem work