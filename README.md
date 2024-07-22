# FIFA World Cup Grouping and Prediction

![Type](https://img.shields.io/badge/Type-Course_Project-yellow)
![Concepts](https://img.shields.io/badge/Concepts-Machine_Learning-blue)
![Language](https://img.shields.io/badge/Language-Python-red)
![Key Libraries](https://img.shields.io/badge/Libraries-Scikit_Learn,_Pandas-green)

In this work, we predict the FIFA World Cup matches and attempt to propose a method to generate ideal grouping of teams for a balanced tournament using past match results and rankings which can be used to faciliate a tournament end-to-end. In summary, we use the data to generate and extract relevant features, and then use multiple supervised techniques to predict winner of a match. We implement multiple algorithms and do a thorough comparison of each of them. Apart from real data, we also explore creating fictitious matches and use semi-supervised learning in an attempt to improve the models. Alongside match predictions, we also use unsupervised clustering techniques to create groups that can facilitate a good tournament. Through these two processes, we create an end-to-end tool that can take in participating teams, build groups, predict results of matches and ultimately, predict a complete tournament.

Please find the full report which contains detailed description and results here - [Report](https://neelabhsinha.github.io/gatech-cs-7641-project-page/)

Team Members - Ananya Sharma, Snigdha Verma, Apoorva Sinha, Yu-Chen Lin, Neelabh Sinha

---

## Datasets 

- [Soccer World Cup Data (Kaggle)](https://www.kaggle.com/datasets/shilongzhuang/soccer-world-cup-challenge/)
- [All International Matches (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?select=results.csv)
- [FIFA World Rankings (Kaggle)](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)

## Results on Base Models

We analyze the performance of the various classification schemes on our dataset using all standard metrics of evaluating a supervised learning algorithm. We primarily want to optimize the accuracy of the prediction, but since we have a balanced dataset and no bias towards avoiding either true negatives or false positives, we treat them fairly. Our final results are as shown below:

| Technique          | Accuracy | Precision | Recall | F-1 score | ROC-AUC |
| ------------------ | -------- | --------- | ------ | --------- | ------- |
| Logistic Regression | 73.49%   | 73.59%    | 73.49% | 73.49%    | 0.81    |
| SVM                | 73.38%   | 73.51%    | 73.38% | 73.37%    | 0.81    |
| Decision Tree      | 70.29%   | 70.46%    | 70.29% | 70.26%    | 0.77    |
| kNN                | 71.41%   | 71.51%    | 71.41% | 71.40%    | 0.80    |
| Gaussian Naive Bayes        | 72.54%   | 72.54%    | 72.54% | 72.53%    | 0.80    |
| Random Forest                | 71.69%   | 71.81%    | 71.69% | 71.68%    | 0.80    |
| Gradient Boosting            | 71.52%   | 72.06%    | 71.52% | 71.41%    | 0.80    |
| Adaptive Boost (base model - logistic regression) | 73.16%   | 73.30%    | 73.16% | 73.14%    | 0.81    |

Please review the report linked above for a detailed result and other experiments conducted including using synthetic data and self-supervised learning.

## Usage Guide

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/neelabhsinha/fifa-world-cup-prediction-ml.git
    cd fifa-world-cup-prediction-ml
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

### Arguments

To run the script, you need to use Python's `argparse` to specify the task and various options. Below are the tasks and arguments available.

#### Tasks

- `train`: Train a machine learning model.
- `preprocess`: Preprocess the data and generate features.
- `simulate_tournament`: Simulate a FIFA World Cup tournament.
- `generate_artificial_data`: Generate artificial match data.

#### Arguments

- `--task`: Specifies the task to perform. Choices are `train`, `preprocess`, `simulate_tournament`, and `generate_artificial_data`.
- `--model`: Specifies the model to use for training and prediction. Choices are defined in the `models` list.
- `--tune`: If provided, tunes the hyperparameters for the model.
- `--do_pca`: If provided, performs PCA on the data (optional).
- `--select_features`: If provided, generates features from the data automatically.
- `--simulate_tournament`: If provided, simulates a tournament end-to-end (task is not required in this case).
- `--use_artificial_data`: If provided, uses artificial data for training. If used, generate the artificial match data

#### Example Commands

To train a model, use the following command:

```sh
python script_name.py --task train --model model_name --do_pca --select_features --use_artificial_data
```

To preprocess data and generate features, 

```sh
python script_name.py --task preprocess
```

To generate artificial match data for self-supervised learning, do the following -

```sh
python script_name.py --task generate_artificial_data
```
