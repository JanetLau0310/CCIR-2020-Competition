# CCIR 2020 Competition 
 Sentiment Analysis of Chinese Netizen during the COVID-19 Pandemic
Also the question of the 26th China Conference on Information Retrieval in 2020
[for more detail](https://www.datafountain.cn/competitions/423)

### Introduction
Given 100,000 labeled Weibo ID and content as the training set, we want you to develop your own classifiers to predict the sentiment labels (positive, negative or neutral) of 10,000 test sets.
Also, we provided additional 900,000 unlabeled corpus for further reserach.

### Offical Platform: DataFountain

### Abstract:
1. Using the provided labeled data, training the hyper-parameters with Bi-LSTM Model, and predict the results by the combination of mean pooling and max pooling.

2. Chose the BERT model trained by Google, then do the fine-tunning work. Considered a single weibo context as a 'sentence', then put it into the BERT model, gain the integration results after backpropagation process and 10-fold cross-validation.

### Result: 
F1-score:0.725<br>
Top 10% through all the teams.

### Improvements
1. By the limitaion of GPU memory, we did not use the 900,000 unlabeled data to build the percise model.
2. The labeled Weibo contents have unbalanced distribution, which made our model not stable enough.