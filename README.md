# CCIR 2020 Competition 
The question of the 26th China Conference on Information Retrieval in 2020<br>
[For more detail](https://www.datafountain.cn/competitions/423)

### Introduction
__Sentiment Analysis of Chinese Netizen during the COVID-19 Pandemic__<br>
Given 100,000 labeled Weibo ID and content as the training set, we want you to develop your own classifiers to predict the sentiment labels (positive, negative or neutral) of 10,000 test sets.<br>
Also, we provided additional 900,000 unlabeled corpus for further reserach.

### Offical Platform: DataFountain

### Result: 
F1-score:0.725<br>
Top 11% through all the teams.

### Abstract:
1. We chose the BERT model which was already trained by Google, then do the fine-tunning work. 
2. We considered a single weibo context as a 'sentence', then putting into BERT, which can do the single sentence classification task. 
3. Fianlly, we trained the 10-fold validation model and combined all the classifiers using a voting strategy to get best results.

### Improvements
1. By the limitaion of GPU memory, we did not use the 900,000 unlabeled data to build the percise model.
2. The labeled Weibo contents have unbalanced distribution, which made our model not stable enough.
