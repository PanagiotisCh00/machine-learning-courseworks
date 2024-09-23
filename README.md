# Machine Learning Courseworks
Project Overview

This repository contains two machine learning coursework projects completed for the Introduction to Machine Learning course at Imperial College London. The courseworks focus on implementing machine learning algorithms from scratch, including decision trees and artificial neural networks, and applying them to real-world tasks.
Coursework 1: Decision Tree Implementation

In this coursework, a decision tree algorithm is implemented, using Object Oriented Design in Python, from scratch to classify a user's location based on Wi-Fi signal strengths in a multi-room environment. The goal is to predict the room in which the user is standing based on continuous Wi-Fi signal features.
Coursework 2: Artificial Neural Networks for Regression

In this coursework, a neural network architecture is developed to perform a regression task predicting house prices using the California Housing Prices Dataset. The task involves creating a neural network mini-library using NumPy and building a regression model using either the mini-library or PyTorch.
Coursework 1: Decision Tree
Problem

The objective is to classify indoor locations based on Wi-Fi signal strength data collected from different rooms. The dataset consists of 2000 samples, each containing 7 Wi-Fi signal strength values and a room number (label). Both clean and noisy datasets are provided.
Key Features

    Decision Tree Algorithm: Implemented a recursive decision tree using information gain to find the best split points for continuous features.
    10-Fold Cross-Validation: Used 10-fold cross-validation to evaluate the model on both clean and noisy datasets.
    Pruning: Implemented post-pruning to improve generalization on noisy data.
    Metrics: Evaluated using confusion matrix, accuracy, recall, precision, and F1-scores.

Coursework 2: Artificial Neural Networks
Problem

The task is to predict the median house prices in California based on features such as location, population, and proximity to the ocean. The dataset contains both numerical and categorical data with missing values.
Key Features

    Neural Network Mini-Library: Implemented a neural network mini-library from scratch using NumPy. The library includes linear layers, activation functions, a training module, and a preprocessing module for normalizing data.
    PyTorch Implementation: Built and trained a regression model using PyTorch for predicting house prices.
    Hyperparameter Tuning: Performed hyperparameter optimization to improve model performance using a grid search approach.
    Evaluation Metrics: Used root mean square error (RMSE) and R2-score to evaluate the model's performance.
